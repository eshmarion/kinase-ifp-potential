"""Приложение Modal: генерация лигандов DiffSBDD в кармане мишени.

Тонкий вход. Своей научной логики здесь нет: модель зовётся как библиотека из клона
в `external/` (чужой код не правится), подготовка лиганда берётся общей
функцией `kinase_ifp.protonate.prepare_ligand`, формат результата задан
форматами папки прогона и паспорта прогона.

Запуск — из контейнера через сервис `cloud`, у которого примонтирован токен:

    docker compose run --rm cloud modal run scripts/modal_sample.py --n 5
    docker compose run --rm cloud modal run scripts/modal_sample.py --n 100
    docker compose run --rm cloud modal run scripts/modal_sample.py --n 100 --suffix prep

Результат остаётся в томе `diffsbdd-runs`; на диск его забирает `scripts/pull_run.py`
 — результаты не должны существовать только в облаке.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

# Modal увозит в облако только сам файл точки входа, поэтому соседние модули из
# `scripts/` там не импортируются: удалённый контейнер падал с `No module named
# 'modal_app'`. Каталог со скриптами кладётся в образ ниже и добавляется в путь здесь —
# первый путь работает локально, второй в облаке.
PROJECT_SCRIPTS_REMOTE = "/opt/project/scripts"
PROJECT_SRC_REMOTE = "/opt/project/src"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, PROJECT_SCRIPTS_REMOTE)
sys.path.insert(0, PROJECT_SRC_REMOTE)

from modal_app import CHECKPOINT_NAME, CHECKPOINTS_PATH, checkpoints, image  # noqa: E402
from modal_targets import (  # noqa: E402
    RUNS_PATH,
    RUNS_VOLUME_NAME,
    TARGETS_PATH,
    resolve_target_dir,
    runs,
    targets,
)

from experiments.layout import (  # noqa: E402
    POCKET_MODE_POCKET_IDS,
    POCKET_MODE_REF_LIGAND,
    SOURCE_DIFFSBDD,
)
from experiments.run_io import (  # noqa: E402
    PLATFORM_MODAL_GPU,
    MoleculeRecord,
    create_run_dir,
    current_project_commit,
    make_run_id,
    write_failures,
    write_molecules,
    write_run_json,
)
from experiments.runs import target_stamp  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSBDD_DIR = PROJECT_ROOT / "external" / "DiffSBDD"


# Образ описан в modal_app и содержит только окружение DiffSBDD. Код проекта в него
# не запечён, поэтому `src/` домонтируется: без него в облаке нет общей функции
# протонирования, а перезаписывать папку прогона нельзя.
image_with_src = image.add_local_dir(
    str(PROJECT_ROOT / "src"), remote_path="/opt/project/src"
).add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path=PROJECT_SCRIPTS_REMOTE)

# Генерация идёт пачками: DiffSBDD по умолчанию берёт весь запрос одним батчем, а на T4
# сотня молекул в один заход не помещается по памяти.
DEFAULT_BATCH_SIZE = 25

# Чем задаётся карман при генерации (вопрос В-13). `ref-ligand` — окружением
# кристаллического лиганда, штатный способ из README DiffSBDD; `pocket-ids` — явным
# списком остатков KLIFS, то есть всем карманом киназы. Объёмы разные, поэтому способ
# пишется в паспорт: числа двух прогонов с разными карманами между собой несравнимы.


app = modal.App("kinase-ifp-sample")


@app.function(
    image=image_with_src,
    gpu="T4",
    volumes={CHECKPOINTS_PATH: checkpoints, TARGETS_PATH: targets, RUNS_PATH: runs},
    timeout=3600,
)
def sample(
    run_id: str,
    pdb_id: str,
    n_requested: int,
    seed: int,
    batch_size: int,
    timesteps: int | None,
    project_commit: str,
    diffsbdd_commit: str,
    pocket_mode: str = POCKET_MODE_REF_LIGAND,
    pocket_ids: list[str] | None = None,
    local_stamp: dict[str, object] | None = None,
) -> dict[str, object]:
    """Генерирует лиганды в кармане мишени и пишет прогон в обычной форме папки.

    `pocket_mode` задаёт, чем определяется карман (вопрос В-13): `ref-ligand` — окружением
    кристаллического лиганда, `pocket-ids` — явным списком остатков KLIFS. Способ пишется
    в паспорт прогона: карманы разного объёма дают несравнимые между собой числа.
    """
    import json
    import time

    sys.path.insert(0, "/opt/DiffSBDD")
    sys.path.insert(0, "/opt/project/src")

    import torch
    from openbabel import openbabel

    # DiffSBDD восстанавливает порядки связей по геометрии, и делает это через openbabel.
    # Тот печатает предупреждение на каждое кольцо, которое не удалось кекулизовать, плюс
    # ошибку про отсутствующий libXrender при загрузке своих плагинов. На сотне молекул
    # это десятки строк, за которыми не видно настоящей ошибки. Их собственный CLI
    # (`generate_ligands.py`) глушит лог ровно так же — это их штатный приём, а не
    # заметание проблем: молекулы, не прошедшие санитизацию, всё равно попадают
    # в `failures.csv` поимённо, и метрика validity считается по нему.
    #
    # Порядок важен: лог глушится до импорта `lightning_modules`, иначе openbabel
    # успевает загрузиться и напечатать свою ошибку о плагинах.
    openbabel.obErrorLog.StopLogging()

    from lightning_modules import LigandPocketDDPM
    from rdkit import Chem

    from kinase_ifp.protonate import ChargeNormalizationError, HydrogenError, prepare_ligand

    начало = time.monotonic()

    каталог_мишени = Path(TARGETS_PATH) / pdb_id
    пакет = json.loads((каталог_мишени / "target.json").read_text(encoding="utf-8"))

    # Модель кодирует тяжёлые атомы кармана. Водороды, пришедшие из конвейера KLIFS,
    # ей не нужны и ломают разбор типов атомов, поэтому на вход идёт белок без них,
    # а протонированный `protein.pdb` остаётся для расчёта отпечатка.
    белок = каталог_мишени / пакет["protein_noh_pdb"]
    эталонный_лиганд = каталог_мишени / пакет["ligand_sdf"]

    папка = create_run_dir(Path(RUNS_PATH) / run_id)

    # DiffSBDD не принимает сид аргументом, а `seed` обязателен в паспорте прогона:
    # без него повторить выборку нельзя.
    torch.manual_seed(seed)

    модель = LigandPocketDDPM.load_from_checkpoint(
        str(Path(CHECKPOINTS_PATH) / CHECKPOINT_NAME),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )
    модель = модель.to("cuda" if torch.cuda.is_available() else "cpu")

    # Пакет в томе обязан совпасть с тем, по которому список остатков построен локально:
    # иначе карман для генерации и карман для скора разъедутся молча.
    штамп = target_stamp(каталог_мишени / "target.json")
    if local_stamp is not None and local_stamp["package_sha256"] != штамп["package_sha256"]:
        raise RuntimeError(
            f"Пакет мишени в томе разошёлся с локальным: {штамп['package_sha256'][:12]} "
            f"против {str(local_stamp['package_sha256'])[:12]}. Перезалейте мишень: "
            "modal run scripts/modal_targets.py::upload_target --force"
        )

    # DiffSBDD принимает карман одним из двух взаимоисключающих способов (В-13).
    # Список остатков строится на локальной стороне и приезжает готовым: в образе
    # DiffSBDD нет ни MDAnalysis, ни pandas, а `kinase_ifp.pocket` тянет первый же
    # строкой импорта. 24.08 на этом прогон `pocket-ids` упал целиком.
    if pocket_mode == POCKET_MODE_POCKET_IDS:
        if not pocket_ids:
            raise ValueError(
                f"pocket_mode={POCKET_MODE_POCKET_IDS!r} требует непустой список остатков"
            )
        карман_аргумент = {"pocket_ids": pocket_ids}
    elif pocket_mode == POCKET_MODE_REF_LIGAND:
        карман_аргумент = {"ref_ligand": str(эталонный_лиганд)}
    else:
        raise ValueError(f"Неизвестный способ задать карман: {pocket_mode!r}")

    сгенерированные = []
    осталось = n_requested
    while осталось > 0:
        размер_пачки = min(batch_size, осталось)
        сгенерированные.extend(
            модель.generate_ligands(
                str(белок),
                размер_пачки,
                sanitize=False,
                timesteps=timesteps,
                **карман_аргумент,
            )
        )
        осталось -= размер_пачки

    отказы: list[tuple[str, str, str]] = []
    годные: list[MoleculeRecord] = []
    for номер, молекула in enumerate(сгенерированные, start=1):
        mol_id = f"{run_id}-{номер:04d}"
        if молекула is None:
            отказы.append((mol_id, "generate", "empty_molecule"))
            continue
        try:
            Chem.SanitizeMol(молекула)
        except Exception as ошибка:  # noqa: BLE001 — причина сохраняется, а не глотается
            отказы.append((mol_id, "sanitize", type(ошибка).__name__))
            continue
        try:
            готовая = prepare_ligand(молекула)
        except HydrogenError:
            # `no_explicit_hydrogens` — строка из соглашения о папке прогона.
            отказы.append((mol_id, "prepare", "no_explicit_hydrogens"))
            continue
        except ChargeNormalizationError as ошибка:
            # Срыв нормализации зарядов не равен отсутствию водородов и не должен
            # прятаться под чужой причиной: молекула, у которой не привелись заряды,
            # чинится иначе.
            отказы.append((mol_id, "prepare", f"charge_normalization_failed: {ошибка}"))
            continue
        # rmsd_to_ref не задан: у сгенерированной молекулы эталонной позы нет, и свойство
        # обязано присутствовать пустым.
        годные.append(MoleculeRecord(mol=готовая, mol_id=mol_id))

    записано = write_molecules(папка, годные)
    write_failures(папка, отказы)

    паспорт = json.loads(
        write_run_json(
            папка,
            run_id=run_id,
            target=штамп,
            source=SOURCE_DIFFSBDD,
            sampling={
                "n_requested": n_requested,
                "pocket_mode": pocket_mode,
                "n_returned": записано,
                "seed": seed,
                "guidance_scale": 0.0,
                "timesteps": timesteps,
            },
            duration_sec=round(time.monotonic() - начало),
            platform=PLATFORM_MODAL_GPU,
            model={"repo_commit": diffsbdd_commit, "checkpoint": CHECKPOINT_NAME},
            project_commit=project_commit,
        ).read_text(encoding="utf-8")
    )
    runs.commit()

    print(f"сгенерировано:  {len(сгенерированные)}")
    print(f"записано в SDF: {записано}")
    print(f"отказов:        {len(отказы)}")
    print(f"время, с:       {паспорт['duration_sec']}")
    return паспорт


def _ensure_free(run_id: str) -> None:
    """Отказывается запускать генерацию, если такой прогон в томе уже есть.

    В `run_id` по формату паспорта прогона нет ничего уникального, кроме даты, мишени, силы
    guidance и числа молекул, поэтому два одинаковых запуска в один день дают одно имя.
    Без этой проверки второй запуск дописал бы файлы поверх первого, а перезапись
    прогона запрещена: восстановить затёртый результат нечем.

    Проверка идёт на локальной стороне — до того, как Modal выделит GPU: платить
    за минуту, которая закончится отказом, незачем.

    **Пустой каталог занятым не считается.** Его оставляет вытеснение контейнера
    («Container terminated due to preemption»), после которого Modal перезапускает
    функцию с тем же входом. 16.09 такой каталог дважды подряд отменил прогон:
    сперва отказом внутри, потом этой проверкой. Результата в нём нет, терять нечего.
    """
    занято = {запись.path.strip("/") for запись in runs.listdir("/")}
    if run_id in занято and any(runs.listdir(run_id)):
        raise SystemExit(
            f"Прогон {run_id} уже есть в томе {RUNS_VOLUME_NAME} и не пуст, перезапись "
            "запрещена. Смените число молекул, дождитесь следующего дня или удалите "
            "прогон вручную, решив, что он не нужен"
        )


@app.local_entrypoint()
def main(
    n: int = 5,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timesteps: int = 0,
    target: str = "",
    suffix: str = "",
    pocket_mode: str = POCKET_MODE_REF_LIGAND,
) -> None:
    """Запускает генерацию и печатает паспорт полученного прогона.

    `pocket_mode` — `ref-ligand` или `pocket-ids` (вопрос В-13). Менять способ после
    того, как по прогонам посчитаны метрики, нельзя: карманы разного объёма дают
    несравнимые числа.

    `suffix` дописывается к `run_id` через дефис (формат паспорта прогона, вопрос В-15). Он нужен
    ровно тогда, когда прогон повторяют теми же параметрами: имя папки обязано быть
    уникальным, а перезапись запрещена. Первый такой случай — пересъёмка
    после перехода на `prepare_ligand`: параметры те же, а молекулы другие.
    """
    каталог = resolve_target_dir(target)
    # run_id считается локально и передаётся в облако: иначе проверка «не занято ли имя»
    # и построение имени могли бы разойтись на смене суток.
    run_id = make_run_id(
        каталог.name,
        n,
        0.0,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        suffix=suffix or None,
    )
    _ensure_free(run_id)

    if pocket_mode not in (POCKET_MODE_REF_LIGAND, POCKET_MODE_POCKET_IDS):
        raise SystemExit(
            f"--pocket-mode={pocket_mode!r}; допустимы "
            f"{POCKET_MODE_REF_LIGAND!r} и {POCKET_MODE_POCKET_IDS!r}"
        )

    # Список остатков и штамп пакета считаются здесь, где есть весь наш стек, а не
    # в облаке, где его нет.
    локальный_штамп = target_stamp(каталог / "target.json")
    остатки = None
    if pocket_mode == POCKET_MODE_POCKET_IDS:
        # Импорт внутри функции, а не в шапке: этот модуль целиком импортируется
        # и на удалённой машине, где `kinase_ifp.pocket` не поднимется — он тянет
        # MDAnalysis. Точка входа выполняется только локально, здесь стек есть.
        from kinase_ifp.pocket import residue_ids_from_mapping

        пакет = json.loads((каталог / "target.json").read_text(encoding="utf-8"))
        остатки = residue_ids_from_mapping(пакет["residue_to_position"])
        print(f"карман задан списком остатков KLIFS: {len(остатки)} шт.")

    паспорт = sample.remote(
        run_id=run_id,
        pdb_id=каталог.name,
        n_requested=n,
        seed=seed,
        batch_size=batch_size,
        # 0 означает «столько шагов, сколько у обученной модели»: у DiffSBDD это None.
        timesteps=timesteps or None,
        project_commit=current_project_commit(PROJECT_ROOT),
        # Чужой клон закреплён коммитом и не правится, поэтому спрашиваем только хэш.
        diffsbdd_commit=current_project_commit(DIFFSBDD_DIR, mark_dirty=False),
        pocket_mode=pocket_mode,
        pocket_ids=остатки,
        local_stamp=локальный_штамп,
    )
    print(f"прогон: {паспорт['run_id']}")
    print(f"забрать на диск: python scripts/pull_run.py --run-id {паспорт['run_id']}")
