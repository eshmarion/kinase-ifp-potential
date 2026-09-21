"""Приложение Modal: тома с данными — пакет мишени на вход и прогоны на выход.

Отдельный файл, а не ещё одна функция в `modal_app.py`: там образ строится из
`docker/Dockerfile.diffsbdd` и монтирует клон DiffSBDD, поэтому любой запуск требует
и `external/DiffSBDD` на месте, и сборки тяжёлого образа. Загрузке файлов в том не нужны
ни образ, ни GPU: она идёт с локальной машины, и в этом приложении нет ни одной удалённой
функции.

Порядок запуска — из контейнера, а не из PowerShell (консоль Windows не печатает символы,
которыми клиент Modal рисует отчёт, и падает на них, возвращая при этом код 0):

    docker compose run --rm dev modal run scripts/modal_targets.py::upload_target
    docker compose run --rm dev modal run scripts/modal_targets.py::list_target

Токен Modal задаётся `modal setup` и в репозиторий не попадает.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from kinase_ifp.config import PROJECT_ROOT, TARGETS_DIR

# Имя тома фиксировано: сэмплер ищет мишень именно здесь. Данные и веса
# лежат в разных томах, чтобы смена мишени не задевала чекпойнт.
TARGETS_VOLUME_NAME = "diffsbdd-targets"
TARGETS_PATH = "/targets"

# Файлы пакета мишени, без которых генерация в кармане невозможна.
REQUIRED_FILES = ("target.json", "protein.pdb", "ligand.sdf", "pocket.pdb")

targets = modal.Volume.from_name(TARGETS_VOLUME_NAME, create_if_missing=True)

# Том результатов. Отдельно от мишеней и весов: он один растёт с каждым прогоном,
# и только он монтируется на запись. Описан здесь, а не в `modal_sample.py`, чтобы
# выгрузка (`scripts/pull_run.py`) не тянула за собой образ DiffSBDD и клон в external/.
RUNS_VOLUME_NAME = "diffsbdd-runs"
RUNS_PATH = "/runs"
runs = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

app = modal.App("kinase-ifp-targets")


def resolve_target_dir(target: str) -> Path:
    """Находит каталог мишени: явно указанный или единственный в `data/targets`."""
    if target:
        каталог = Path(target)
        if not каталог.is_absolute():
            каталог = PROJECT_ROOT / каталог
        return каталог

    if not TARGETS_DIR.is_dir():
        raise SystemExit(
            f"Нет каталога {TARGETS_DIR}. Сначала соберите пакет мишени: "
            "scripts/fetch_klifs.py, затем scripts/select_target.py"
        )
    каталоги = sorted(путь for путь in TARGETS_DIR.iterdir() if путь.is_dir())
    if len(каталоги) != 1:
        raise SystemExit(
            f"В {TARGETS_DIR} мишеней не одна, а {len(каталоги)}: "
            "укажите нужную явно через --target"
        )
    return каталоги[0]


def _check_package(каталог: Path) -> dict[str, object]:
    """Проверяет состав пакета мишени и возвращает его `target.json`.

    Проверка идёт до загрузки, а не после: неполный пакет в томе выглядит как готовый
    и обнаруживается уже во время генерации, за GPU-минуты.
    """
    if not каталог.is_dir():
        raise SystemExit(f"Нет каталога мишени {каталог}")

    отсутствуют = [имя for имя in REQUIRED_FILES if not (каталог / имя).is_file()]
    if отсутствуют:
        raise SystemExit(
            f"В пакете {каталог} нет файлов: {', '.join(отсутствуют)}. "
            "Пакет мишени требует их все"
        )

    пакет: dict[str, object] = json.loads((каталог / "target.json").read_text(encoding="utf-8"))
    if not пакет.get("protonation"):
        raise SystemExit(
            f"В {каталог / 'target.json'} не заполнено поле 'protonation'. "
            "Запустите scripts/annotate_protonation.py"
        )
    return пакет


@app.local_entrypoint()
def upload_target(target: str = "", force: bool = False) -> None:
    """Кладёт каталог мишени в том `diffsbdd-targets`.

    Без `--force` уже загруженная мишень не перезаписывается: прогоны, сделанные
    на прежнем пакете, иначе стали бы невоспроизводимыми — их `run.json` ссылается
    на мишень по имени, а не по содержимому.
    """
    каталог = resolve_target_dir(target)
    пакет = _check_package(каталог)

    занято = {запись.path.strip("/") for запись in targets.listdir("/")}
    if каталог.name in занято and not force:
        raise SystemExit(
            f"Мишень {каталог.name} уже есть в томе {TARGETS_VOLUME_NAME}. "
            "Перезаписать: --force"
        )

    with targets.batch_upload(force=force) as загрузка:
        загрузка.put_directory(каталог, f"/{каталог.name}")

    print(f"мишень:      {пакет.get('pdb_id')} ({пакет.get('kinase_name')})")
    print(f"KLIFS ID:    {пакет.get('klifs_structure_id')}")
    print(f"протонация:  {пакет.get('protonation')}")
    print(f"том:         {TARGETS_VOLUME_NAME}, путь в контейнере {TARGETS_PATH}")
    _print_volume(каталог.name)


@app.local_entrypoint()
def list_target(target: str = "") -> None:
    """Печатает состав тома: «залилось ли» проверяется выводом, а не верой."""
    _print_volume(target)


def _print_volume(путь: str) -> None:
    записи = targets.listdir(f"/{путь}" if путь else "/", recursive=True)
    if not записи:
        print(f"том {TARGETS_VOLUME_NAME} пуст")
        return
    for запись in sorted(записи, key=lambda запись: запись.path):
        print(f"{запись.size:>10}  {запись.path}")
