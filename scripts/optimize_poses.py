"""Оптимизация поз прогона дискретным потенциалом взаимодействий.

    docker compose run --rm dev python scripts/optimize_poses.py --run runs/<id>
    docker compose run --rm dev python scripts/optimize_poses.py --run runs/<id> --all-types
    docker compose run --rm dev python scripts/optimize_poses.py --run runs/<id> --control
    docker compose run --rm dev python scripts/optimize_poses.py --run runs/<id> \
        --potential clash
    docker compose run --rm dev python scripts/optimize_poses.py --run runs/<id> \
        --potential weighted

Пишет таблицу по молекулам и сводку в `results/`. Прогон
не меняется: перезаписывать `runs/` нельзя, и уточнённые положения
остаются производным результатом, а не подменой исходных молекул.

Ключ `--potential` выбирает вариант построения потенциала (`kinase_ifp.potential`):
`discrete` — базовая формула как есть (М0), `clash` — плюс член стерического
клэша по образцу BInD (М1), `weighted` — биты взвешены частотой в базе KLIFS
по образцу PADIF (М2). Колонки `score_*` во всех трёх случаях считаются **базовой**
формулой, поэтому таблицы вариантов сравнимы между собой напрямую; то, что вариант
оптимизировал на самом деле, лежит в колонках `objective_*`.

GPU не нужен: это перебор поз, а не генерация.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import Chem  # noqa: E402

from experiments.layout import MOLECULES_SDF  # noqa: E402
from experiments.measurements import Measurement, upsert_measurements  # noqa: E402
from kinase_ifp.config import (  # noqa: E402
    CLASH_PENALTY_WEIGHT,
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    MEASUREMENTS_CSV,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.fingerprint_klifs import ifp_from_groups, load_pocket_rules  # noqa: E402
from kinase_ifp.ligand_flags import groups_from_mol  # noqa: E402
from kinase_ifp.molecule_io import open_sdf  # noqa: E402
from kinase_ifp.pose_optimization import (  # noqa: E402
    MAX_RMSD_A,
    optimize_pose,
    pocket_heavy_atoms,
    pocket_vdw_radii,
)
from kinase_ifp.potential import POTENTIAL_VARIANTS, make_potential  # noqa: E402
from kinase_ifp.similarity import select_types, tanimoto, tversky  # noqa: E402

РЕЗУЛЬТАТЫ = Path("results")


def эталон_мишени(package_json: Path) -> np.ndarray:
    """Эталонный отпечаток KLIFS из паспорта пакета, форма (7, 85)."""
    биты = json.loads(package_json.read_text(encoding="utf-8"))["klifs_ifp_bits"]
    массив = np.array([символ == "1" for символ in биты])
    return массив.reshape(KLIFS_IFP_SHAPE[1], KLIFS_IFP_SHAPE[0]).T


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__)
    разбор.add_argument("--run", type=Path, required=True, help="папка прогона")
    разбор.add_argument("--target", type=Path, default=Path("data/targets/6tgu/target.json"))
    разбор.add_argument("--steps", type=int, default=1000)
    разбор.add_argument("--max-rmsd", type=float, default=MAX_RMSD_A)
    разбор.add_argument(
        "--all-types",
        action="store_true",
        help="считать скор по всем семи типам, включая гидрофобные (иначе по шести)",
    )
    разбор.add_argument(
        "--control",
        action="store_true",
        help="отрицательный контроль: позиции эталона перемешиваются",
    )
    разбор.add_argument(
        "--potential",
        choices=POTENTIAL_VARIANTS,
        default="discrete",
        help="вариант построения потенциала: discrete - базовая формула (М0), "
        "clash - плюс член стерического клэша (М1), weighted - биты взвешены "
        "частотой в базе KLIFS (М2)",
    )
    разбор.add_argument(
        "--clash-weight",
        type=float,
        default=CLASH_PENALTY_WEIGHT,
        help="вес члена клэша у варианта clash",
    )
    разбор.add_argument("--seed", type=int, default=20260917)
    разбор.add_argument(
        "--measurements",
        type=Path,
        default=MEASUREMENTS_CSV,
        help="реестр измерений по мишеням: туда идёт сводка прогона",
    )
    аргументы = разбор.parse_args()

    типы = KLIFS_INTERACTION_TYPES if аргументы.all_types else SCORING_INTERACTION_TYPES
    эталон = эталон_мишени(аргументы.target)
    цель = эталон
    if аргументы.control:
        rng = np.random.default_rng(аргументы.seed)
        цель = эталон[:, rng.permutation(эталон.shape[1])]

    карман = load_pocket_rules(аргументы.target)
    атомы = pocket_heavy_atoms(аргументы.target)
    эталон_типов = select_types(эталон, типы)

    # Потенциал строится от **цели**, а не от эталона: в режиме контроля оптимизация
    # обязана идти к перемешанному отпечатку, иначе контроль перестал бы быть контролем.
    # Радиусы кармана нужны только варианту clash, но читаются всегда: файл тот же,
    # что у координат, и ветвление ради одного чтения дороже самого чтения.
    потенциал = make_potential(
        аргументы.potential,
        цель,
        scoring_types=типы,
        pocket_atoms=атомы,
        pocket_radii=pocket_vdw_radii(аргументы.target),
        clash_weight=аргументы.clash_weight,
    )

    начало = time.time()
    строки: list[dict[str, object]] = []
    # Оптимизированные позы нужны целиком, а не только числами: по ним считается
    # физичность (PoseBusters), и без файла проверку не повторить.
    позы: list[Chem.Mol] = []
    with open_sdf(аргументы.run / MOLECULES_SDF) as supplier:
        for номер, mol in enumerate(supplier):
            if mol is None:
                continue
            итог = optimize_pose(
                mol,
                карман,
                цель,
                атомы,
                steps=аргументы.steps,
                max_rmsd=аргументы.max_rmsd,
                seed=номер,
                scoring_types=типы,
                potential=потенциал,
            )
            имя = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol{номер:04d}"
            итог.mol.SetProp("_Name", имя)
            позы.append(итог.mol)
            до = ifp_from_groups(groups_from_mol(mol), карман)
            после = ifp_from_groups(groups_from_mol(итог.mol), карман)
            строки.append(
                {
                    "mol_id": mol.GetProp("mol_id") if mol.HasProp("mol_id") else f"{номер:04d}",
                    "score_before": round(
                        tversky(эталон_типов, select_types(до, типы), alpha=1.0, beta=0.0), 6
                    ),
                    "score_after": round(
                        tversky(эталон_типов, select_types(после, типы), alpha=1.0, beta=0.0), 6
                    ),
                    "tanimoto_before": round(tanimoto(до, эталон), 6),
                    "tanimoto_after": round(tanimoto(после, эталон), 6),
                    "hbond_before": int(до[3:5].any()),
                    "hbond_after": int(после[3:5].any()),
                    # Значение целевой функции варианта: у discrete совпадает
                    # со `score_*`, у clash и weighted расходится, и по этой паре
                    # видно, что именно вариант оптимизировал.
                    "objective_before": round(итог.objective_before, 6),
                    "objective_after": round(итог.objective_after, 6),
                    "rmsd_shift": round(итог.rmsd_shift, 4),
                    "min_distance_before": round(итог.min_distance_before, 4),
                    "min_distance_after": round(итог.min_distance_after, 4),
                    "steps_accepted": итог.steps_accepted,
                }
            )

    if not строки:
        print(f"в прогоне {аргументы.run} нет разобранных молекул", file=sys.stderr)
        return 1

    набор = "all7" if аргументы.all_types else "scoring6"
    метка = набор + ("-control" if аргументы.control else "")
    # Нештатный предел смещения попадает в имя файла: иначе прогон со свипом молча
    # затёр бы таблицу основного расчёта, и разница осталась бы незамеченной.
    # Так уже терялись числа — проверочный запуск на 200 шагах перезаписал 6tgu.
    if аргументы.max_rmsd != MAX_RMSD_A:
        метка += f"-r{аргументы.max_rmsd:g}"
    # Вариант потенциала попадает в метку по той же причине, что и предел смещения:
    # три варианта считаются на одном прогоне и одном наборе типов, поэтому без метки
    # второй расчёт молча затёр бы таблицу первого. Базовый вариант метки не получает,
    # чтобы имена прежних таблиц остались прежними и ссылки на них не сломались.
    if аргументы.potential != "discrete":
        метка += f"-{аргументы.potential}"
    if аргументы.potential == "clash" and аргументы.clash_weight != CLASH_PENALTY_WEIGHT:
        метка += f"-w{аргументы.clash_weight:g}"
    РЕЗУЛЬТАТЫ.mkdir(parents=True, exist_ok=True)
    таблица = РЕЗУЛЬТАТЫ / f"pose_optimization-{аргументы.run.name}-{метка}.csv"
    # CSV с '\n', а не с '\r\n': иначе файл, записанный кодом, отличался бы
    # в diff от того же файла, сохранённого git, во всех строках сразу.
    with таблица.open("w", encoding="utf-8", newline="\n") as поток:
        писатель = csv.DictWriter(поток, fieldnames=list(строки[0]), lineterminator="\n")
        писатель.writeheader()
        писатель.writerows(строки)

    sdf = таблица.with_suffix(".sdf")
    # SDWriter получает открытый поток, а не строку пути: путь он передаёт в C++
    # через ANSI и на Windows падает с «Bad output file» на любом пути с кириллицей
    #. Так же пишет молекулы `experiments.run_io.write_molecules`.
    with sdf.open("w", encoding="utf-8", newline="") as поток:
        писатель_sdf = Chem.SDWriter(поток)
        for поза in позы:
            писатель_sdf.write(поза)
        писатель_sdf.close()

    сд = np.array([с["score_before"] for с in строки], dtype=float)
    сп = np.array([с["score_after"] for с in строки], dtype=float)
    тд = np.array([с["tanimoto_before"] for с in строки], dtype=float)
    тп = np.array([с["tanimoto_after"] for с in строки], dtype=float)
    сдвиги = np.array([с["rmsd_shift"] for с in строки], dtype=float)
    дистанции = np.array([с["min_distance_after"] for с in строки], dtype=float)

    # Итоговые числа идут и в реестр измерений: по-молекульные таблицы лежат рядом,
    # но сводку по мишеням из них не собрать, не зная, какие файлы существуют. В реестре
    # прогон по одной мишени обновляет только свои строки, поэтому числа прежних мишеней
    # и прежних видов измерений остаются (`experiments.measurements`).
    мишень = json.loads(аргументы.target.read_text(encoding="utf-8"))["pdb_id"]
    # Вариант несёт ту же метку, что и имя файла таблицы, включая суффикс нештатного
    # предела смещения: иначе прогон свипа затёр бы в реестре строки основного расчёта —
    # ровно то, от чего метка защищает имя файла.
    вариант = f"{аргументы.run.name}:{метка}"
    од = np.array([с["objective_before"] for с in строки], dtype=float)
    оп = np.array([с["objective_after"] for с in строки], dtype=float)
    сводка = {
        "molecules": (str(len(строки)), ""),
        "score_before": (f"{сд.mean():.3f}", ""),
        "score_after": (f"{сп.mean():.3f}", ""),
        "improved": (str(int((сп > сд).sum())), ""),
        "objective_before": (f"{од.mean():.3f}", ""),
        "objective_after": (f"{оп.mean():.3f}", ""),
        "tanimoto_before_median": (f"{np.median(тд):.3f}", ""),
        "tanimoto_after_median": (f"{np.median(тп):.3f}", ""),
        "rmsd_shift_median": (f"{np.median(сдвиги):.3f}", "A"),
        "min_distance_after_median": (f"{np.median(дистанции):.3f}", "A"),
    }
    upsert_measurements(
        [
            Measurement(
                target=str(мишень),
                measurement="pose_optimization",
                variant=вариант,
                metric=имя,
                value=значение,
                unit=единица,
                measured_utc=datetime.now(timezone.utc).date().isoformat(),
                source=(
                    f"python scripts/optimize_poses.py --run runs/{аргументы.run.name} "
                    f"--steps {аргументы.steps} --max-rmsd {аргументы.max_rmsd}"
                ),
                notes="отрицательный контроль: позиции эталона перемешаны"
                if аргументы.control
                else "",
            )
            for имя, (значение, единица) in сводка.items()
        ],
        аргументы.measurements,
    )
    print(
        f"молекул {len(строки)}, {time.time() - начало:.0f} с, типов скора {len(типы)}, "
        f"потенциал {аргументы.potential}"
    )
    print(f"цель:     среднее {од.mean():.3f} -> {оп.mean():.3f}")
    print(f"скор:     среднее {сд.mean():.3f} -> {сп.mean():.3f}")
    print(f"Танимото: медиана {np.median(тд):.3f} -> {np.median(тп):.3f}")
    print(f"улучшилось {(сп > сд).sum()} из {len(строки)}")
    print(f"таблица: {таблица}")
    print(f"позы:    {sdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
