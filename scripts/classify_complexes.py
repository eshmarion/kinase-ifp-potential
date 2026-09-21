"""CLI: соответствие комплексов условиям работы DiffSBDD.

Запуск:
    docker compose run --rm dev python scripts/classify_complexes.py --pdb 6tgu --ligand N92
    docker compose run --rm dev python scripts/classify_complexes.py --sample 60
    docker compose run --rm dev python scripts/classify_complexes.py --calibration

Отвечает на вопрос, который по пакету KLIFS не задать: что ещё занимало карман
в исходном кристалле. KLIFS отдаёт белок очищенным, поэтому источник — файлы RCSB;
скачанное складывается в кэш (`--cache`), и повторный прогон сети не требует.

Логики здесь нет — она в `src/kinase_ifp/pocket_content.py`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from kinase_ifp.config import (
    CALIBRATION_KINASES,
    DEFAULT_SPECIES,
    DIFFSBDD_POCKET_CUTOFF_A,
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
)
from kinase_ifp.klifs import (
    _is_present,
    filter_structures,
    read_fingerprint_table,
    select_best_per_kinase,
)
from kinase_ifp.pocket_content import PocketContentError, PocketVerdict, classify_structure
from kinase_ifp.reselect import reselect_calibration, reselect_target

ROOT = Path(__file__).resolve().parents[1]

CSV_COLUMNS = (
    "pdb_id",
    "expo_id",
    "pocket_class",
    "matches_diffsbdd",
    "radius",
    "qed",
    "ligand_atoms",
    "ligand_copies",
    "water_atoms",
    "neighbours",
    "reasons",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdb", nargs="*", help="коды структур PDB")
    parser.add_argument("--ligand", nargs="*", help="коды лигандов, по одному на структуру")
    parser.add_argument("--sample", type=int, help="случайная выборка из отфильтрованных KLIFS")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="выборка сверки: по одной лучшей структуре на киназу CALIBRATION_KINASES",
    )
    parser.add_argument(
        "--reselect-target",
        action="store_true",
        help="перевыбрать мишень: первая годная структура в порядке rank_structures",
    )
    parser.add_argument(
        "--reselect-calibration",
        action="store_true",
        help="перевыбрать выборку сверки: по одной годной структуре на киназу",
    )
    parser.add_argument("--seed", type=int, default=0, help="зерно выборки (по умолчанию 0)")
    parser.add_argument(
        "--radius",
        type=float,
        default=DIFFSBDD_POCKET_CUTOFF_A,
        help=(
            f"радиус кармана в ангстремах "
            f"(по умолчанию {DIFFSBDD_POCKET_CUTOFF_A} — как у DiffSBDD)"
        ),
    )
    parser.add_argument(
        "--no-qed", action="store_true", help="не считать QED (без обращения к компонентам)"
    )
    parser.add_argument("--out", type=Path, help="куда записать таблицу CSV")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "rcsb-cache")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def отобрать_выборку(root: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Случайная выборка из структур, прошедших фильтр, — по одной на код PDB."""
    таблица = pd.read_csv(root / "data" / "klifs" / "structures_all.csv")
    годные = таблица[
        (таблица["structure.resolution"] <= MAX_RESOLUTION)
        & (таблица["structure.qualityscore"] >= MIN_QUALITY_SCORE)
        & (таблица["species.klifs"] == DEFAULT_SPECIES)
        & таблица["ligand.expo_id"].map(_is_present)
    ].drop_duplicates(subset=["structure.pdb_id"])
    выборка = годные.sample(n=min(n, len(годные)), random_state=seed)
    return [
        (str(строка["structure.pdb_id"]), str(строка["ligand.expo_id"]).strip())
        for _, строка in выборка.iterrows()
    ]


def _структуры_и_эталоны(root: Path) -> tuple[pd.DataFrame, set[int]]:
    структуры = filter_structures(pd.read_csv(root / "data" / "klifs" / "structures_all.csv"))
    таблица = read_fingerprint_table(root / "data" / "klifs" / "klifs_ifp.csv")
    return структуры, set(int(i) for i in таблица["structure_id"])


def перевыбор_мишени(args: argparse.Namespace) -> None:
    """Первая структура в прежнем порядке, чей карман годен для модели."""
    структуры, эталоны = _структуры_и_эталоны(args.root)
    итог = reselect_target(
        структуры, эталоны, args.cache, radius=args.radius, with_qed=not args.no_qed
    )
    print("ПЕРЕВЫБОР МИШЕНИ — порядок rank_structures, первая годная")
    print(f"проверено кандидатов: {итог.checked}")
    if итог.chosen is None:
        print("годной структуры среди проверенных нет")
    else:
        строка = итог.chosen_row
        print(f"выбрана: {итог.chosen.pdb_id} ({итог.chosen.expo_id})")
        if строка is not None:
            print(
                f"  киназа {строка['kinase.klifs_name']}, разрешение "
                f"{строка['structure.resolution']}, качество {строка['structure.qualityscore']}"
            )
        прежняя = итог.changed_from
        print(f"  прежний первый кандидат: {прежняя or 'тот же самый'}")
    for отклонён in итог.rejected:
        print(f"  отвергнут {отклонён.summary()}")
    for строка_ошибки in итог.failed:
        print(f"  не проверен {строка_ошибки}")


def перевыбор_калибровки(args: argparse.Namespace) -> None:
    """По одной годной структуре на киназу выборки сверки."""
    структуры, эталоны = _структуры_и_эталоны(args.root)
    итоги = reselect_calibration(
        структуры,
        эталоны,
        CALIBRATION_KINASES,
        args.cache,
        radius=args.radius,
        with_qed=not args.no_qed,
    )
    print("ПЕРЕВЫБОР ВЫБОРКИ СВЕРКИ — по одной годной структуре на киназу")
    нашлось = [имя for имя, итог in итоги.items() if итог.chosen is not None]
    print(f"киназ с годной структурой: {len(нашлось)} из {len(итоги)}")
    print()
    for имя, итог in итоги.items():
        if итог.chosen is None:
            причины = "; ".join(о.pocket_class for о in итог.rejected) or "нет кандидатов"
            print(f"  {имя:<10} — годной нет (проверено {итог.checked}: {причины})")
            continue
        сменилась = итог.changed_from
        пометка = f" (было {сменилась})" if сменилась else ""
        print(f"  {имя:<10} {итог.chosen.pdb_id} {итог.chosen.expo_id}{пометка}")

    if args.out:
        записать(args.out, [и.chosen for и in итоги.values() if и.chosen is not None])
        print()
        print(f"таблица записана: {args.out}")


def выборка_калибровки(root: Path) -> list[tuple[str, str]]:
    """Те же структуры, на которых считается полная сверка.

    Отбор не переписывается, а зовётся: `select_best_per_kinase` — то же место, что
    в `calibrate_ifp.py`. Иначе состав выборки существовал бы в двух видах, и первое
    же расхождение осталось бы незамеченным.
    """
    структуры = filter_structures(pd.read_csv(root / "data" / "klifs" / "structures_all.csv"))
    таблица = read_fingerprint_table(root / "data" / "klifs" / "klifs_ifp.csv")
    выборка = select_best_per_kinase(структуры, CALIBRATION_KINASES, set(таблица["structure_id"]))
    return [
        (str(строка["structure.pdb_id"]), str(строка["ligand.expo_id"]).strip())
        for _, строка in выборка.iterrows()
    ]


def записать(путь: Path, вердикты: list[PocketVerdict]) -> None:
    путь.parent.mkdir(parents=True, exist_ok=True)
    with путь.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл, lineterminator="\n")
        писатель.writerow(CSV_COLUMNS)
        for в in вердикты:
            писатель.writerow(
                [
                    в.pdb_id,
                    в.expo_id,
                    в.pocket_class,
                    int(в.matches_diffsbdd),
                    в.radius,
                    "" if в.qed is None else f"{в.qed:.4f}",
                    в.ligand_atoms,
                    в.ligand_copies,
                    в.water_atoms,
                    ";".join(
                        f"{с.resname}:{с.atoms}" for с in в.neighbours if с.resname != "HOH"
                    ),
                    " | ".join(в.reasons),
                ]
            )


def main() -> None:
    args = parse_args()

    if args.reselect_target:
        перевыбор_мишени(args)
        return
    if args.reselect_calibration:
        перевыбор_калибровки(args)
        return

    пары: list[tuple[str, str]] = []
    if args.pdb:
        if not args.ligand or len(args.ligand) != len(args.pdb):
            raise SystemExit("--ligand должен перечислять по одному коду на каждую --pdb")
        пары = list(zip(args.pdb, args.ligand, strict=True))
    elif args.calibration:
        пары = выборка_калибровки(args.root)
    elif args.sample:
        пары = отобрать_выборку(args.root, args.sample, args.seed)
    else:
        raise SystemExit("нужен --pdb ... --ligand ..., --sample N или --calibration")

    вердикты: list[PocketVerdict] = []
    отказы: list[str] = []
    for pdb_id, expo_id in пары:
        try:
            вердикты.append(
                classify_structure(
                    pdb_id,
                    expo_id,
                    args.cache,
                    radius=args.radius,
                    with_qed=not args.no_qed,
                )
            )
        except PocketContentError as ошибка:
            отказы.append(f"{pdb_id} {expo_id}: {ошибка}")

    print(f"РАДИУС {args.radius} Å — карман DiffSBDD (dist_cutoff в process_bindingmoad.py)")
    print(f"разобрано структур: {len(вердикты)}, не удалось: {len(отказы)}")
    print()

    классы = Counter(в.pocket_class for в in вердикты)
    годных = sum(1 for в in вердикты if в.matches_diffsbdd)
    print(f"  соответствуют условиям DiffSBDD ... {годных} из {len(вердикты)}")
    for класс, число in классы.most_common():
        print(f"  {класс:<16} {число}")

    с_водой = sum(1 for в in вердикты if в.water_atoms > 0)
    print()
    print(f"  вода рядом с лигандом (в счёт не идёт) ... {с_водой} из {len(вердикты)}")

    негодные = [в for в in вердикты if not в.matches_diffsbdd]
    if негодные:
        print()
        print("  НЕ СООТВЕТСТВУЮТ:")
        for в in негодные:
            print(f"    {в.summary()}")

    if отказы:
        print()
        print("  НЕ РАЗОБРАНЫ:")
        for строка in отказы:
            print(f"    {строка}")

    if args.out:
        записать(args.out, вердикты)
        print()
        print(f"таблица записана: {args.out}")


if __name__ == "__main__":
    main()
