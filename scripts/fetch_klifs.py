"""CLI: выгрузка структур KLIFS с разметкой по типу и сводка по киназам.

Запуск:
    uv run python scripts/fetch_klifs.py --limit 20
    uv run python scripts/fetch_klifs.py --species all      # все организмы, не только человек

Создаёт три файла:
    data/klifs/structures_all.csv — все выгруженные структуры с колонкой inhibitor_type;
    data/klifs/structures.csv     — прошедшие фильтры качества и отбор по типу;
    data/klifs/kinases.csv        — по одной строке на киназу: сколько пригодных структур,
                                    какая из них лучшая, группа и семейство по Manning.

Логики здесь нет: разбор аргументов и вызов `kinase_ifp`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from opencadd.databases.klifs import setup_remote

from kinase_ifp.config import DATA_DIR, DEFAULT_SPECIES, TARGET_INHIBITOR_TYPES
from kinase_ifp.klifs import (
    annotate_kinase_taxonomy,
    count_by_inhibitor_type,
    fetch_kinase_taxonomy,
    fetch_structures,
    filter_structures,
    summarize_kinases,
)

# Значение --species, снимающее ограничение по организму.
ANY_SPECIES = "all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="сколько структур брать из KLIFS (по умолчанию все)",
    )
    parser.add_argument(
        "--species",
        default=DEFAULT_SPECIES,
        help=f"организм; {ANY_SPECIES!r} — не ограничивать (по умолчанию {DEFAULT_SPECIES})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_DIR / "klifs",
        help="каталог для таблиц структур",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    species = None if args.species == ANY_SPECIES else args.species

    session = setup_remote()
    structures = fetch_structures(session, limit=args.limit)
    selected = filter_structures(structures, species=species)
    kinases = summarize_kinases(selected)

    # Группа и семейство берутся отдельным запросом по именам: в таблице структур
    # эти колонки приходят пустыми.
    taxonomy = fetch_kinase_taxonomy(session, list(kinases["kinase.klifs_name"]))
    kinases = annotate_kinase_taxonomy(kinases, taxonomy)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "structures_all.csv": structures,
        "structures.csv": selected,
        "kinases.csv": kinases,
    }
    for name, table in paths.items():
        table.to_csv(args.out_dir / name, index=False, encoding="utf-8")

    print(f"Выгружено структур: {len(structures)}")
    print("Распределение по типу ингибирования (до фильтров качества):")
    for name, count in sorted(count_by_inhibitor_type(structures).items()):
        print(f"  {name:>8}: {count}")

    species_label = "все организмы" if species is None else species
    print(f"Отобрано структур (качество + тип {list(TARGET_INHIBITOR_TYPES)}, "
          f"{species_label}): {len(selected)}")
    print(f"Пригодных киназ: {len(kinases)}")
    без_группы = int(kinases["kinase.group"].isna().sum())
    print(f"Из них без группы KLIFS: {без_группы}")
    print("Распределение по группам Manning:")
    for name, count in kinases["kinase.group"].value_counts(dropna=False).items():
        print(f"  {str(name):>6}: {count}")
    print(f"Таблицы записаны в {args.out_dir}")


if __name__ == "__main__":
    main()
