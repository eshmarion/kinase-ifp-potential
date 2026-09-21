"""CLI: выгрузка эталонных 595-битных отпечатков KLIFS.

Запуск:
    uv run python scripts/fetch_klifs_ifp.py                # 500 лучших структур
    uv run python scripts/fetch_klifs_ifp.py --limit 0      # вся отобранная выборка

Создаёт `data/klifs/klifs_ifp.csv` с колонками `structure_id, pdb_id, bits`. Это
единственный независимый источник для сверки нашего расчёта IFP:
отпечатки считает сторонняя программа KLIFS, а не наш код.

Файл дописывается, а не переписывается: повторный запуск с большим `--limit` добавляет
недостающие строки и не запрашивает заново уже выгруженные.

Логики здесь нет: разбор аргументов и вызов `kinase_ifp`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from opencadd.databases.klifs import setup_remote

from kinase_ifp.config import DATA_DIR, DEFAULT_FINGERPRINT_LIMIT
from kinase_ifp.klifs import (
    fetch_fingerprint_table,
    merge_fingerprint_tables,
    rank_structures,
    read_fingerprint_table,
)

# Значение --limit, снимающее ограничение на число структур.
NO_LIMIT = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_FINGERPRINT_LIMIT,
        help=f"сколько лучших по качеству структур брать; {NO_LIMIT} — все "
             f"(по умолчанию {DEFAULT_FINGERPRINT_LIMIT})",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        default=DATA_DIR / "klifs" / "structures.csv",
        help="таблица отобранных структур от scripts/fetch_klifs.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "klifs" / "klifs_ifp.csv",
        help="файл эталонных отпечатков",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    structures = pd.read_csv(args.structures)
    candidates = rank_structures(structures)
    if args.limit != NO_LIMIT:
        candidates = candidates.head(args.limit)

    existing = read_fingerprint_table(args.out)
    known = {
        int(structure_id): str(bits)
        for structure_id, bits in zip(existing["structure_id"], existing["bits"], strict=True)
    }

    table = fetch_fingerprint_table(setup_remote(), candidates, known=known)
    merged = merge_fingerprint_tables(existing, table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8")

    # Три числа, а не одно: структура без отпечатка — это отсутствие данных в KLIFS,
    # и молча пропадать она не должна.
    requested = len(candidates)
    print(f"Структур рассмотрено: {requested}, из них уже было выгружено: "
          f"{sum(1 for i in candidates['structure.klifs_id'] if int(i) in known)}")
    print(f"С эталонным отпечатком: {len(table)}, KLIFS не отдал отпечаток: "
          f"{requested - len(table)}")
    print(f"Всего строк в {args.out}: {len(merged)}")


if __name__ == "__main__":
    main()
