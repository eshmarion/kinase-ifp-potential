"""CLI: свип по размеру отбираемого топа.

Запуск:
    docker compose run --rm dev python scripts/run_sweep.py --run runs/<run_id>

Считает кривую компромисса и пишет `results/sweep_top_fraction.md` и `.csv`
. Папку прогона не меняет: `ranking.csv` остаётся с той долей топа,
с которой посчитан прогон, а разрез по другим долям делается на лету.

Логики здесь нет: разбор аргументов и вызов `experiments.sweep`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.sweep import DEFAULT_SWEEP_FRACTIONS, SweepError, build_sweep

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="папка прогона; можно повторять для нескольких прогонов",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_SWEEP_FRACTIONS),
        help="доли топа в единицах (0, 1]; по умолчанию 0.05 0.1 0.2 0.3 0.5 1.0",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results",
        help="куда положить таблицу (по умолчанию results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        md_path, csv_path = build_sweep(
            [Path(каталог) for каталог in args.run],
            args.out_dir,
            tuple(args.fractions),
        )
    except SweepError as ошибка:
        raise SystemExit(str(ошибка)) from ошибка
    print(f"Записано: {md_path}")
    print(f"Записано: {csv_path}")


if __name__ == "__main__":
    main()
