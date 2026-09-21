"""CLI: набор поз из кристаллического лиганда, без генерации.

Запуск:
    uv run python scripts/make_local_poses.py --n 100

Из кристаллического лиганда мишени делается набор: нативная поза плюс искусственно
испорченные («декои») с равномерно разложенным RMSD. По нему меряются ROC-AUC, top-1
и монотонность скора — то есть проверяется сам потенциал, а не модель.

Прогонов нужно несколько: top-1 при одном наборе — это одно испытание, а не доля.
Разные наборы задаются `--seed`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.layout import FAILURES_CSV
from experiments.local_poses import build_local_pose_run
from kinase_ifp.config import DATA_DIR, POSE_DEFAULT_SEED, RUNS_DIR

DEFAULT_TARGET_JSON = DATA_DIR / "targets" / "6tgu" / "target.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_JSON,
        help="target.json пакета мишени",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="сколько положений в наборе, считая нативное (по умолчанию 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=POSE_DEFAULT_SEED,
        help="сид набора; разные сиды дают независимые наборы для top-1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RUNS_DIR,
        help="каталог, в котором создаётся папка прогона",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = build_local_pose_run(
        target_json=args.target,
        n=args.n,
        seed=args.seed,
        runs_dir=args.out,
    )

    failures = (run_dir / FAILURES_CSV).read_text(encoding="utf-8").splitlines()[1:]
    print(f"прогон: {run_dir}")
    print(f"молекул записано: {args.n - len(failures)} из {args.n}")
    print(f"не построено поз: {len(failures)} (см. failures.csv)")


if __name__ == "__main__":
    main()
