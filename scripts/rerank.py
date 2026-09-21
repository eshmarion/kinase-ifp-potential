"""CLI: переранжирование прогона по IFP-скору.

Запуск:
    uv run python scripts/rerank.py --run runs/2026-08-24-6tgu-s0-n100

Создаёт в папке прогона `ranking.csv` с колонками
`mol_id, ifp_score, ifp_score_norm, rank, selected` и записывает параметры отбора
в `run.json`. Логики здесь нет: разбор аргументов и вызов `experiments.ranking`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ranking import read_ranking, score_run
from experiments.runs import target_for_run
from kinase_ifp.config import TARGETS_DIR
from kinase_ifp.scoring import DEFAULT_TOP_FRACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="папка прогона в runs/")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="target.json мишени; по умолчанию берётся по target.pdb_id из паспорта",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=DEFAULT_TOP_FRACTION,
        help=f"доля прогона в отобранном топе (по умолчанию {DEFAULT_TOP_FRACTION})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run.is_dir():
        raise SystemExit(f"Нет папки прогона {args.run}")

    target_json = (
        args.target if args.target is not None else target_for_run(args.run, TARGETS_DIR)
    )
    if not target_json.is_file():
        raise SystemExit(f"Нет пакета мишени {target_json}")

    путь = score_run(args.run, target_json, top_fraction=args.top_fraction)
    строки = read_ranking(args.run)
    отобрано = sum(int(строка["selected"]) for строка in строки)
    лучшая = min(строки, key=lambda s: int(s["rank"]))

    print(f"Молекул посчитано: {len(строки)}, отобрано в топ: {отобрано}")
    print(f"Лучшая: {лучшая['mol_id']}, скор {float(лучшая['ifp_score']):.3f}")
    print(f"Записано: {путь}")


if __name__ == "__main__":
    main()
