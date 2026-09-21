"""CLI: целевые метрики прогона.

Запуск:
    docker compose run --rm dev python scripts/compute_metrics.py --run runs/2026-08-23-6tgu-s0-n100

Создаёт в папке прогона `metrics_per_molecule.csv` с полным заголовком таблицы метрик.
Логики здесь нет: разбор аргументов и вызов `experiments.metrics`.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Iterable
from pathlib import Path

from experiments.metrics import compute_run_metrics, read_metrics
from experiments.runs import target_for_run
from kinase_ifp.config import TARGETS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="папка прогона в runs/")
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="target.json мишени; по умолчанию берётся по target.pdb_id из паспорта",
    )
    return parser.parse_args()


def сводка_танимото(значения: Iterable[float]) -> tuple[float, float, float]:
    """Медиана, минимум и максимум IFP-Танимото для печати в консоль.

    Принимает значения колонки `ifp_tanimoto`, возвращает тройку чисел.

    Вынесена из `main` не ради переиспользования, а ради проверки: печать этой команды
    дословно попала в записи и разошлась с таблицей «было/стало».
    Медиана обязана считаться так же, как её считает таблица, — с линейной
    интерполяцией; тест сверяет её с `pandas` на прогонах, лежащих в репозитории.
    """
    упорядоченные = sorted(значения)
    if not упорядоченные:
        raise ValueError("Пустая колонка ifp_tanimoto: сводку считать не по чему")
    return statistics.median(упорядоченные), упорядоченные[0], упорядоченные[-1]


def main() -> None:
    args = parse_args()
    if not args.run.is_dir():
        raise SystemExit(f"Нет папки прогона {args.run}")

    target_json = (
        args.target if args.target is not None else target_for_run(args.run, TARGETS_DIR)
    )
    if not target_json.is_file():
        raise SystemExit(f"Нет пакета мишени {target_json}")

    путь = compute_run_metrics(args.run, target_json)
    строки = read_metrics(args.run)

    доля = sum(int(строка["hinge_hbond"]) for строка in строки) / len(строки)
    медиана, минимум, максимум = сводка_танимото(
        float(строка["ifp_tanimoto"]) for строка in строки
    )

    print(f"Молекул посчитано: {len(строки)}")
    print(f"IFP-Танимото: медиана {медиана:.3f}, размах {минимум:.3f}..{максимум:.3f}")
    print(f"Доля молекул с ключевой водородной связью: {доля:.2f}")
    print(f"Записано: {путь}")


if __name__ == "__main__":
    main()
