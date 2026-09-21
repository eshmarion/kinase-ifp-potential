"""CLI: доверительные интервалы и p-value прогона.

Запуск:
    docker compose run --rm dev python scripts/compute_stats.py --run runs/2026-08-24-6tgu-s0-n100

Создаёт в папке прогона `metrics_summary.csv` — строка на каждое условие, границы
95-процентного бутстрэп-интервала по каждой метрике и p-value критерия Манна–Уитни
для отобранного топа. Отсюда границы забирает сборка таблицы «было/стало»
(`scripts/build_report.py`), поэтому порядок такой: метрики, скор, статистика, отчёт.

Логики здесь нет: разбор аргументов и вызов `evaluation.stats`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.stats import SUMMARY_COLUMNS, StatsError, summarize_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="папка прогона в runs/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run.is_dir():
        raise SystemExit(f"Нет папки прогона {args.run}")

    try:
        путь = summarize_run(args.run)
    except StatsError as ошибка:
        raise SystemExit(f"Статистику посчитать не удалось: {ошибка}") from ошибка

    with путь.open(encoding="utf-8", newline="") as файл:
        строки = list(csv.DictReader(файл))

    print(f"Условий посчитано: {len(строки)}")
    for строка in строки:
        границы = [
            f"{имя[: -len('_lo')]} {строка[имя]}..{строка[имя[: -len('_lo')] + '_hi']}"
            for имя in SUMMARY_COLUMNS
            if имя.endswith("_lo") and строка[имя]
        ]
        p = [
            f"{имя[: -len('_p')]} p={float(строка[имя]):.4f}"
            for имя in SUMMARY_COLUMNS
            if имя.endswith("_p") and строка[имя]
        ]
        print(f"  {строка['condition']}: интервалов {len(границы)}, сравнений {len(p)}")
        for текст in p:
            print(f"    {текст}")
    print(f"Записано: {путь}")


if __name__ == "__main__":
    main()
