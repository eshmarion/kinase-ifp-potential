"""CLI: сборка таблицы «было/стало» из прогонов реестра.

Запуск:
    docker compose run --rm dev python scripts/build_report.py
    docker compose run --rm dev python scripts/build_report.py --run runs/<A> --run runs/<B>

Список прогонов берётся из реестра `runs/index.csv`: в таблицу идут
строки со статусом `scored`. Аргументы `--run` остаются как явное переопределение —
ими собирают отчёт по прогонам, которых в реестре нет: чужая выгрузка, фикстуры,
проверка одной папки.

Результат — `results/before_after.md` и `before_after.csv`:
у числа в тексте курсовой должно быть одно место, откуда его берут и где его сверяет
сверкой, а не шестью папками прогонов, из которых его надо пересчитать.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.registry import INDEX_CSV, STATUS_SCORED, main_line_runs, scored_runs
from experiments.report import build_report
from kinase_ifp.config import MAIN_TARGET_PDB_ID, RESULTS_DIR, RUNS_DIR

DEFAULT_OUT_DIR = RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        metavar="RUN_DIR",
        help="папка прогона с run.json, metrics_per_molecule.csv и ranking.csv; "
        "аргумент повторяется по разу на прогон. Без него список берётся из реестра",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="каталог прогонов с реестром index.csv (по умолчанию runs/ в корне)",
    )
    parser.add_argument(
        "--all-scored",
        action="store_true",
        help="взять все прогоны со статусом scored, а не по одному на источник. "
        "Для диагностики: в курсовую идёт таблица без дубликатов и проб",
    )
    parser.add_argument(
        "--pdb-id",
        default=MAIN_TARGET_PDB_ID,
        metavar="PDB_ID",
        help="мишень, по которой собирается таблица (по умолчанию мишень работы). "
        "Колонки разных киназ подписаны одним и тем же условием и различаются только "
        "идентификатором прогона, поэтому по умолчанию показывается одна",
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="показать в таблице все мишени, а не одну. Ответ на вопрос «входит ли "
        "вторая киназа в результаты работы» принимают авторы разделов",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="куда класть before_after.md и before_after.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run:
        runs = args.run
    elif args.all_scored:
        runs = scored_runs(args.runs_dir)
    else:
        # По умолчанию — по одному прогону на источник: прямой отбор
        # всех `scored` даёт колонки-дубликаты и мешает методические пробы
        # с основными прогонами.
        runs = main_line_runs(args.runs_dir, pdb_id=None if args.all_targets else args.pdb_id)
    if not runs:
        # Пустая таблица выглядела бы как «посчитали и ничего не вышло». На деле это
        # ожидаемое состояние до расчёта скора и метрик: прогоны есть, статус generated.
        raise SystemExit(
            f"В реестре {args.runs_dir / INDEX_CSV} нет прогонов со статусом "
            f"{STATUS_SCORED}. Прогон получает его, когда рядом с молекулами лягут "
            "метрики (`metrics_per_molecule.csv`) и ranking.csv. "
            "Реестр пересобирается командой scripts/index_runs.py; собрать отчёт "
            "по конкретным папкам можно аргументами --run"
        )

    missing = [run_dir for run_dir in runs if not run_dir.is_dir()]
    if missing:
        raise SystemExit(
            "Нет таких папок прогонов: "
            + ", ".join(str(run_dir) for run_dir in missing)
            + ". Прогон создаётся генерацией или набором положений."
        )

    markdown_path, csv_path = build_report(runs, args.out_dir)
    print("Прогоны в таблице: " + ", ".join(run_dir.name for run_dir in runs))
    print(f"Таблица: {markdown_path}")
    print(f"Числа машинно: {csv_path}")


if __name__ == "__main__":
    main()
