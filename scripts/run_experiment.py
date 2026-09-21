"""CLI: провести прогон через расчёт и завести его в реестре.

Запуск:
    docker compose run --rm dev python scripts/run_experiment.py --run runs/<run_id>

Одна команда вместо четырёх: проверка папки, метрики, скор, реестр. Логики здесь
нет — только порядок вызовов, поэтому цепочка одинакова для набора поз
и для генерации, и её не приходится помнить руками.

Порядок шагов задан зависимостями, а не удобством: метрики и скор читают одни и те же
молекулы, но пишут разные файлы, а реестр выводит статус `scored` из `ranking.csv`,
поэтому регистрация идёт последней. Метрики стоят перед скором потому, что колонку
`condition` реестр собирает из `metrics_per_molecule.csv`, а не из паспорта.

Шаг метрик появился позже скора; до него цепочка обрывалась на скоре,
и полный прогон требовал второй команды — `scripts/compute_metrics.py`. Тот скрипт
остался: пересчитать метрики без скора иногда нужно, как и переранжировать без
пересчёта метрик (`scripts/rerank.py`).
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from experiments.metrics import MetricsError, compute_run_metrics, read_metrics
from experiments.ranking import read_ranking, score_run
from experiments.registry import INDEX_CSV, read_index, refresh_run, register_run
from experiments.restamp import verify_and_stamp
from experiments.runs import RunError, check_pulled, record_condition, target_for_run
from kinase_ifp.config import RUNS_DIR, TARGETS_DIR
from kinase_ifp.scoring import DEFAULT_TOP_FRACTION, ScoringError


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
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="каталог прогонов с реестром index.csv (по умолчанию runs/ в корне)",
    )
    parser.add_argument(
        "--stamp-target",
        action="store_true",
        help=(
            "сверить прогон с пакетом мишени и проставить штамп, если он сделан "
            "до формата паспорта прогона. Пересчёт идёт в копии, оригинал не трогается"
        ),
    )
    parser.add_argument(
        "--condition",
        default=None,
        help=(
            "условие эксперимента; попадает в паспорт, метрики и реестр. "
            "Умолчания нет: условие прогона нужно указать явно"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Путь прогона и путь реестра приводятся к одному виду: `describe_run` считает
    # `path` относительно каталога прогонов, а смесь «runs/x» и абсолютного RUNS_DIR
    # даёт ValueError из pathlib вместо внятного сообщения.
    run_dir = args.run.resolve()
    runs_dir = args.runs_dir.resolve()

    try:
        паспорт = check_pulled(run_dir)
        target_json = (
            args.target if args.target is not None else target_for_run(run_dir, TARGETS_DIR)
        )
    except RunError as ошибка:
        raise SystemExit(str(ошибка)) from ошибка
    if not target_json.is_file():
        raise SystemExit(f"Нет пакета мишени {target_json}")

    print(f"прогон {паспорт['run_id']}: source={паспорт['source']}, мишень {target_json}")

    if args.stamp_target:
        try:
            отчёт = verify_and_stamp(run_dir, target_json)
        except RunError as ошибка:
            raise SystemExit(str(ошибка)) from ошибка
        print(отчёт.describe())

    # Условие пишется до метрик: те копируют его в каждую строку из паспорта,
    # а реестр собирает колонку уже из метрик.
    if args.condition is not None:
        try:
            record_condition(run_dir, args.condition)
        except RunError as ошибка:
            raise SystemExit(str(ошибка)) from ошибка

    try:
        compute_run_metrics(run_dir, target_json)
    except MetricsError as ошибка:
        raise SystemExit(f"Метрики не посчитаны: {ошибка}") from ошибка

    метрики = read_metrics(run_dir)
    # `statistics.median`, а не элемент по индексу `len // 2`: при чётном числе
    # молекул тот даёт верхнюю медиану, и печать расходилась бы с таблицей
    # «было/стало», где медиана берётся с интерполяцией. Ровно это уже было
    # починено в `scripts/compute_metrics.py`, а здесь пережило починку.
    танимото = [float(строка["ifp_tanimoto"]) for строка in метрики]
    print(
        f"метрики: молекул {len(метрики)}, IFP-Танимото медиана "
        f"{statistics.median(танимото):.3f}"
    )

    try:
        score_run(run_dir, target_json, top_fraction=args.top_fraction)
    except ScoringError as ошибка:
        raise SystemExit(f"Скор не посчитан: {ошибка}") from ошибка

    строки = read_ranking(run_dir)
    отобрано = sum(int(строка["selected"]) for строка in строки)

    # Прогон мог быть заведён раньше — генерацией или пересборкой реестра. Тогда
    # строку нужно обновить, а не завести вторую: `register_run` вторую отвергает.
    заведён = any(
        строка["run_id"] == run_dir.name
        for строка in read_index(runs_dir / INDEX_CSV)
    )
    строка = (refresh_run if заведён else register_run)(run_dir, runs_dir)

    print(f"скор: посчитано молекул {len(строки)}, отобрано в топ {отобрано}")
    print(f"в реестре: status={строка['status']}, condition={строка['condition'] or '—'}")


if __name__ == "__main__":
    main()
