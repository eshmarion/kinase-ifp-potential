"""CLI: рисунки раздела «Результаты и обсуждение».

Запуск:
    docker compose run --rm dev python scripts/make_figures.py
    docker compose run --rm dev python scripts/make_figures.py --only control

Без аргументов берёт все прогоны из `runs/index.csv`, у которых посчитан
`metrics_per_molecule.csv`, и кладёт рисунки в `results/figures/`.
Логики здесь нет: разбор аргументов и вызов `evaluation.figures`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.figures import (
    FigureEntry,
    FiguresError,
    collect_figure_data,
    figure_base_composition,
    figure_control,
    figure_ifp_map,
    figure_pipeline,
    figure_pose_scores,
    figure_refinement,
    figure_selection_size,
    figure_similarity,
    figure_states,
    runs_with_metrics,
    write_environment,
    write_figure_facts,
    write_panel_sizes,
    write_readme,
)
from experiments.layout import (
    FIGURE_BASE,
    FIGURE_CONTROL,
    FIGURE_IFP_MAP,
    FIGURE_PIPELINE,
    FIGURE_POSE_SCORES,
    FIGURE_REFINEMENT,
    FIGURE_SELECTION,
    FIGURE_SIMILARITY,
    FIGURE_STATES,
)
from kinase_ifp.config import FIGURES_DIR, RUNS_DIR

ВСЕ_РИСУНКИ = (
    "similarity",
    "control",
    "pipeline",
    "base",
    "refinement",
    "selection",
    "poses",
    "ifpmap",
    "states",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=None,
        help="папка прогона; можно повторить. По умолчанию — все прогоны с метриками",
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="каталог runs/")
    parser.add_argument("--out", type=Path, default=FIGURES_DIR, help="куда класть рисунки")
    parser.add_argument(
        "--only",
        choices=ВСЕ_РИСУНКИ,
        default=None,
        help="построить только один рисунок; по умолчанию строятся все",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    нужны = ВСЕ_РИСУНКИ if args.only is None else (args.only,)

    нужны_данные = "similarity" in нужны or "control" in нужны
    прогоны: list[Path] = args.run if args.run else runs_with_metrics(args.runs_dir)
    if нужны_данные:
        if not прогоны:
            raise SystemExit(
                f"В {args.runs_dir} нет ни одного прогона с metrics_per_molecule.csv. "
                "Сначала scripts/compute_metrics.py --run <папка прогона>"
            )
        print(f"Прогонов взято: {len(прогоны)}")

    готовые: list[FigureEntry] = []
    неудачи: list[str] = []

    # Рисунки строятся независимо: отсутствие данных для одного (например,
    # контрольных метрик до первого прогона source=diffsbdd) не должно отменять
    # остальные. Причина при этом печатается и меняет код возврата — молчаливого
    # пропуска здесь нет.
    if нужны_данные:
        данные = collect_figure_data(прогоны)
        print(f"Молекул в выборке: {len(данные)}")
        for имя, рисовать, файл in (
            ("similarity", figure_similarity, FIGURE_SIMILARITY),
            ("control", figure_control, FIGURE_CONTROL),
        ):
            if имя not in нужны:
                continue
            try:
                готовые.append(рисовать(данные, args.out / файл))
            except FiguresError as ошибка:
                неудачи.append(f"{имя}: {ошибка}")

    if "pipeline" in нужны:
        готовые.append(figure_pipeline(args.out / FIGURE_PIPELINE))

    # Состав базы строится из выгрузки KLIFS и пакета мишени, прогоны ему не нужны —
    # как и схеме конвейера. Отказ возможен один: выгрузки нет на диске, и тогда
    # рисунок не строится, а причина печатается вместе с остальными неудачами.
    if "base" in нужны:
        try:
            готовые.append(figure_base_composition(args.out / FIGURE_BASE))
        except FiguresError as ошибка:
            неудачи.append(f"base: {ошибка}")

    # Уточнение позы считается после прогона, и его результат живёт в `results/`,
    # а не в папке прогона. Поэтому рисунок берёт таблицы выхода этапа и отказывается,
    # если их нет или если у опыта не нашлось парного контроля.
    if "refinement" in нужны:
        try:
            готовые.append(figure_refinement(args.out / FIGURE_REFINEMENT))
        except FiguresError as ошибка:
            неудачи.append(f"refinement: {ошибка}")

    # Отбор и размер: та же помолекулярная выгрузка, что у проверки отбора, и то же
    # правило топа. Прогоны ему тоже не нужны — выгрузка живёт в выходе этапа.
    if "selection" in нужны:
        try:
            готовые.append(figure_selection_size(args.out / FIGURE_SELECTION))
        except FiguresError as ошибка:
            неудачи.append(f"selection: {ошибка}")

    # Облако поз: помолекулярные точки пересчёта по правилам KLIFS. Их выгружает
    # тот же скрипт, что и сводки, поэтому отдельного прогона рисунку не нужно.
    if "poses" in нужны:
        try:
            готовые.append(figure_pose_scores(args.out / FIGURE_POSE_SCORES))
        except FiguresError as ошибка:
            неудачи.append(f"poses: {ошибка}")

    # Карта отпечатка считается из пакета мишени на лету: готовых бит на диске нет,
    # в файлах калибровки лежат только счётчики расхождений.
    if "ifpmap" in нужны:
        try:
            готовые.append(figure_ifp_map(args.out / FIGURE_IFP_MAP))
        except FiguresError as ошибка:
            неудачи.append(f"ifpmap: {ошибка}")

    # Три состояния позы: числа лежат в реестре измерений, а не в выходе этапа —
    # физичность кладёт check_physics.py, энергию score_docking.py.
    if "states" in нужны:
        try:
            готовые.append(figure_states(args.out / FIGURE_STATES))
        except FiguresError as ошибка:
            неудачи.append(f"states: {ошибка}")

    for entry in готовые:
        print(f"Создан {entry.path}")

    # README переписывается целиком, поэтому обновляется только при полной сборке:
    # иначе `--only pipeline` стёр бы из него описания двух других рисунков.
    if args.only is None and готовые:
        print(f"Создан {write_readme(готовые, args.out)}")
        # Тем же условием, что README: файл описывает набор рисунков целиком,
        # и после `--only` он отражал бы один рисунок из трёх.
        print(f"Создан {write_panel_sizes(готовые, args.out)}")
        print(f"Создан {write_figure_facts(готовые, args.out)}")
        # И по той же причине — окружение: оно описывает, чем нарисован набор,
        # а не отдельный файл. После `--only` запись говорила бы о версии, которой
        # нарисован один рисунок из трёх, и это было бы хуже отсутствия записи.
        print(f"Создан {write_environment(args.out)}")
    elif args.only is not None:
        print("README не тронут: он собирается только при полном запуске, без --only")

    for строка in неудачи:
        print(f"НЕ ПОСТРОЕН {строка}")
    if неудачи:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
