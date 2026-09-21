"""CLI: метрики плана Б по набору поз — ROC-AUC, ранг нативной позы, Спирмен.

Запуск:
    docker compose run --rm dev python scripts/plan_b_metrics.py --run runs/<run_id>

Несколько прогонов за раз — повторённым ключом; тогда печатается ещё и top-1 как доля
наборов, а не как одно испытание, и со `--csv` пишется сводка по всем наборам:

    docker compose run --rm dev python scripts/plan_b_metrics.py \
        --run runs/<id-1> --run runs/<id-2> --csv data/calibration/pose_discrimination.csv

Логики здесь нет: разбор аргументов, печать и запись. Считает `evaluation.discrimination`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.discrimination import (
    DiscriminationError,
    PoseDiscrimination,
    discriminate_run,
    summarize_by_target,
    top1_rate,
    write_discrimination_csv,
    write_plan_b_metrics,
    write_target_summary,
)
from experiments.runs import RunError
from kinase_ifp.config import CORRECT_POSE_RMSD_A, NEAR_POSE_RMSD_A, RESULTS_DIR, TARGETS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="папка прогона в runs/; ключ можно повторить",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=TARGETS_DIR,
        help="каталог пакетов мишеней — нужен только ради имени киназы в сводке",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="куда записать строки по каждому прогону; без ключа они не пишутся",
    )
    parser.add_argument(
        "--results",
        action="store_true",
        help=(
            "записать сводку по мишеням в results/ (формат выхода этапа) — медианы "
            "и размахи, то есть те числа, что идут в текст"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    измерения: list[PoseDiscrimination] = []
    for run_dir in args.run:
        if not run_dir.is_dir():
            raise SystemExit(f"Нет папки прогона {run_dir}")
        try:
            измерение = discriminate_run(run_dir, args.targets_dir)
        except (DiscriminationError, RunError) as ошибка:
            raise SystemExit(f"{run_dir.name}: метрики не посчитаны — {ошибка}") from ошибка
        write_plan_b_metrics(run_dir, измерение)
        измерения.append(измерение)

        ничьи = f", ничьих {измерение.native_ties}" if измерение.native_ties else ""
        print(
            f"{измерение.run_id}: ROC-AUC {измерение.roc_auc:.3f} "
            f"(поз ближе {NEAR_POSE_RMSD_A} Å — {измерение.n_near}, "
            f"дальше {CORRECT_POSE_RMSD_A} Å — {измерение.n_far}), "
            f"Спирмен {измерение.spearman_rho:+.3f}, "
            f"ранг нативной {измерение.native_rank}{ничьи}"
        )

    выиграно, всего = top1_rate(измерения)
    print(f"top-1 (скор нативной позы строго лучший): {выиграно} из {всего}")

    if args.csv is not None:
        путь = write_discrimination_csv(измерения, args.csv)
        print(f"строки по прогонам: {путь}")

    if args.results:
        сводка = summarize_by_target(измерения)
        md, csv_путь = write_target_summary(сводка, RESULTS_DIR)
        for с in сводка:
            print(
                f"{с.pdb_id} ({с.kinase}): наборов {с.n_runs}, "
                f"ROC-AUC {с.roc_auc[0]:.3f} ({с.roc_auc[1]:.3f}-{с.roc_auc[2]:.3f}), "
                f"Спирмен {с.spearman[0]:+.3f}"
            )
        print(f"сводка по мишеням: {md} и {csv_путь}")


if __name__ == "__main__":
    main()
