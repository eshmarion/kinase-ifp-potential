"""CLI: различение поз и отбор по скору, пересчитанные по правилам KLIFS.

`ranking.csv` прогонов посчитан через ProLIF, а в текст работы идут только числа,
полученные по правилам KLIFS, — по ним расчёт воспроизводит эталон 190 битами из 191
Скрипт пересчитывает ровно то, что держится на отпечатке: метрики
различения поз по наборам и эффект отбора верхней доли на прогонах генерации.

Пример:

    docker compose run --rm dev python scripts/rescore_klifs_rules.py \\
        --poses runs/2026-08-23-6tgu-s0-n100 ... --generation runs/2026-08-24-6tgu-s0-n100-seed1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

from evaluation.discrimination import (
    POSE_KIND_PROP,
    DiscriminationError,
    PoseDiscrimination,
    discriminate_scores,
    pose_points,
    summarize_by_target,
    top1_rate,
    write_discrimination_csv,
)
from experiments.klifs_rescore import (
    KlifsMolecule,
    KlifsRescoreError,
    native_score,
    read_controls,
    rescore_run,
    selection_effect,
)
from experiments.layout import CSV_EOL
from experiments.results_io import Строка, записать
from experiments.run_io import read_molecule_props
from experiments.runs import RunError, target_for_run
from kinase_ifp.config import RESULTS_DIR, TARGETS_DIR

РЕЗУЛЬТАТЫ = RESULTS_DIR
ИМЯ = "klifs_rules_rescore"
КОМАНДА = "python scripts/rescore_klifs_rules.py --poses <наборы поз> --generation <прогоны>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", type=Path, nargs="+", required=True, help="наборы поз")
    parser.add_argument(
        "--generation", type=Path, nargs="*", default=[], help="прогоны генерации"
    )
    parser.add_argument("--targets-dir", type=Path, default=TARGETS_DIR)
    return parser.parse_args()


def _пакет(run_dir: Path, targets_dir: Path) -> Path:
    try:
        return target_for_run(run_dir, targets_dir)
    except RunError as ошибка:
        raise SystemExit(f"{run_dir.name}: не найден пакет мишени — {ошибка}") from ошибка


def _строки_различения(
    измерения: list[PoseDiscrimination], набор_типов: str
) -> list[Строка]:
    строки: list[Строка] = []
    for с in summarize_by_target(измерения):
        префикс = f"{с.pdb_id}, {набор_типов}: "
        строки += [
            Строка(префикс + "наборов поз", с.n_runs, "наборов"),
            Строка(префикс + "ROC-AUC, медиана", с.roc_auc[0], "AUC"),
            Строка(префикс + "ROC-AUC, минимум", с.roc_auc[1], "AUC"),
            Строка(префикс + "ROC-AUC, максимум", с.roc_auc[2], "AUC"),
            Строка(префикс + "Спирмен с отклонением, медиана", с.spearman[0], "ро"),
            Строка(префикс + "Спирмен с отклонением, минимум", с.spearman[1], "ро"),
            Строка(префикс + "Спирмен с отклонением, максимум", с.spearman[2], "ро"),
            Строка(префикс + "наихудший ранг нативной позы", с.native_rank_max, "ранг"),
            Строка(префикс + "ничьих с нативной позой, медиана", с.native_ties_median, "поз"),
            Строка(префикс + "наборов, где нативная строго первая", с.top1_strict, "наборов"),
        ]
    return строки


def _расхождение_с_уточнением(run_dir: Path, молекулы: list[KlifsMolecule]) -> float | None:
    """Наибольшая разница со `score_before` таблицы уточнения позы того же прогона.

    Обе величины посчитаны по правилам KLIFS по одним и тем же молекулам, поэтому
    должны совпасть; `None` — таблицы уточнения для прогона нет.
    """
    путь = РЕЗУЛЬТАТЫ / f"pose_optimization-{run_dir.name}-all7.csv"
    if not путь.is_file():
        return None
    with путь.open(encoding="utf-8", newline="") as файл:
        до = {с["mol_id"]: float(с["score_before"]) for с in csv.DictReader(файл)}
    общие = [m for m in молекулы if m.mol_id in до]
    return max(abs(m.score_all7 - до[m.mol_id]) for m in общие) if общие else None


def main() -> None:
    args = parse_args()
    строки: list[Строка] = []

    наборы_типов = (("все семь типов", "score_all7"), ("направленные", "score_scoring6"))
    по_типам: dict[str, list[PoseDiscrimination]] = {набор: [] for набор, _ in наборы_типов}
    нативные: dict[str, list[float]] = {}
    # Точки «скор против отклонения позы» помолекулярно: из них рисуется облако
    # подраздела о различении положения. Агрегаты по наборам этого не заменяют —
    # по медиане ROC-AUC не видно ни наклона облака, ни отделённой нативной позы.
    точки: list[dict[str, object]] = []
    for run_dir in args.poses:
        try:
            пересчёт = rescore_run(run_dir, _пакет(run_dir, args.targets_dir))
            for набор, поле in наборы_типов:
                скоры = {m.mol_id: getattr(m, поле) for m in пересчёт.molecules}
                по_типам[набор].append(discriminate_scores(run_dir, скоры, args.targets_dir))
            скоры_всех = {m.mol_id: m.score_all7 for m in пересчёт.molecules}
            последний = по_типам["все семь типов"][-1]
            вид_позы = read_molecule_props(run_dir, (POSE_KIND_PROP,))
            точки += [
                {
                    "run_id": последний.run_id,
                    "pdb_id": последний.pdb_id,
                    "mol_id": mol_id,
                    "score_all7": round(скор, 6),
                    "rmsd_to_ref": round(rmsd, 6),
                    "pose_kind": вид_позы.get(mol_id, {}).get(POSE_KIND_PROP, ""),
                }
                for mol_id, скор, rmsd in pose_points(run_dir, скоры_всех)
            ]
        except (KlifsRescoreError, DiscriminationError) as ошибка:
            raise SystemExit(f"{run_dir.name}: не пересчитан — {ошибка}") from ошибка
        последнее = по_типам["все семь типов"][-1]
        нативные.setdefault(последнее.pdb_id, []).append(
            native_score(run_dir, пересчёт.molecules)
        )
        print(
            f"{последнее.run_id}: ROC-AUC {последнее.roc_auc:.3f}, "
            f"Спирмен {последнее.spearman_rho:+.3f}, ранг нативной {последнее.native_rank}, "
            f"ничьих {последнее.native_ties}"
        )

    for pdb_id, скоры in sorted(нативные.items()):
        строки += [
            Строка(f"{pdb_id}: скор нативной позы, минимум по наборам", min(скоры), "скор"),
            Строка(f"{pdb_id}: скор нативной позы, максимум по наборам", max(скоры), "скор"),
        ]
    for набор, измерения in по_типам.items():
        строки += _строки_различения(измерения, набор)
        выиграно, всего = top1_rate(измерения)
        print(f"{набор}: нативная строго первая в {выиграно} наборах из {всего}")
    write_discrimination_csv(по_типам["все семь типов"], РЕЗУЛЬТАТЫ / f"{ИМЯ}_poses.csv")

    файл_точек = РЕЗУЛЬТАТЫ / f"{ИМЯ}_points.csv"
    with файл_точек.open("w", encoding="utf-8", newline="") as поток:
        писатель = csv.DictWriter(
            поток,
            fieldnames=[
                "run_id",
                "pdb_id",
                "mol_id",
                "score_all7",
                "rmsd_to_ref",
                "pose_kind",
            ],
            lineterminator=CSV_EOL,
        )
        писатель.writeheader()
        писатель.writerows(точки)
    print(f"точек поз: {len(точки)} -> {файл_точек}")

    помолекулярно: list[dict[str, object]] = []
    for run_dir in args.generation:
        try:
            пересчёт = rescore_run(run_dir, _пакет(run_dir, args.targets_dir))
            топ, сравнения = selection_effect(пересчёт.molecules, read_controls(run_dir))
        except KlifsRescoreError as ошибка:
            raise SystemExit(f"{run_dir.name}: не пересчитан — {ошибка}") from ошибка
        префикс = f"{run_dir.name}: "
        строки += [
            Строка(префикс + "молекул", len(пересчёт.molecules), "молекул"),
            Строка(префикс + "нечитаемых записей SDF", пересчёт.n_unreadable, "записей"),
            Строка(префикс + "молекул в топе 20 %", топ, "молекул"),
        ]
        for с in сравнения:
            строки += [
                Строка(префикс + f"{с.name}, весь прогон", с.all_value, "медиана или доля"),
                Строка(префикс + f"{с.name}, топ 20 %", с.top_value, "медиана или доля"),
                Строка(префикс + f"{с.name}, p топ против отбракованных", с.p_value, "p"),
            ]
        разница = _расхождение_с_уточнением(run_dir, пересчёт.molecules)
        if разница is not None:
            строки.append(
                Строка(префикс + "расхождение со score_before уточнения позы", разница, "скор")
            )
        print(f"{run_dir.name}: молекул {len(пересчёт.molecules)}, топ {топ}")
        for с in сравнения:
            print(f"  {с.name}: {с.all_value:.4f} -> {с.top_value:.4f}, p={с.p_value:.2e}")
        помолекулярно += [
            {"run_id": run_dir.name, **{k: getattr(m, k) for k in KlifsMolecule.__annotations__}}
            for m in пересчёт.molecules
        ]
        if разница is not None:
            print(f"  расхождение со score_before уточнения позы: {разница:.2e}")

    md, csv_путь = записать(
        строки,
        РЕЗУЛЬТАТЫ,
        имя_md=f"{ИМЯ}.md",
        имя_csv=f"{ИМЯ}.csv",
        заголовок="Различение поз и отбор по скору: пересчёт по правилам KLIFS",
        откуда=КОМАНДА,
    )
    if помолекулярно:
        путь = РЕЗУЛЬТАТЫ / f"{ИМЯ}_molecules.csv"
        with путь.open("w", encoding="utf-8", newline="") as файл:
            писатель = csv.DictWriter(
                файл, fieldnames=list(помолекулярно[0]), lineterminator=CSV_EOL
            )
            писатель.writeheader()
            for строка in помолекулярно:
                писатель.writerow(
                    {k: (round(v, 6) if isinstance(v, float) else v) for k, v in строка.items()}
                )
    print(f"сводка: {md} и {csv_путь}; медиана ROC-AUC по всем наборам "
          f"{median(m.roc_auc for m in по_типам['все семь типов']):.3f}")


if __name__ == "__main__":
    main()
