"""Целевые метрики прогона: `metrics_per_molecule.csv`.

Здесь только работа с прогоном — чтение молекул, сверка пакета мишени, запись файла.
Сами метрики живут в `evaluation.target` и ни от чего, кроме двух отпечатков,
не зависят: на уровне B они могут понадобиться внутри цикла сэмплирования, где
никаких папок нет. То же разделение, что у `kinase_ifp.scoring` и `experiments.ranking`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Final

from rdkit import Chem

from evaluation.control import (
    ControlError,
    is_connected,
    is_valid,
    n_heavy_atoms,
    qed,
    sa_score,
)
from evaluation.target import hinge_hbond, ifp_tanimoto, ifp_tversky, key_hbond_positions
from experiments.layout import CSV_EOL, METRICS_CSV, MOLECULES_SDF, SOURCE_DIFFSBDD
from experiments.run_io import append_failures
from experiments.runs import RunError, ensure_target_matches, read_passport
from kinase_ifp.fingerprint import IfpError, compute_ifp
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.pocket import load_pocket

# Заголовок из формата таблицы метрик, ровно в этом порядке. Колонки, которые в прогоне
# не считались, присутствуют и остаются пустыми: отсутствие колонки и пустая ячейка
# означают разное, и потребители рассчитывают на полный заголовок.
METRICS_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "mol_id",
    "condition",
    "guidance_scale",
    "seed",
    "smiles",
    "valid",
    "connected",
    "ifp_tanimoto",
    "ifp_tversky",
    "hinge_hbond",
    "qed",
    "sa_score",
    "n_heavy_atoms",
    "posebusters_valid",
    "posebusters_fail_reasons",
    "vina_score",
    "ligand_efficiency",
)

# Контрольные свойства молекулы. Считаются только при `source=diffsbdd`:
# на наборе поз это одна молекула в ста конформациях, где они вырождены по построению.
CONTROL_COLUMNS: Final[tuple[str, ...]] = ("qed", "sa_score", "n_heavy_atoms")

# Колонки, которые не заполняет никто: PoseBusters и докинг (по умолчанию
# не входит). Перечислены явно, чтобы пустая ячейка читалась как «неприменимо»,
# а не как забытый расчёт.
NEVER_FILLED_COLUMNS: Final[tuple[str, ...]] = (
    "posebusters_valid",
    "posebusters_fail_reasons",
    "vina_score",
    "ligand_efficiency",
)


def unfilled_columns(source: str) -> tuple[str, ...]:
    """Колонки таблицы метрик, которые у прогона этого источника обязаны остаться пустыми.

    Зависит от источника, а не является константой: контрольные метрики применимы
    только к генерации, и прежде разница была невидима — пустые ячейки у обоих
    источников читались одинаково.
    """
    if source == SOURCE_DIFFSBDD:
        return NEVER_FILLED_COLUMNS
    return CONTROL_COLUMNS + NEVER_FILLED_COLUMNS

# Этап, под которым в `failures.csv` записываются молекулы без отпечатка. Отличается
# от `scoring`: скор и метрики считаются разными командами и в разное время, и по
# `failures.csv` должно быть видно, какая из них молекулу не приняла.
METRICS_STAGE: Final[str] = "metrics"


class MetricsError(RuntimeError):
    """Метрики посчитать нельзя: нет эталона, нет молекул или испорчен прогон."""


def compute_run_metrics(run_dir: Path, target_json: Path) -> Path:
    """Считает целевые метрики всех молекул прогона и пишет `metrics_per_molecule.csv`.

    Возвращает путь к записанному файлу. Молекулу, для которой отпечаток не считается,
    функция не пропускает молча: причина уходит в `failures.csv` со стадией `metrics`,
    а расчёт продолжается. Уронить прогон из-за одной молекулы так же неверно, как
    потерять её без следа.

    Поднимает `MetricsError`, если у мишени нет эталонного отпечатка, если у эталона
    нет ни одной водородной связи в кармане (тогда `hinge_hbond` не определён) или если
    не посчиталась ни одна молекула — последнее означает поломку, а не свойство данных.
    """
    pocket = load_pocket(target_json)
    if pocket.reference_ifp is None:
        raise MetricsError(
            f"{target_json}: у мишени нет эталонного отпечатка KLIFS, сравнивать не с чем"
        )

    sdf = run_dir / MOLECULES_SDF
    if not sdf.is_file():
        raise MetricsError(f"В прогоне нет {MOLECULES_SDF}: {run_dir}")

    # До расчёта: считать метрики по чужому пакету мишени нельзя ровно по той же
    # причине, по которой нельзя считать скор.
    try:
        ensure_target_matches(run_dir, target_json)
    except RunError as ошибка:
        raise MetricsError(str(ошибка)) from ошибка

    ключевые = key_hbond_positions(pocket.reference_ifp)
    if not ключевые:
        raise MetricsError(
            f"{target_json}: у эталонного лиганда нет ни одной водородной связи "
            "с карманом, метрика hinge_hbond на этой мишени не определена"
        )

    паспорт = read_passport(run_dir)
    источник = паспорт.get("source", "")
    сэмплинг = паспорт.get("sampling", {})
    общее = {
        "run_id": паспорт["run_id"],
        "condition": паспорт.get("condition", ""),
        "guidance_scale": сэмплинг.get("guidance_scale", ""),
        "seed": сэмплинг.get("seed", ""),
    }

    строки: list[dict[str, Any]] = []
    отказы: list[tuple[str, str, str]] = []

    # Поток, а не путь строкой: RDKit передаёт путь в C++ через ANSI и на Windows
    # не открывает ничего с кириллицей. Каталог прогона может лежать во временной
    # копии (`experiments.restamp`), а её корень — `%TEMP%` внутри домашнего
    # каталога пользователя. Правило и обёртка — `kinase_ifp.molecule_io`.
    # removeHs=False обязателен: водороды поставил производитель молекул, и без них
    # ProLIF не отличит донор от акцептора.
    with open_sdf(sdf) as supplier:
        for номер, mol in enumerate(supplier):
            mol_id = f"{run_dir.name}-{номер:04d}"
            if mol is None:
                отказы.append((mol_id, METRICS_STAGE, "RDKit не разобрал запись SDF"))
                continue
            if mol.HasProp("mol_id"):
                mol_id = mol.GetProp("mol_id")

            try:
                отпечаток = compute_ifp(pocket, mol)
            except IfpError as ошибка:
                отказы.append((mol_id, METRICS_STAGE, str(ошибка)))
                continue

            строка: dict[str, Any] = dict.fromkeys(METRICS_COLUMNS, "")
            строка.update(общее)
            строка.update(
                {
                    "mol_id": mol_id,
                    "smiles": Chem.MolToSmiles(mol),
                    "valid": is_valid(mol),
                    "connected": is_connected(mol),
                    "ifp_tanimoto": ifp_tanimoto(отпечаток, pocket.reference_ifp),
                    "ifp_tversky": ifp_tversky(отпечаток, pocket.reference_ifp),
                    "hinge_hbond": hinge_hbond(отпечаток, ключевые),
                }
            )
            # Контрольные свойства считаются только при генерации: у набора поз
            # это одна молекула в ста конформациях, и колонки обязаны остаться пустыми.
            if источник == SOURCE_DIFFSBDD:
                try:
                    строка.update(
                        {
                            "qed": _или_пусто(qed(mol)),
                            "sa_score": _или_пусто(sa_score(mol)),
                            "n_heavy_atoms": _или_пусто(n_heavy_atoms(mol)),
                        }
                    )
                except ControlError as ошибка:
                    # Молекула посчитана по целевым метрикам и в таблицу попадёт; пустая
                    # тройка контрольных колонок без следа выглядела бы как «неприменимо».
                    отказы.append((mol_id, METRICS_STAGE, f"контрольные метрики: {ошибка}"))

            строки.append(строка)

    if отказы:
        append_failures(run_dir, отказы)
    if not строки:
        raise MetricsError(
            f"В прогоне {run_dir.name} не посчиталась ни одна молекула из "
            f"{len(строки) + len(отказы)}: причины в failures.csv"
        )

    return write_metrics(run_dir, строки)


def _или_пусто(значение: float | int | None) -> float | int | str:
    """Значение метрики или пустая ячейка, если метрика не определена.

    У невалидной молекулы контрольных свойств нет (`docs/metrics.md`, раздел 4),
    и записывать вместо них ноль нельзя: ноль — это значение, а не его отсутствие.
    """
    return "" if значение is None else значение


def write_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Пишет `metrics_per_molecule.csv` по формату таблицы метрик и возвращает путь к нему."""
    путь = run_dir / METRICS_CSV
    with путь.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRICS_COLUMNS, lineterminator=CSV_EOL)
        writer.writeheader()
        writer.writerows(rows)
    return путь


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Читает `metrics_per_molecule.csv` прогона."""
    путь = run_dir / METRICS_CSV
    if not путь.is_file():
        raise MetricsError(f"В прогоне нет {METRICS_CSV}: {run_dir}")
    with путь.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))
