"""Пересчёт отпечатка готового прогона по правилам KLIFS.

Зачем модуль нужен. `ranking.csv` и `metrics_per_molecule.csv` прогона посчитаны
через ProLIF по нашим порогам (`fingerprint.compute_ifp`), а эталон KLIFS — по правилам
FingerPrintLib. По этим правилам расчёт воспроизводит эталон 190 битами из 191,
включая все 143 гидрофобных (`docs/calibration.md`, раздел 7), тогда как путь через
ProLIF находит лишь около четверти гидрофобных бит. Сравнение с эталоном и всё, что
из него выведено, поэтому пересчитывается здесь по правилам KLIFS; уточнение позы
(`scripts/optimize_poses.py`) считает по ним с самого начала.

Молекулы берутся из SDF как есть: производитель прогона уже подготовил их
`prepare_ligand`, и так же поступает `optimize_poses`. Отсюда
проверяемое следствие: скор исходных молекул прогона генерации здесь обязан совпасть
со `score_before` таблиц уточнения позы.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from evaluation.discrimination import POSE_KIND_PROP, DiscriminationError, native_mol_id
from evaluation.stats import mann_whitney_u
from evaluation.target import hinge_hbond, key_hbond_positions
from experiments.layout import METRICS_CSV, MOLECULES_SDF
from experiments.run_io import read_molecule_props
from kinase_ifp.config import (
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.fingerprint_klifs import ifp_from_groups, load_pocket_rules
from kinase_ifp.ligand_flags import groups_from_mol
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.scoring import DEFAULT_TOP_FRACTION, ScoredMolecule, rank_molecules
from kinase_ifp.similarity import select_types, tanimoto, tversky

# Контрольные характеристики не зависят от правил отпечатка, поэтому берутся из метрик
# прогона как есть; меняется только то, какие молекулы попали в топ.
CONTROL_COLUMNS: tuple[str, ...] = ("qed", "sa_score", "n_heavy_atoms")


class KlifsRescoreError(RuntimeError):
    """Прогон нельзя пересчитать: нет молекул, эталона или метрик."""


@dataclass(frozen=True)
class KlifsMolecule:
    """Одна молекула прогона, посчитанная по правилам KLIFS."""

    mol_id: str
    score_all7: float
    score_scoring6: float
    tanimoto: float
    key_hbond: int
    n_heavy: int


@dataclass(frozen=True)
class RescoredRun:
    """Молекулы прогона и число записей SDF, которые не прочитались вовсе."""

    molecules: list[KlifsMolecule]
    n_unreadable: int


def reference_ifp(package_json: Path) -> np.ndarray:
    """Эталонный отпечаток KLIFS из паспорта пакета мишени, форма (7, 85).

    Принимает `target.json`; поднимает `KlifsRescoreError`, если эталона у пакета нет.
    """
    биты = json.loads(package_json.read_text(encoding="utf-8")).get("klifs_ifp_bits")
    if not биты:
        raise KlifsRescoreError(f"{package_json}: у пакета нет эталонного отпечатка KLIFS")
    массив = np.array([символ == "1" for символ in биты])
    return массив.reshape(KLIFS_IFP_SHAPE[1], KLIFS_IFP_SHAPE[0]).T


def rescore_run(run_dir: Path, package_json: Path) -> RescoredRun:
    """Считает каждую молекулу прогона по правилам KLIFS против эталона мишени.

    Принимает папку прогона и `target.json` его мишени; возвращает `RescoredRun`.
    Скор — `tversky(эталон, молекула, α=1, β=0)` в двух наборах типов: все семь
    и шесть направленных (без `HYD`). Ключевая водородная связь — по позициям,
    где у эталона стоит донор или акцептор.

    Нечитаемая запись SDF не пропускается молча: она считается и возвращается
    числом, чтобы отчёт мог показать, из скольких молекул получены медианы.
    """
    sdf = run_dir / MOLECULES_SDF
    if not sdf.is_file():
        raise KlifsRescoreError(f"{run_dir.name}: нет {MOLECULES_SDF}, пересчитывать нечего")

    карман = load_pocket_rules(package_json)
    эталон = reference_ifp(package_json)
    ключевые = key_hbond_positions(эталон)
    эталон_все = select_types(эталон, KLIFS_INTERACTION_TYPES)
    эталон_направленные = select_types(эталон, SCORING_INTERACTION_TYPES)

    молекулы: list[KlifsMolecule] = []
    нечитаемых = 0
    with open_sdf(sdf) as supplier:
        for номер, mol in enumerate(supplier):
            if mol is None:
                нечитаемых += 1
                continue
            отпечаток = ifp_from_groups(groups_from_mol(mol), карман)
            молекулы.append(
                KlifsMolecule(
                    mol_id=mol.GetProp("mol_id") if mol.HasProp("mol_id") else f"{номер:04d}",
                    score_all7=tversky(
                        эталон_все, select_types(отпечаток, KLIFS_INTERACTION_TYPES), 1.0, 0.0
                    ),
                    score_scoring6=tversky(
                        эталон_направленные,
                        select_types(отпечаток, SCORING_INTERACTION_TYPES),
                        1.0,
                        0.0,
                    ),
                    tanimoto=tanimoto(отпечаток, эталон),
                    key_hbond=hinge_hbond(отпечаток, ключевые),
                    n_heavy=mol.GetNumHeavyAtoms(),
                )
            )
    if not молекулы:
        raise KlifsRescoreError(f"{run_dir.name}: ни одна молекула SDF не прочиталась")
    return RescoredRun(molecules=молекулы, n_unreadable=нечитаемых)


def native_score(run_dir: Path, molecules: Sequence[KlifsMolecule]) -> float:
    """Скор нативной позы набора — верхняя планка сравнения для этой мишени.

    Нативная поза ищется тем же `native_mol_id`, что и в метриках плана Б: по метке
    `pose_kind`, а при её отсутствии — по нулевому отклонению. Свой поиск здесь завести
    нельзя: наборы, записанные до появления метки, лежат в `runs/` и переписаны быть
    не могут, а два разных правила выбора дали бы две разные «нативные позы».
    """
    свойства = read_molecule_props(run_dir, ("rmsd_to_ref", POSE_KIND_PROP))
    пары = [
        (m.mol_id, m.score_all7, float(свойства[m.mol_id]["rmsd_to_ref"]))
        for m in molecules
        if свойства.get(m.mol_id, {}).get("rmsd_to_ref", "")
    ]
    if not пары:
        raise KlifsRescoreError(f"{run_dir.name}: ни у одной позы нет rmsd_to_ref")
    try:
        нативная = native_mol_id(свойства, пары)
    except DiscriminationError as ошибка:
        raise KlifsRescoreError(f"{run_dir.name}: {ошибка}") from ошибка
    return next(скор for mol_id, скор, _ in пары if mol_id == нативная)


@dataclass(frozen=True)
class Comparison:
    """Медиана по всему прогону, медиана по топу и p «топ против отбракованных»."""

    name: str
    all_value: float
    top_value: float
    p_value: float


def selection_effect(
    molecules: Sequence[KlifsMolecule],
    controls: dict[str, dict[str, float]] | None = None,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> tuple[int, list[Comparison]]:
    """Что даёт отбор верхней доли по скору всех семи типов.

    Принимает молекулы прогона и, по желанию, контрольные характеристики
    `{mol_id: {колонка: значение}}`; возвращает размер топа и список сравнений.
    Топ выбирается тем же `rank_molecules`, что и штатное ранжирование, — с тем же
    правилом ничьих и округлением размера вверх. Уровень значимости — Манна—Уитни
    топа против **отбракованных**, а не против всего прогона, куда топ входит сам.
    Для двоичной ключевой связи вместо медианы приводится доля.
    """
    ранжированные = rank_molecules(
        [ScoredMolecule(m.mol_id, m.score_all7, m.score_all7 / m.n_heavy) for m in molecules],
        top_fraction=top_fraction,
    )
    выбрано = {строка["mol_id"] for строка in ранжированные if строка["selected"]}
    топ = [m for m in molecules if m.mol_id in выбрано]
    прочие = [m for m in molecules if m.mol_id not in выбрано]

    def сравнить(имя: str, значения: dict[str, float], доля: bool = False) -> Comparison:
        все = [значения[m.mol_id] for m in molecules if m.mol_id in значения]
        верх = [значения[m.mol_id] for m in топ if m.mol_id in значения]
        низ = [значения[m.mol_id] for m in прочие if m.mol_id in значения]
        свёртка = (lambda x: float(np.mean(x))) if доля else (lambda x: float(median(x)))
        return Comparison(имя, свёртка(все), свёртка(верх), mann_whitney_u(верх, низ)[1])

    сравнения = [
        сравнить("скор, все семь типов", {m.mol_id: m.score_all7 for m in molecules}),
        сравнить("скор, направленные типы", {m.mol_id: m.score_scoring6 for m in molecules}),
        сравнить(
            "скор на тяжёлый атом", {m.mol_id: m.score_all7 / m.n_heavy for m in molecules}
        ),
        сравнить("Танимото к эталону", {m.mol_id: m.tanimoto for m in molecules}),
        сравнить(
            "доля с ключевой водородной связью",
            {m.mol_id: float(m.key_hbond) for m in molecules},
            доля=True,
        ),
    ]
    for колонка in CONTROL_COLUMNS if controls else ():
        значения = {mol_id: строка[колонка] for mol_id, строка in (controls or {}).items()}
        сравнения.append(сравнить(колонка, значения))
    return len(топ), сравнения


def read_controls(run_dir: Path) -> dict[str, dict[str, float]]:
    """Контрольные характеристики молекул из метрик прогона (`METRICS_CSV`).

    Пустые значения пропускаются: у несвязной молекулы QED не считается, и ноль
    на её месте сдвинул бы медиану.
    """
    путь = run_dir / METRICS_CSV
    if not путь.is_file():
        raise KlifsRescoreError(f"{run_dir.name}: нет {METRICS_CSV}")
    итог: dict[str, dict[str, float]] = {}
    with путь.open(encoding="utf-8", newline="") as файл:
        for строка in csv.DictReader(файл):
            итог[строка["mol_id"]] = {
                колонка: float(строка[колонка])
                for колонка in CONTROL_COLUMNS
                if строка.get(колонка, "") not in ("", None)
            }
    полные = len(CONTROL_COLUMNS)
    return {mol_id: значения for mol_id, значения in итог.items() if len(значения) == полные}
