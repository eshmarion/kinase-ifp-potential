"""Позиционный отпечаток по правилам KLIFS для произвольной молекулы.

Отличие от `fingerprint.compute_ifp`. Тот считает взаимодействия через ProLIF, то есть
по **его** порогам и его определениям типов: водородная связь — угол 130…180°,
ароматика — два условия по углам при 5.5 и 6.5 Å, гидрофоб — 4.5 Å. Правила KLIFS,
прочитанные в исходнике FingerPrintLib (`docs/klifs-rules-source.md`), другие:
водородная связь — отклонение D–H от D→A меньше 45° при 3.5 Å, ароматика — 4.0 Å
между **атомами** с одним углом между нормалями, ионный контакт — 4.0 Å. Эталонный
отпечаток KLIFS посчитан по вторым, поэтому сравнение нашего расчёта с эталоном
и любая величина, производная от отпечатка, должны считаться по ним же.

Этот модуль замыкает правила на молекулы, которых в KLIFS нет: карман берётся
из `pocket.mol2` пакета мишени (типы SYBYL на месте, правила применяются как
в оригинале), лиганд — из RDKit через `ligand_flags.groups_from_mol`.

Возвращается массив `(7, 85)` — та же раскладка `KLIFS_IFP_SHAPE`, что у
`compute_ifp`, поэтому `similarity`, `scoring` и метрики работают с ним без правок.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from kinase_ifp.config import KLIFS_IFP_SHAPE, N_KLIFS_INTERACTION_TYPES, N_KLIFS_POSITIONS
from kinase_ifp.klifs_rules import (
    ПРАВИЛА_KLIFS,
    Groups,
    KlifsRulesError,
    RuleThresholds,
    bits_for_residue,
    pocket_groups,
)
from kinase_ifp.ligand_flags import groups_from_mol

_КАРМАН_MOL2 = "pocket.mol2"


@dataclass(frozen=True)
class PocketRules:
    """Карман мишени, разобранный по флагам правил KLIFS, и его разметка позиций.

    Существует ради повторного расчёта: разбор кармана занимает почти всё время,
    а от молекулы не зависит вовсе. Оптимизация позы вызывает отпечаток десятки
    тысяч раз, и перечитывать mol2 на каждом шаге означало бы считать сутками.
    """

    residues: dict[str, Groups]
    position_to_residue: dict[int, str]
    label: str


def load_pocket_rules(package_json: Path) -> PocketRules:
    """Читает `pocket.mol2` пакета мишени и раскладывает остатки по флагам KLIFS.

    Принимает `target.json` пакета; возвращает `PocketRules`. Цепь берётся
    из `residue_to_position`, как в `klifs_rules.prepare_package`: mol2 её не хранит,
    а все ключи разметки относятся к одной цепи по построению пакета.
    """
    пакет = json.loads(package_json.read_text(encoding="utf-8"))
    разметка = пакет.get("residue_to_position") or {}
    if not разметка:
        raise KlifsRulesError(f"в пакете нет residue_to_position: {package_json}")
    цепь = next(iter(разметка)).rsplit(".", 1)[-1]
    карман = package_json.parent / _КАРМАН_MOL2
    if not карман.is_file():
        raise KlifsRulesError(
            f"нет {_КАРМАН_MOL2} в пакете {package_json.parent}: правила KLIFS требуют "
            "mol2 самого KLIFS, потому что флаги атома определяются типом SYBYL"
        )
    return PocketRules(
        residues=pocket_groups(карман, цепь),
        position_to_residue={позиция: остаток for остаток, позиция in разметка.items()},
        label=package_json.parent.name,
    )


def ifp_by_klifs_rules(
    mol: Chem.Mol,
    pocket: PocketRules,
    thresholds: RuleThresholds = ПРАВИЛА_KLIFS,
) -> np.ndarray:
    """Отпечаток `(7, 85)` молекулы по правилам KLIFS в заданном кармане.

    Принимает молекулу с явными водородами и конформером (`protonate.prepare_ligand`),
    разобранный карман и пороги правил; возвращает булев массив формы
    `KLIFS_IFP_SHAPE`. Позиция без остатка в разметке даёт семь нулей — как в эталоне.
    """
    отпечаток = np.zeros(KLIFS_IFP_SHAPE, dtype=bool)
    лиганд = groups_from_mol(mol)
    for позиция in range(1, N_KLIFS_POSITIONS + 1):
        остаток = pocket.position_to_residue.get(позиция)
        группы = pocket.residues.get(остаток) if остаток else None
        if группы is None:
            continue
        биты = bits_for_residue(группы, лиганд, thresholds)
        for тип in range(N_KLIFS_INTERACTION_TYPES):
            отпечаток[тип, позиция - 1] = биты[тип]
    return отпечаток


def ifp_from_groups(
    ligand: Groups,
    pocket: PocketRules,
    thresholds: RuleThresholds = ПРАВИЛА_KLIFS,
) -> np.ndarray:
    """То же, но по уже готовым флагам лиганда — для многократного пересчёта позы.

    Оптимизация позы двигает координаты, а состав флагов при твёрдотельном сдвиге
    не меняется: пересобирать `Groups` из молекулы на каждом шаге незачем.
    """
    отпечаток = np.zeros(KLIFS_IFP_SHAPE, dtype=bool)
    for позиция in range(1, N_KLIFS_POSITIONS + 1):
        остаток = pocket.position_to_residue.get(позиция)
        группы = pocket.residues.get(остаток) if остаток else None
        if группы is None:
            continue
        биты = bits_for_residue(группы, ligand, thresholds)
        for тип in range(N_KLIFS_INTERACTION_TYPES):
            отпечаток[тип, позиция - 1] = биты[тип]
    return отпечаток
