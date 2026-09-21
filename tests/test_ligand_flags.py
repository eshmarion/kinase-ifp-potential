"""Флаги лиганда по правилам KLIFS из RDKit и отпечаток, посчитанный по ним.

Главная проверка здесь одна: отпечаток кристаллического лиганда, посчитанный через
перцепцию RDKit, обязан совпасть с эталоном KLIFS **бит в бит**. У этого лиганда есть
оба представления — `ligand.mol2` с типами SYBYL и `ligand.sdf`, — поэтому подмена
типов перцепцией проверяется измерением, а не рассуждением.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from kinase_ifp.config import KLIFS_IFP_SHAPE
from kinase_ifp.fingerprint_klifs import ifp_by_klifs_rules, load_pocket_rules
from kinase_ifp.klifs_rules import fingerprint_by_klifs_rules
from kinase_ifp.ligand_flags import (
    LigandFlagsError,
    groups_from_mol,
    is_acceptor,
)
from kinase_ifp.protonate import prepare_ligand

ПАКЕТ = Path("tests/fixtures/targets/6tgu/target.json")
if not ПАКЕТ.is_file():
    ПАКЕТ = Path("data/targets/6tgu/target.json")


def _лиганд() -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(ПАКЕТ.parent / "ligand.sdf"), removeHs=False)
    assert mol is not None
    return prepare_ligand(mol)


def _эталон() -> np.ndarray:
    биты = json.loads(ПАКЕТ.read_text(encoding="utf-8"))["klifs_ifp_bits"]
    return np.array([c == "1" for c in биты]).reshape(85, 7).T


@pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")
def test_отпечаток_через_rdkit_совпадает_с_эталоном_klifs() -> None:
    отпечаток = ifp_by_klifs_rules(_лиганд(), load_pocket_rules(ПАКЕТ))
    assert отпечаток.shape == KLIFS_IFP_SHAPE
    расхождения = np.argwhere(отпечаток != _эталон())
    assert расхождения.size == 0, f"бит расходится с эталоном KLIFS: {расхождения.tolist()}"


@pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")
def test_путь_через_rdkit_совпадает_с_путём_через_типы_sybyl() -> None:
    строка = fingerprint_by_klifs_rules(ПАКЕТ)
    через_sybyl = np.array([c == "1" for c in строка]).reshape(85, 7).T
    через_rdkit = ifp_by_klifs_rules(_лиганд(), load_pocket_rules(ПАКЕТ))
    assert (через_sybyl == через_rdkit).all()


def test_карбоксилат_даёт_два_аниона() -> None:
    """Оба кислорода карбоксилата анионные: правило типа `O.co2` оригинала.

    RDKit пишет −1 только на одном кислороде, и без делокализации бит `ION+`
    у лиганда 6tgu пропадает — именно на этом расходились два пути расчёта.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]"), addCoords=True)
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D)
    assert len(groups_from_mol(mol).anions) == 2


def test_делокализация_не_распространяется_на_нитрогруппу() -> None:
    """У нитрогруппы анионом остаётся один кислород — тот, что несёт заряд.

    Правило оригинала (`AFP::SetIsAnn`) считает анионом любой атом с отрицательным
    формальным зарядом, поэтому один бит здесь законен. Незаконно было бы второе:
    заряд нитрогруппы в целом равен нулю, и делокализация карбоксилатного типа
    к ней не применяется — центр несёт `+1`.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("C[N+](=O)[O-]"), addCoords=True)
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D)
    assert len(groups_from_mol(mol).anions) == 1


@pytest.mark.parametrize(
    ("smiles", "индекс", "ожидается", "почему"),
    [
        ("c1ccncc1", 3, True, "азот пиридина — акцептор"),
        ("CC(=O)NC", 3, False, "амидный азот — тип N.am, не акцептор"),
        ("c1cc[nH]c1", 3, False, "пиррольный азот плоский трёхсвязный — N.pl3"),
        ("C[N+](C)(C)C", 1, False, "азот с четырьмя связями — N.4"),
        ("CO", 1, True, "нейтральный кислород — акцептор"),
    ],
)
def test_правило_акцептора(smiles: str, индекс: int, ожидается: bool, почему: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    assert is_acceptor(mol.GetAtomWithIdx(индекс)) is ожидается, почему


def test_без_конформера_отказ() -> None:
    with pytest.raises(LigandFlagsError, match="конформера"):
        groups_from_mol(Chem.MolFromSmiles("CCO"))


def test_неявные_водороды_отказ() -> None:
    """Правило донора требует сам атом водорода, поэтому неявные — отказ, а не нули."""
    mol = Chem.MolFromSmiles("CCO")
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D)
    with pytest.raises(LigandFlagsError, match="водороды неявные"):
        groups_from_mol(mol)
