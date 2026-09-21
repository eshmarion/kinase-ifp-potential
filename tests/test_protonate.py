"""Тесты общей функции протонирования лиганда."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from kinase_ifp.protonate import (
    ProtonationError,
    add_explicit_hydrogens,
    has_explicit_hydrogens,
    normalize_charges,
    prepare_ligand,
)


def _ethanol_3d() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return Chem.RemoveHs(mol)


def test_добавляет_водороды() -> None:
    mol = _ethanol_3d()
    assert not has_explicit_hydrogens(mol)

    with_hydrogens = add_explicit_hydrogens(mol)

    assert has_explicit_hydrogens(with_hydrogens)
    assert with_hydrogens.GetNumAtoms() > mol.GetNumAtoms()


def test_у_водородов_ненулевые_координаты() -> None:
    """Без addCoords=True RDKit ставит всем водородам (0, 0, 0) и расчёт тихо врёт."""
    with_hydrogens = add_explicit_hydrogens(_ethanol_3d())

    positions = with_hydrogens.GetConformer().GetPositions()
    hydrogen_positions = np.array(
        [positions[a.GetIdx()] for a in with_hydrogens.GetAtoms() if a.GetAtomicNum() == 1]
    )
    assert len(hydrogen_positions) > 0
    assert not np.allclose(hydrogen_positions, 0.0)


def test_молекула_без_конформации_отвергается() -> None:
    with pytest.raises(ProtonationError, match="конформации"):
        add_explicit_hydrogens(Chem.MolFromSmiles("CCO"))


def test_исходная_молекула_не_меняется() -> None:
    mol = _ethanol_3d()
    before = mol.GetNumAtoms()

    add_explicit_hydrogens(mol)

    assert mol.GetNumAtoms() == before


@pytest.mark.parametrize(
    ("название", "вход", "ожидание"),
    [
        ("карбоновая кислота депротонируется", "CC(=O)O", "CC(=O)[O-]"),
        ("сульфоновая кислота депротонируется", "CS(=O)(=O)O", "CS(=O)(=O)[O-]"),
        ("алифатический амин протонируется", "CCN", "CC[NH3+]"),
        ("цвиттер-ион собирается целиком", "NCC(=O)O", "[NH3+]CC(=O)[O-]"),
        ("амид не трогаем", "CC(N)=O", "CC(N)=O"),
        ("анилин не трогаем", "Nc1ccccc1", "Nc1ccccc1"),
        ("фенол не трогаем", "Oc1ccccc1", "Oc1ccccc1"),
        ("уже заряженное не трогаем", "CC(=O)[O-]", "CC(=O)[O-]"),
    ],
)
def test_состояние_ионизации_при_ph74(название: str, вход: str, ожидание: str) -> None:
    получилось = Chem.MolToSmiles(normalize_charges(Chem.MolFromSmiles(вход)))

    assert получилось == Chem.MolToSmiles(Chem.MolFromSmiles(ожидание)), название


def test_нормализация_не_меняет_исходную_молекулу() -> None:
    mol = Chem.MolFromSmiles("CC(=O)O")

    normalize_charges(mol)

    assert Chem.MolToSmiles(mol) == "CC(=O)O"


def test_подготовка_даёт_и_заряды_и_водороды() -> None:
    mol = _ethanol_3d()

    готовый = prepare_ligand(mol)

    assert has_explicit_hydrogens(готовый)
