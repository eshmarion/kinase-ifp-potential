"""Общие фикстуры тестов.

Структура 6tgu (CK2a2) — мишень работы, отобранная процедурой из базы KLIFS:
человек, разрешение 0.83 Å, quality score 8.0, ингибитор I типа.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from kinase_ifp.molecule_io import read_mol
from kinase_ifp.pocket import Pocket, load_pocket
from kinase_ifp.protonate import add_explicit_hydrogens

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TARGET_JSON = FIXTURES_DIR / "6tgu" / "target.json"


@pytest.fixture(scope="session")
def target_json() -> Path:
    return TARGET_JSON


@pytest.fixture(scope="session")
def pocket() -> Pocket:
    return load_pocket(TARGET_JSON)


@pytest.fixture(scope="session")
def crystal_ligand(pocket: Pocket) -> Chem.Mol:
    """Кристаллический лиганд мишени с достроенными водородами."""
    mol = read_mol(pocket.ligand_path)
    assert mol is not None, f"RDKit не разобрал {pocket.ligand_path}"
    return add_explicit_hydrogens(mol)
