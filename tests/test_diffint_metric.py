"""Метрика воспроизведения H-связей в определении DiffInt (`scripts/diffint_metric.py`).

Проверка одна и она поверочная: кристаллический лиганд обязан дать ровно 1.0. Он и есть
эталон, по которому метрика определена, поэтому любое другое число означало бы ошибку
в знаменателе или в отборе типов, а не свойство молекулы. Ровно так же устроена
калибровка PoseBusters: измерение считается настроенным, когда эталон проходит его сам.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

from kinase_ifp.config import HBOND_TYPES
from kinase_ifp.fingerprint import bits_to_ifp
from kinase_ifp.fingerprint_klifs import load_pocket_rules
from kinase_ifp.protonate import prepare_ligand
from kinase_ifp.similarity import select_types

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "diffint_metric.py"

ПАКЕТ = Path("tests/fixtures/targets/6tgu/target.json")
if not ПАКЕТ.is_file():
    ПАКЕТ = Path("data/targets/6tgu/target.json")


def load_script() -> Any:
    """Загружает скрипт как модуль: `scripts/` намеренно не лежит на `pythonpath`."""
    spec = importlib.util.spec_from_file_location("diffint_metric_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")
def test_кристаллический_лиганд_воспроизводит_все_связи(tmp_path: Path) -> None:
    модуль = load_script()
    эталон = bits_to_ifp(json.loads(ПАКЕТ.read_text(encoding="utf-8"))["klifs_ifp_bits"])
    эталон_hb = select_types(эталон, HBOND_TYPES)
    assert эталон_hb.sum() > 0, "у 6tgu должны быть H-связи, иначе тест бессмыслен"

    # Лиганд кладётся в отдельный SDF: функция читает файл целиком, а не молекулу.
    лиганд = Chem.MolFromMolFile(str(ПАКЕТ.parent / "ligand.sdf"), removeHs=False)
    assert лиганд is not None
    лиганд = prepare_ligand(лиганд)
    sdf = tmp_path / "ligand.sdf"
    with sdf.open("w", encoding="utf-8", newline="") as поток:
        писатель = Chem.SDWriter(поток)
        писатель.write(лиганд)
        писатель.close()

    доли = модуль.доли_по_молекулам(sdf, load_pocket_rules(ПАКЕТ), эталон_hb)
    assert доли.tolist() == [1.0]
