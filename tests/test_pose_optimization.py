"""Оптимизация позы дискретным потенциалом взаимодействий.

Проверяется не «код отработал», а четыре свойства, без которых результат не значит
ничего: потенциал не ухудшает позу, он не двигает молекулу при бессмысленной цели,
он не выпускает молекулу за заданный предел смещения и не оставляет её налезшей
на белок.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from kinase_ifp.fingerprint_klifs import load_pocket_rules
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.pose_optimization import (
    CLASH_DISTANCE_A,
    PoseOptimizationError,
    has_clash,
    min_pocket_distance,
    optimize_pose,
    pocket_heavy_atoms,
)
from kinase_ifp.poses import heavy_atom_rmsd

ПАКЕТ = Path("data/targets/6tgu/target.json")
ПРОГОН = Path("runs/2026-08-24-6tgu-s0-n100/molecules.sdf")
ЕСТЬ_ДАННЫЕ = ПАКЕТ.is_file() and ПРОГОН.is_file()

pytestmark = pytest.mark.skipif(
    not ЕСТЬ_ДАННЫЕ, reason="нужны пакет мишени 6tgu и прогон генерации"
)


@pytest.fixture(scope="module")
def мишень() -> tuple:
    биты = json.loads(ПАКЕТ.read_text(encoding="utf-8"))["klifs_ifp_bits"]
    эталон = np.array([c == "1" for c in биты]).reshape(85, 7).T
    return load_pocket_rules(ПАКЕТ), эталон, pocket_heavy_atoms(ПАКЕТ)


@pytest.fixture(scope="module")
def молекулы() -> list[Chem.Mol]:
    """Первые пять разобранных молекул прогона — больше для проверки свойств не нужно."""
    отобранные: list[Chem.Mol] = []
    with open_sdf(ПРОГОН) as supplier:
        for mol in supplier:
            if mol is not None:
                отобранные.append(mol)
            if len(отобранные) == 5:
                break
    return отобранные


def test_скор_не_ухудшается(мишень: tuple, молекулы: list[Chem.Mol]) -> None:
    """Возвращается лучшая поза траектории, а не последняя, поэтому ухудшение невозможно."""
    карман, эталон, атомы = мишень
    for номер, mol in enumerate(молекулы):
        итог = optimize_pose(mol, карман, эталон, атомы, steps=50, seed=номер)
        assert итог.score_after >= итог.score_before


def test_предел_смещения_соблюдается(мишень: tuple, молекулы: list[Chem.Mol]) -> None:
    """Молекула не уходит дальше заданного RMSD: иначе оптимизация ушла бы в чужой карман."""
    карман, эталон, атомы = мишень
    предел = 0.8
    for номер, mol in enumerate(молекулы):
        итог = optimize_pose(mol, карман, эталон, атомы, steps=100, seed=номер, max_rmsd=предел)
        assert heavy_atom_rmsd(итог.mol, mol) <= предел + 1e-9


def test_контакт_с_белком_не_ухудшается(мишень: tuple, молекулы: list[Chem.Mol]) -> None:
    """Потенциал «больше контактов — лучше» вжимал бы молекулу в белок без этой проверки.

    Правило двустороннее и проверяется как есть: хорошая поза не пускается за порог
    наложения, уже налезшая — не делается хуже, чем пришла. Второе существенно:
    у 84 молекул прогона из 100 тяжёлый атом стоит к белку ближе 2.0 Å, и требование
    «нет наложения» просто запретило бы им любое движение.
    """
    карман, эталон, атомы = мишень
    for номер, mol in enumerate(молекулы):
        итог = optimize_pose(mol, карман, эталон, атомы, steps=100, seed=номер)
        дно = min(CLASH_DISTANCE_A, min_pocket_distance(mol, атомы))
        assert min_pocket_distance(итог.mol, атомы) >= дно - 1e-9


def test_пустой_эталон_молекулу_не_двигает(мишень: tuple, молекулы: list[Chem.Mol]) -> None:
    """Отрицательный контроль: без бит у эталона двигать молекулу нечем.

    Это та же проверка, что даёт перемешанный эталон на всём прогоне, только
    в предельном виде: скор тождественно нулевой, и любое смещение означало бы,
    что процедура двигает позу сама по себе.
    """
    карман, _, атомы = мишень
    пустой = np.zeros((7, 85), dtype=bool)
    итог = optimize_pose(молекулы[0], карман, пустой, атомы, steps=100, seed=0)
    assert итог.score_after == 0.0
    assert итог.rmsd_shift == 0.0


def test_конформация_не_меняется(мишень: tuple, молекулы: list[Chem.Mol]) -> None:
    """Движение твёрдотельное: внутренние расстояния сохраняются до численной точности.

    Отсюда прямое следствие для контрольных метрик работы: лекарствоподобие,
    синтетическая доступность и число тяжёлых атомов не могут измениться от такой
    оптимизации вообще — молекула та же, изменилось только её положение.
    """
    карман, эталон, атомы = мишень
    mol = молекулы[0]
    итог = optimize_pose(mol, карман, эталон, атомы, steps=100, seed=0)
    было = Chem.rdmolops.Get3DDistanceMatrix(mol)
    стало = Chem.rdmolops.Get3DDistanceMatrix(итог.mol)
    assert np.allclose(было, стало, atol=1e-6)


def test_клэш_ловится() -> None:
    """Проверка налезания срабатывает на заведомо наложенных атомах."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    Chem.rdDistGeom.EmbedMolecule(mol, randomSeed=0xF00D)
    центр = np.array([[0.0, 0.0, 0.0]])
    conf = mol.GetConformer()
    for и in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(и, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
    assert has_clash(mol, центр)
    далеко = np.array([[100.0, 0.0, 0.0]])
    assert not has_clash(mol, далеко)


def test_без_конформера_отказ(мишень: tuple) -> None:
    карман, эталон, атомы = мишень
    with pytest.raises(PoseOptimizationError, match="конформера"):
        optimize_pose(Chem.MolFromSmiles("CCO"), карман, эталон, атомы, steps=1)


def test_порог_налезания_меньше_химической_связи() -> None:
    """Порог отсекает наложение атомов, а не тесный контакт."""
    assert CLASH_DISTANCE_A < 2.2
