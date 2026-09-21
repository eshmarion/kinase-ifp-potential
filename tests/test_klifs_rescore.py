"""Пересчёт прогона по правилам KLIFS: эталон, скор, отбор верхней доли.

Главная проверка — на кристаллическом лиганде 6tgu: по правилам KLIFS его отпечаток
совпадает с эталоном бит в бит, поэтому скор по всем семи типам, скор по направленным
и Танимото обязаны быть ровно 1.0. Если пересчёт взял не те правила или не тот набор
типов, это видно здесь первым.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from experiments.klifs_rescore import (
    KlifsMolecule,
    KlifsRescoreError,
    reference_ifp,
    rescore_run,
    selection_effect,
)
from experiments.layout import MOLECULES_SDF
from kinase_ifp.config import KLIFS_IFP_SHAPE
from kinase_ifp.protonate import prepare_ligand

ПАКЕТ = Path("tests/fixtures/targets/6tgu/target.json")
if not ПАКЕТ.is_file():
    ПАКЕТ = Path("data/targets/6tgu/target.json")


def _прогон_с_кристаллом(каталог: Path) -> Path:
    mol = Chem.MolFromMolFile(str(ПАКЕТ.parent / "ligand.sdf"), removeHs=False)
    assert mol is not None
    mol = prepare_ligand(mol)
    mol.SetProp("mol_id", "кристалл")
    писатель = Chem.SDWriter(str(каталог / MOLECULES_SDF))
    писатель.write(mol)
    писатель.close()
    return каталог


@pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")
def test_эталон_читается_в_раскладке_отпечатка() -> None:
    эталон = reference_ifp(ПАКЕТ)
    assert эталон.shape == KLIFS_IFP_SHAPE
    assert эталон.sum() > 0


@pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")
def test_кристаллический_лиганд_воспроизводит_эталон_полностью(tmp_path: Path) -> None:
    пересчёт = rescore_run(_прогон_с_кристаллом(tmp_path), ПАКЕТ)
    assert пересчёт.n_unreadable == 0
    [кристалл] = пересчёт.molecules
    assert кристалл.score_all7 == pytest.approx(1.0)
    assert кристалл.score_scoring6 == pytest.approx(1.0)
    assert кристалл.tanimoto == pytest.approx(1.0)
    assert кристалл.key_hbond == 1


def test_нативная_поза_без_отклонений_не_угадывается(tmp_path: Path) -> None:
    """Без `rmsd_to_ref` и метки `pose_kind` нативная поза не подменяется лучшей из набора."""
    from experiments.klifs_rescore import native_score

    (tmp_path / MOLECULES_SDF).write_text("", encoding="utf-8")
    with pytest.raises(KlifsRescoreError, match="rmsd_to_ref"):
        native_score(tmp_path, [_молекула(1, 0.5, 1)])


def test_прогон_без_молекул_отвергается(tmp_path: Path) -> None:
    with pytest.raises(KlifsRescoreError, match=MOLECULES_SDF):
        rescore_run(tmp_path, ПАКЕТ)


def _молекула(номер: int, скор: float, связь: int) -> KlifsMolecule:
    return KlifsMolecule(f"m{номер}", скор, скор, скор, связь, 20)


def test_отбор_берёт_верхнюю_долю_и_сравнивает_с_отбракованными() -> None:
    молекулы = [_молекула(i, i / 10, int(i >= 8)) for i in range(10)]
    топ, сравнения = selection_effect(молекулы, top_fraction=0.2)
    assert топ == 2
    по_имени = {с.name: с for с in сравнения}
    скор = по_имени["скор, все семь типов"]
    assert скор.top_value == pytest.approx(0.85)
    assert скор.all_value == pytest.approx(0.45)
    связь = по_имени["доля с ключевой водородной связью"]
    assert связь.all_value == pytest.approx(0.2)
    assert связь.top_value == pytest.approx(1.0)


def test_контрольные_характеристики_сравниваются_по_тем_же_топу_и_отбракованным() -> None:
    молекулы = [_молекула(i, i / 10, 0) for i in range(10)]
    контроли = {m.mol_id: {"qed": 0.5, "sa_score": 3.0, "n_heavy_atoms": 20.0} for m in молекулы}
    _, сравнения = selection_effect(молекулы, контроли, top_fraction=0.2)
    qed = next(с for с in сравнения if с.name == "qed")
    assert qed.p_value == pytest.approx(1.0)
