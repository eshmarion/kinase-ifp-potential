"""Внешняя оценка позы докингом (`src/kinase_ifp/docking.py`).

Калибровка здесь та же, что у PoseBusters и у метрики DiffInt: измерение считается
настроенным, когда кристаллическая поза получает величину, осмысленную для настоящего
ингибитора. Абсолютного порога у энергии Vina нет, поэтому проверяется не «сошлось
с числом», а диапазон, вне которого расчёт заведомо сломан.

Тесты пропускаются, если `vina` или `meeko` не установлены: нативная установка без них
остаётся рабочей, а в контейнере они идут из `uv.lock`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

vina = pytest.importorskip("vina", reason="нет пакета vina")
pytest.importorskip("meeko", reason="нет пакета meeko")

from kinase_ifp.docking import (  # noqa: E402
    DockingError,
    DockingScore,
    PoseScorer,
    ligand_pdbqt,
)
from kinase_ifp.protonate import prepare_ligand  # noqa: E402

ПАКЕТ = Path("tests/fixtures/targets/6tgu/target.json")
if not ПАКЕТ.is_file():
    ПАКЕТ = Path("data/targets/6tgu/target.json")

pytestmark = pytest.mark.skipif(not ПАКЕТ.is_file(), reason="нет пакета мишени 6tgu")


def _лиганд() -> Chem.Mol:
    mol = Chem.MolFromMolFile(str(ПАКЕТ.parent / "ligand.sdf"), removeHs=False)
    assert mol is not None
    return prepare_ligand(mol)


@pytest.fixture(scope="module")
def оценщик(tmp_path_factory: pytest.TempPathFactory) -> PoseScorer:
    """Один оценщик на модуль: карты Vina считаются секунды, а от теста не зависят."""
    return PoseScorer(ПАКЕТ, tmp_path_factory.mktemp("docking"))


def test_кристаллическая_поза_получает_осмысленную_энергию(оценщик: PoseScorer) -> None:
    оценка = оценщик.score(_лиганд())
    assert -15.0 < оценка.total < -5.0, f"энергия вне разумного диапазона: {оценка.total}"
    assert оценка.inter < 0.0, "межмолекулярный член кристаллической позы обязан быть притяжением"
    assert 0.2 < оценка.ligand_efficiency < 0.6, оценка.ligand_efficiency


def test_поза_вне_бокса_отвергается(оценщик: PoseScorer) -> None:
    """Молчаливый нуль вместо отказа исказил бы выборку сильнее, чем потеря молекулы."""
    mol = Chem.Mol(_лиганд())
    conf = mol.GetConformer()
    for и in range(mol.GetNumAtoms()):
        точка = conf.GetAtomPosition(и)
        conf.SetAtomPosition(и, (точка.x + 100.0, точка.y, точка.z))
    with pytest.raises(DockingError, match="за бокс"):
        оценщик.score(mol)


def test_карты_переживают_смену_лиганда(оценщик: PoseScorer) -> None:
    """Оценка сотни поз опирается на то, что карты считаются один раз на мишень.

    Если смена лиганда требовала бы пересчёта карт, число второй молекулы поехало бы
    молча — поэтому проверяется, что повторная оценка того же лиганда даёт то же число.
    """
    первая = оценщик.score(_лиганд())
    другая = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"), addCoords=True)
    Chem.rdDistGeom.EmbedMolecule(другая, randomSeed=0xF00D)
    conf = другая.GetConformer()
    центр = _лиганд().GetConformer().GetPositions().mean(axis=0)
    смещение = центр - другая.GetConformer().GetPositions().mean(axis=0)
    for и in range(другая.GetNumAtoms()):
        точка = conf.GetAtomPosition(и)
        conf.SetAtomPosition(
            и, (точка.x + смещение[0], точка.y + смещение[1], точка.z + смещение[2])
        )
    оценщик.score(другая)
    повтор = оценщик.score(_лиганд())
    assert повтор.total == pytest.approx(первая.total, abs=1e-6)


def test_лигандная_эффективность_нормирует_на_размер() -> None:
    оценка = DockingScore(total=-8.0, inter=-10.0, intra=0.0, n_heavy_atoms=20)
    assert оценка.ligand_efficiency == pytest.approx(0.4)


def test_молекула_без_конформера_отвергается() -> None:
    with pytest.raises(DockingError, match="конформера"):
        ligand_pdbqt(Chem.MolFromSmiles("CCO"))
