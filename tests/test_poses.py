"""Набор поз без генерации.

Проверяется не «код отработал», а измеримые свойства набора: RMSD меряется
относительно кристалла, а не после выравнивания; лестница попадает в целевые
значения; лиганд не улетает из кармана; один сид даёт один и тот же набор.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

from kinase_ifp.config import (
    POSE_MAX_CENTROID_SHIFT_A,
    POSE_RMSD_RANGE_A,
    POSE_RMSD_TOLERANCE_A,
)
from kinase_ifp.poses import (
    POSE_KIND_CONFORMER,
    POSE_KIND_NATIVE,
    POSE_KIND_RIGID,
    generate_poses,
    heavy_atom_centroid,
    heavy_atom_rmsd,
    perturb_to_target_rmsd,
)
from kinase_ifp.protonate import UNNORMALIZED_ACID


def _сдвинуть(mol: Chem.Mol, dx: float, dy: float = 0.0, dz: float = 0.0) -> Chem.Mol:
    """Копия молекулы, сдвинутая как жёсткое тело."""
    copy = Chem.Mol(mol)
    conf = copy.GetConformer()
    for i in range(copy.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + dx, p.y + dy, p.z + dz))
    return copy


def test_rmsd_меряет_смещение_а_не_выравненную_форму(crystal_ligand: Chem.Mol) -> None:
    """Сдвиг жёсткого тела на 1 Å даёт RMSD ровно 1 Å.

    `GetBestRMS` сначала выровнял бы молекулы и вернул 0 — на наборе поз это
    обнулило бы ровно ту величину, которую мы измеряем.
    """
    assert heavy_atom_rmsd(_сдвинуть(crystal_ligand, 1.0), crystal_ligand) == pytest.approx(
        1.0, abs=1e-6
    )


def test_rmsd_не_учитывает_водороды(crystal_ligand: Chem.Mol) -> None:
    """Сдвиг одного водорода не меняет RMSD: граница 2 Å в докинге — по тяжёлым атомам."""
    moved = Chem.Mol(crystal_ligand)
    conf = moved.GetConformer()
    hydrogen = next(a.GetIdx() for a in moved.GetAtoms() if a.GetAtomicNum() == 1)
    p = conf.GetAtomPosition(hydrogen)
    conf.SetAtomPosition(hydrogen, Point3D(p.x + 5.0, p.y, p.z))

    assert heavy_atom_rmsd(moved, crystal_ligand) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("цель", [0.5, 2.0, 5.0])
def test_возмущение_попадает_в_целевой_rmsd(crystal_ligand: Chem.Mol, цель: float) -> None:
    """Заданный RMSD достигается с точностью до допуска на всём диапазоне лестницы."""
    поза = perturb_to_target_rmsd(crystal_ligand, crystal_ligand, цель, np.random.default_rng(0))

    assert heavy_atom_rmsd(поза, crystal_ligand) == pytest.approx(
        цель, abs=POSE_RMSD_TOLERANCE_A
    )


def test_центр_лиганда_не_уходит_из_кармана(crystal_ligand: Chem.Mol) -> None:
    """Сдвиг центра ограничен, даже когда целевой RMSD велик.

    Улетевший из кармана декой получил бы почти пустой отпечаток, и ROC-AUC мерил бы
    «есть контакты или нет», а не качество потенциала. Верх лестницы обязан
    набираться поворотом, а не переносом.
    """
    rng = np.random.default_rng(0)
    родной = heavy_atom_centroid(crystal_ligand)

    for цель in (1.0, 3.0, 5.0):
        поза = perturb_to_target_rmsd(crystal_ligand, crystal_ligand, цель, rng)
        сдвиг = float(np.linalg.norm(heavy_atom_centroid(поза) - родной))

        assert сдвиг <= POSE_MAX_CENTROID_SHIFT_A + 1e-6


def test_возмущение_не_меняет_внутреннюю_геометрию(crystal_ligand: Chem.Mol) -> None:
    """Поза двигается как жёсткое тело: длины связей сохраняются.

    Иначе испорченной оказалась бы сама молекула, а не её размещение, и декой
    отличался бы от нативной позы по химии, а не по положению в кармане.
    """
    поза = perturb_to_target_rmsd(crystal_ligand, crystal_ligand, 4.0, np.random.default_rng(1))
    было = crystal_ligand.GetConformer()
    стало = поза.GetConformer()

    for bond in crystal_ligand.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        assert rdMolTransforms.GetBondLength(стало, i, j) == pytest.approx(
            rdMolTransforms.GetBondLength(было, i, j), abs=1e-6
        )


@pytest.fixture(scope="module")
def набор(pocket) -> object:
    """Небольшой набор поз: двадцати хватает на все проверки, сотня растянула бы pytest."""
    return generate_poses(pocket.ligand_path, n=20, seed=0)


def test_набор_нужного_размера_и_нативная_поза_первая(набор) -> None:
    """Нативная поза обязана быть в наборе: без неё не считаются ни top-1, ни ROC-AUC."""
    assert len(набор.poses) == 20
    assert набор.poses[0].kind == POSE_KIND_NATIVE
    assert набор.poses[0].rmsd_to_ref == 0.0
    assert [p.kind for p in набор.poses[1:]].count(POSE_KIND_NATIVE) == 0


def test_лестница_покрывает_весь_диапазон(набор) -> None:
    """Искажения разложены по диапазону, а не сбиты в одну кучу.

    Непрерывная шкала нужна коэффициенту Спирмена: по набору из двух кучек
    он говорит лишь «группы различаются», то есть повторяет ROC-AUC.
    """
    низ, верх = POSE_RMSD_RANGE_A
    значения = sorted(p.rmsd_to_ref for p in набор.poses[1:])

    assert значения[0] == pytest.approx(низ, abs=POSE_RMSD_TOLERANCE_A)
    assert значения[-1] == pytest.approx(верх, abs=POSE_RMSD_TOLERANCE_A)
    # Обе стороны границы 2 Å заселены, иначе классы для ROC-AUC не разделены.
    assert sum(v < 2.0 for v in значения) >= 3
    assert sum(v > 2.0 for v in значения) >= 3


def test_у_каждой_позы_явные_водороды(набор) -> None:
    """Без явных водородов ProLIF не отличит донор от акцептора."""
    for поза in набор.poses:
        assert поза.mol.GetNumAtoms() > поза.mol.GetNumHeavyAtoms()


def test_один_сид_даёт_один_и_тот_же_набор(pocket) -> None:
    """Прогон воспроизводим: иначе числа в курсовой невозможно перепроверить."""
    первый = generate_poses(pocket.ligand_path, n=8, seed=7)
    второй = generate_poses(pocket.ligand_path, n=8, seed=7)

    assert [p.rmsd_to_ref for p in первый.poses] == [p.rmsd_to_ref for p in второй.poses]
    for a, b in zip(первый.poses, второй.poses, strict=True):
        assert np.allclose(
            a.mol.GetConformer().GetPositions(), b.mol.GetConformer().GetPositions()
        )


def test_разные_сиды_дают_разные_наборы(pocket) -> None:
    """Сид действительно управляет набором — иначе прогоны для top-1 не независимы."""
    первый = generate_poses(pocket.ligand_path, n=8, seed=0)
    второй = generate_poses(pocket.ligand_path, n=8, seed=1)

    assert not np.allclose(
        первый.poses[-1].mol.GetConformer().GetPositions(),
        второй.poses[-1].mol.GetConformer().GetPositions(),
    )


def test_часть_поз_построена_на_других_конформерах(набор) -> None:
    """В наборе есть и жёсткие сдвиги кристаллического конформера, и изменённые торсии."""
    виды = [p.kind for p in набор.poses]

    assert виды.count(POSE_KIND_CONFORMER) > 0
    assert виды.count(POSE_KIND_RIGID) > 0


def test_позы_проходят_подготовку_целиком_а_не_только_водороды(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Производитель зовёт `prepare_ligand`, то есть заряды и водороды.

    Проверяется на аспирине, а не на лиганде 6tgu: тот приходит из KLIFS с уже
    размеченным карбоксилатом, и пропущенная нормализация на нём не видна вовсе.
    Молекула с нейтральной кислотой ловит подмену сразу — `compute_ifp` на такой
    позе падает по `UNNORMALIZED_ACID`, то есть весь набор оказался бы негодным.
    """
    from rdkit.Chem import rdDistGeom

    аспирин = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert rdDistGeom.EmbedMolecule(аспирин, randomSeed=0) == 0
    # Каталог берётся у `tmp_path_factory` с латинским именем, а не у `tmp_path`:
    # последний называется по имени теста, имена тестов здесь русские, и путь
    # с кириллицей RDKit на Windows не открывает — «Bad input file».
    путь = tmp_path_factory.mktemp("poses") / "acid.sdf"
    with путь.open("w", encoding="utf-8", newline="") as поток:
        писатель = Chem.SDWriter(поток)
        писатель.write(аспирин)
        писатель.close()

    набор = generate_poses(путь, n=2, seed=0)

    assert len(набор.poses) == 2
    for поза in набор.poses:
        assert not поза.mol.HasSubstructMatch(UNNORMALIZED_ACID)
        assert any(a.GetFormalCharge() == -1 for a in поза.mol.GetAtoms())
        assert поза.mol.GetNumAtoms() > поза.mol.GetNumHeavyAtoms()
