"""Проверки трёх вариантов построения потенциала (`kinase_ifp.potential`).

Главное, что здесь проверяется, — **сравнимость вариантов**. Три формулы имеют смысл
только тогда, когда считаются на одном и том же отпечатке и сводятся к одной базе:
взвешенный вариант обязан переходить в базовый при единичных весах, а вариант с членом
клэша — при отсутствии наложений. Если это не так, разница между вариантами перестаёт
быть разницей формул и становится разницей расчётов.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from kinase_ifp.bit_weights import BitWeightError, bit_counts, bit_weights
from kinase_ifp.config import (
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.fingerprint_klifs import ifp_from_groups, load_pocket_rules
from kinase_ifp.ligand_flags import groups_from_mol
from kinase_ifp.molecule_io import read_mol
from kinase_ifp.pose_optimization import pocket_heavy_atoms, pocket_vdw_radii
from kinase_ifp.potential import (
    POTENTIAL_VARIANTS,
    ClashPenaltyPotential,
    DiscretePotential,
    WeightedPotential,
    make_potential,
)
from kinase_ifp.protonate import prepare_ligand

ПАКЕТ = Path("data/targets/6tgu/target.json")


def _эталон(package_json: Path) -> np.ndarray:
    биты = json.loads(package_json.read_text(encoding="utf-8"))["klifs_ifp_bits"]
    массив = np.array([символ == "1" for символ in биты])
    return массив.reshape(KLIFS_IFP_SHAPE[1], KLIFS_IFP_SHAPE[0]).T


@pytest.fixture(scope="module")
def пакет() -> Path:
    if not ПАКЕТ.is_file():
        pytest.skip(f"нет фикстуры пакета мишени {ПАКЕТ}")
    return ПАКЕТ


@pytest.fixture(scope="module")
def кристалл(пакет: Path) -> tuple[Chem.Mol, np.ndarray]:
    """Кристаллический лиганд мишени и его отпечаток — верхняя планка любого варианта."""
    молекула = prepare_ligand(read_mol(пакет.parent / "ligand.sdf"))
    отпечаток = ifp_from_groups(groups_from_mol(молекула), load_pocket_rules(пакет))
    return молекула, отпечаток


def test_все_варианты_дают_единицу_на_кристалле(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """Кристаллический лиганд воспроизводит эталон целиком — значит верх шкалы 1.0.

    Это калибровка, общая для трёх вариантов: формула, не дающая единицы на молекуле,
    из которой эталон и получен, измеряет не то, что заявлено.
    """
    молекула, отпечаток = кристалл
    эталон = _эталон(пакет)
    for имя in POTENTIAL_VARIANTS:
        потенциал = make_potential(
            имя,
            эталон,
            pocket_atoms=pocket_heavy_atoms(пакет),
            pocket_radii=pocket_vdw_radii(пакет),
        )
        assert потенциал(молекула, отпечаток) == pytest.approx(1.0), (
            f"вариант {имя} не дал 1.0 на кристаллическом лиганде"
        )


def test_веса_из_единиц_дают_базовый_скор(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """М2 с единичными весами обязан совпасть с М0 — он его обобщение, а не замена.

    Проверяется на частичном отпечатке, а не на кристалле: на кристалле обе формулы
    дают 1.0 и совпали бы при любой ошибке во взвешивании.
    """
    молекула, полный = кристалл
    эталон = _эталон(пакет)
    # Половина бит позы гасится, чтобы скор был строго между 0 и 1.
    частичный = полный.copy()
    частичный[:, : частичный.shape[1] // 2] = 0

    базовый = DiscretePotential(reference_ifp=эталон)
    единичный = WeightedPotential(
        reference_ifp=эталон, weights=np.ones(KLIFS_IFP_SHAPE, dtype=float)
    )
    значение = базовый(молекула, частичный)
    assert 0.0 < значение < 1.0, "проверка бессмысленна на вырожденном скоре"
    assert единичный(молекула, частичный) == pytest.approx(значение)


def test_взвешивание_меняет_число_на_настоящих_весах(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """На частотных весах М2 обязан отличаться от М0, иначе вариант ничего не добавляет."""
    молекула, полный = кристалл
    эталон = _эталон(пакет)
    частичный = полный.copy()
    частичный[:, : частичный.shape[1] // 2] = 0

    базовый = DiscretePotential(reference_ifp=эталон)
    взвешенный = make_potential("weighted", эталон)
    assert взвешенный(молекула, частичный) != pytest.approx(базовый(молекула, частичный))


def test_штраф_клэша_нулевой_у_кристалла(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """Кристаллическая поза не нарушает ван-дер-ваальсовых границ — штраф обязан быть нулём.

    Это и есть проверка того, что порог `CLASH_VDW_FRACTION` выставлен вменяемо:
    штраф у структуры из PDB означал бы, что потенциал наказывает за физичность.
    """
    молекула, _ = кристалл
    потенциал = make_potential(
        "clash",
        _эталон(пакет),
        pocket_atoms=pocket_heavy_atoms(пакет),
        pocket_radii=pocket_vdw_radii(пакет),
    )
    assert isinstance(потенциал, ClashPenaltyPotential)
    assert потенциал.clash_penalty(молекула) == pytest.approx(0.0)


def test_штраф_клэша_растёт_при_вдавливании(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """Сдвиг молекулы внутрь белка обязан увеличивать штраф, а значение потенциала — падать.

    Без этого член клэша был бы декоративным: он должен именно направлять, а не только
    отличаться от нуля.
    """
    молекула, отпечаток = кристалл
    атомы = pocket_heavy_atoms(пакет)
    потенциал = make_potential(
        "clash", _эталон(пакет), pocket_atoms=атомы, pocket_radii=pocket_vdw_radii(пакет)
    )
    assert isinstance(потенциал, ClashPenaltyPotential)

    # Сдвиг к центру тяжести кармана гарантированно вгоняет лиганд в белок.
    сдвинутая = Chem.Mol(молекула)
    conf = сдвинутая.GetConformer()
    направление = атомы.mean(axis=0) - conf.GetPositions().mean(axis=0)
    направление = направление / np.linalg.norm(направление) * 2.0
    for индекс in range(сдвинутая.GetNumAtoms()):
        точка = conf.GetAtomPosition(индекс)
        conf.SetAtomPosition(
            индекс,
            Chem.rdGeometry.Point3D(
                точка.x + направление[0], точка.y + направление[1], точка.z + направление[2]
            ),
        )

    assert потенциал.clash_penalty(сдвинутая) > потенциал.clash_penalty(молекула)
    assert потенциал(сдвинутая, отпечаток) < потенциал(молекула, отпечаток)


def test_clash_без_кармана_отказывается_собираться(пакет: Path) -> None:
    """Вариант `clash` без координат кармана обязан упасть, а не подменяться базовым.

    Подмена дала бы таблицу с меткой «clash», посчитанную формулой М0, и отличить её
    от настоящей было бы нечем.
    """
    with pytest.raises(ValueError, match="clash"):
        make_potential("clash", _эталон(пакет))


def test_неизвестный_вариант_отвергается(пакет: Path) -> None:
    with pytest.raises(ValueError, match="неизвестный вариант"):
        make_potential("м5", _эталон(пакет))


def test_радиусы_кармана_совпадают_по_числу_с_координатами(пакет: Path) -> None:
    """Порядок и число атомов у координат и радиусов обязаны совпадать.

    Расхождение здесь дало бы попарные пороги, приписанные не тем атомам, и заметить
    это по итоговому числу было бы невозможно.
    """
    координаты = pocket_heavy_atoms(пакет)
    радиусы = pocket_vdw_radii(пакет)
    assert len(координаты) == len(радиусы)
    assert радиусы.min() > 1.0, "ван-дер-ваальсов радиус тяжёлого атома не бывает меньше 1 A"
    assert радиусы.max() < 3.0


def test_веса_klifs_в_допустимых_границах() -> None:
    """Веса лежат в (0, 1], и ни один бит не выпадает из счёта полностью.

    Нижняя граница строго больше нуля — это и есть работа сглаживания: бит с нулевой
    частотой обязан сохранить малый, но ненулевой вес, иначе он перестал бы требоваться.
    """
    try:
        веса = bit_weights()
    except BitWeightError:
        pytest.skip("таблица отпечатков KLIFS недоступна")
    assert веса.shape == KLIFS_IFP_SHAPE
    assert веса.min() > 0.0
    assert веса.max() == pytest.approx(1.0)


def test_частоты_бит_согласованы_с_химией_кармана() -> None:
    """У типов, невозможных на большинстве позиций, частоты обязаны быть заметно ниже.

    Проверка содержательная, а не формальная: ионные взаимодействия в киназном кармане
    редки, гидрофобные есть почти всегда. Если бы раскладка (85, 7) была перепутана,
    это соотношение развалилось бы — тем же способом была поймана неверная раскладка.
    """
    try:
        счёт = bit_counts()
    except BitWeightError:
        pytest.skip("таблица отпечатков KLIFS недоступна")
    гидрофобные = счёт[KLIFS_INTERACTION_TYPES.index("HYD")].sum()
    ионные = счёт[KLIFS_INTERACTION_TYPES.index("ION-")].sum()
    assert гидрофобные > 10 * ионные


def test_скор_по_типам_не_видит_гидрофобных(
    пакет: Path, кристалл: tuple[Chem.Mol, np.ndarray]
) -> None:
    """М2 обязан считаться только по `SCORING_INTERACTION_TYPES`, как и М0.

    Гидрофобный бит — самый частый в базе, поэтому случайное включение `HYD`
    в взвешенный скор изменило бы его сильнее всего и осталось бы незамеченным.
    """
    молекула, отпечаток = кристалл
    эталон = _эталон(пакет)
    испорченный = отпечаток.copy()
    испорченный[KLIFS_INTERACTION_TYPES.index("HYD")] = 0

    взвешенный = make_potential("weighted", эталон, scoring_types=SCORING_INTERACTION_TYPES)
    assert взвешенный(молекула, испорченный) == pytest.approx(взвешенный(молекула, отпечаток))
