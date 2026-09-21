"""Тесты расчёта отпечатка взаимодействий."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from kinase_ifp.config import (
    KLIFS_IFP_LENGTH,
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.fingerprint import IfpError, bits_to_ifp, compute_ifp, ifp_to_bits
from kinase_ifp.pocket import Pocket
from kinase_ifp.protonate import add_explicit_hydrogens, prepare_ligand

_HYD = KLIFS_INTERACTION_TYPES.index("HYD")
_DON = KLIFS_INTERACTION_TYPES.index("DON")
_ACC = KLIFS_INTERACTION_TYPES.index("ACC")


def test_форма_и_тип_отпечатка(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    ifp = compute_ifp(pocket, crystal_ligand)

    assert ifp.shape == KLIFS_IFP_SHAPE
    assert ifp.dtype == np.uint8
    assert set(np.unique(ifp)) <= {0, 1}


def test_есть_водородная_связь(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    """Хотя бы один бит DON или ACC у кристаллического лиганда."""
    ifp = compute_ifp(pocket, crystal_ligand)

    assert ifp[_DON].sum() + ifp[_ACC].sum() > 0


def test_отпечаток_не_только_гидрофобный(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    """Регрессия на дефект черновика: без водородов остаются одни гидрофобные контакты."""
    ifp = compute_ifp(pocket, crystal_ligand)

    specific = sum(ifp[KLIFS_INTERACTION_TYPES.index(t)].sum() for t in SCORING_INTERACTION_TYPES)
    assert specific > 0, "все биты пришлись на HYD — признак расчёта без водородов"


def test_лиганд_без_водородов_отвергается(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    """Молчаливый пустой отпечаток — то, из-за чего черновик считал неверный IFP."""
    without_hydrogens = Chem.RemoveHs(crystal_ligand)

    with pytest.raises(IfpError, match="водород"):
        compute_ifp(pocket, without_hydrogens)


def test_лиганд_без_конформации_отвергается(pocket: Pocket) -> None:
    with pytest.raises(IfpError, match="конформации"):
        compute_ifp(pocket, Chem.AddHs(Chem.MolFromSmiles("CCO")))


def test_неизвестный_режим_протонирования_отвергается(
    pocket: Pocket, crystal_ligand: Chem.Mol
) -> None:
    broken = Pocket(**{**pocket.__dict__, "protonation": "какой-то-свой"})

    with pytest.raises(IfpError, match="протонирования"):
        compute_ifp(broken, crystal_ligand)


def test_биты_и_массив_переводятся_друг_в_друга(pocket: Pocket) -> None:
    assert pocket.reference_ifp is not None
    bits = ifp_to_bits(pocket.reference_ifp)

    assert len(bits) == KLIFS_IFP_LENGTH
    assert np.array_equal(bits_to_ifp(bits), pocket.reference_ifp)


def test_строка_неверной_длины_отвергается() -> None:
    with pytest.raises(IfpError, match="длиной"):
        bits_to_ifp("0101")


def test_раскладка_строки_klifs_позиционная(pocket: Pocket) -> None:
    """Проверка раскладки 595-битной строки — по химии, а не по совпадению с расчётом.

    Эталон 6tgu ставит ароматические взаимодействия на позиции 45 (PHE114) и 74 (HIS161)
    и катионное — на 17 (LYS69, каталитический лизин). Это единственная раскладка,
    при которой ароматика приходится на ароматические остатки: при обратной, (7, 85),
    те же биты попадали бы на VAL66, LEU86 и GLY47, у которых ни кольца, ни заряда.
    """
    reference = pocket.reference_ifp
    assert reference is not None
    position_of = pocket.residue_to_position

    assert reference[KLIFS_INTERACTION_TYPES.index("F-F"), position_of["PHE114.A"] - 1] == 1
    assert reference[KLIFS_INTERACTION_TYPES.index("F-E"), position_of["HIS161.A"] - 1] == 1
    assert reference[KLIFS_INTERACTION_TYPES.index("ION+"), position_of["LYS69.A"] - 1] == 1


def test_отпечаток_согласован_с_эталоном_klifs(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    """Дымовая сверка: наши биты должны быть подмножеством эталонных.

    KLIFS считает отпечатки сторонней программой (FingerPrintLib) с другими правилами,
    поэтому совпадения 1:1 не ждём и порога здесь не назначаем — полная сверка
    с числом идёт в полную сверку. Но ложное срабатывание, то есть бит, которого у KLIFS нет,
    означало бы ошибку раскладки или адресации остатков, а не разницу правил.
    """
    ifp = compute_ifp(pocket, crystal_ligand)
    reference = pocket.reference_ifp
    assert reference is not None

    ours, theirs = ifp.astype(bool), reference.astype(bool)
    assert ours.sum() > 0
    false_positives = np.flatnonzero((ours & ~theirs).reshape(-1))
    assert not false_positives.size, (
        f"бит(ы) {false_positives.tolist()} есть у нас и отсутствуют у KLIFS — "
        f"признак неверной раскладки или адресации остатков"
    )


def test_направленность_донора_и_акцептора_не_переставлена(
    pocket: Pocket, crystal_ligand: Chem.Mol
) -> None:
    """Закрывает `Н3б`: перестановка `HBDonor` и `HBAcceptor` обязана валить тест.

    Имя типа в KLIFS описывает роль БЕЛКА, в ProLIF — роль ЛИГАНДА, поэтому
    соответствие в `KLIFS_TO_PROLIF` инвертировано намеренно. Ошибка здесь не ломает
    ни один расчёт: отпечаток выходит зеркальным при исправном коде, и до 14.09
    направленность держалась только разовым измерением, а не механизмом
    (`docs/audit-findings.md`, `Н3б`).

    Проверяется химия, а не состав словаря. У кристаллического лиганда `6tgu` белок
    выступает донором дважды — LYS69 и ASP176 — и акцептором ни разу; то же говорит
    эталон KLIFS. При перестановке те же два бита переезжают в `ACC`, и измерение
    в `src/audit/t3.py` показывает, во что это обходится: скор нативной позы 1.000 → 0.600.
    """
    ifp = compute_ifp(pocket, crystal_ligand)
    reference = pocket.reference_ifp
    assert reference is not None

    assert ifp[_DON].sum() == 2, "белок-донор у 6tgu — LYS69 и ASP176"
    assert ifp[_ACC].sum() == 0, "белок-акцептором у 6tgu не выступает"
    assert reference[_DON].sum() == 2
    assert reference[_ACC].sum() == 0

    # Проверка на зеркальность: наши DON-биты обязаны стоять там же, где у KLIFS,
    # а не на позициях его ACC-бит. Именно это различает верную направленность
    # и переставленную — при перестановке первое условие нарушится.
    assert np.array_equal(ifp[_DON].astype(bool), reference[_DON].astype(bool))


def _как_из_diffsbdd(ligand: Chem.Mol) -> Chem.Mol:
    """Та же геометрия, но без разметки зарядов — так молекула приходит от модели."""
    editable = Chem.RWMol(Chem.RemoveHs(ligand))
    for atom in editable.GetAtoms():
        atom.SetFormalCharge(0)
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def test_подготовка_уравнивает_кристалл_и_генерацию(
    pocket: Pocket, crystal_ligand: Chem.Mol
) -> None:
    """Регрессия на В-12: путь молекулы не должен влиять на её отпечаток.

    Кристаллический лиганд приходит из KLIFS с размеченным карбоксилатом,
    сгенерированный — без разметки вовсе. Без нормализации зарядов второй теряет
    солевой мостик с каталитическим лизином, и таблица «было/стало» сравнивала бы
    два протокола подготовки вместо двух способов отбора.
    """
    из_кристалла = compute_ifp(pocket, crystal_ligand)
    из_генерации = compute_ifp(pocket, prepare_ligand(_как_из_diffsbdd(crystal_ligand)))

    assert np.array_equal(из_генерации, из_кристалла)


def test_ненормализованная_кислота_отвергается(pocket: Pocket, crystal_ligand: Chem.Mol) -> None:
    """Молекула, прошедшая мимо prepare_ligand, обязана падать, а не терять биты молча."""
    мимо_нормализации = add_explicit_hydrogens(_как_из_diffsbdd(crystal_ligand))

    with pytest.raises(IfpError, match="кислотная группа"):
        compute_ifp(pocket, мимо_нормализации)
