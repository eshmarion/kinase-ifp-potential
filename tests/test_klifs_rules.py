"""Правила KLIFS: разбор mol2, шесть флагов атома, геометрия и сверка с эталоном.

Главная проверка здесь одна — `test_расчёт_совпадает_с_эталоном_бит_в_бит`. Остальные
закрепляют те места, где буквальное прочтение печатных таблиц расходится с исходником
FingerPrintLib: правило акцептора, источник ароматичности, точка отсчёта угла
водородной связи и то, что ионное расстояние меряется между атомами.
"""

from __future__ import annotations

import json
from math import cos, radians, sin
from pathlib import Path

import pytest

from kinase_ifp.config import KLIFS_INTERACTION_TYPES
from kinase_ifp.klifs_rules import (
    ПРАВИЛА_KLIFS,
    AromaticCentre,
    BitDifference,
    Donor,
    FingerprintComparison,
    Groups,
    KlifsRulesError,
    TypeCount,
    _акцептор,
    _есть_водородная_связь,
    _нормаль_ароматики,
    _разобрать_mol2,
    _угол_плоскостей,
    _формальный_заряд,
    bits_for_residue,
    compare_prepared,
    compare_with_reference,
    fingerprint_from_prepared,
    groups_from_mol2,
    hbond_candidates,
    ligand_groups,
    pocket_groups,
    prepare_package,
    ring_pairs,
)
from kinase_ifp.mol2_reader import Mol2Atom, Mol2ReadError, read_mol2
from kinase_ifp.pdb_reader import PdbReadError, read_pdb_residues

ФИКСТУРА = Path(__file__).parent / "fixtures" / "6tgu"


def _атом(sybyl: str, element: str) -> Mol2Atom:
    return Mol2Atom(
        name="X", sybyl=sybyl, element=element, xyz=(0.0, 0.0, 0.0), charge=0.0, residue="LIG1"
    )


@pytest.fixture(scope="module")
def эталон_6tgu() -> str:
    пакет = json.loads((ФИКСТУРА / "target.json").read_text(encoding="utf-8"))
    return str(пакет["klifs_ifp_bits"])


@pytest.fixture(scope="module")
def готовый_пакет(эталон_6tgu: str):
    return prepare_package(ФИКСТУРА / "target.json", эталон_6tgu, "6tgu")


# --- разборщик mol2 -------------------------------------------------------------


def test_mol2_читается_с_типами_зарядами_и_связями() -> None:
    структура = read_mol2(ФИКСТУРА / "pocket.mol2")
    assert len(структура.atoms) == 1446
    assert len(структура.bonds) == 1452
    первый = структура.atoms[0]
    assert (первый.name, первый.sybyl, первый.element, первый.residue) == (
        "N",
        "N.3",
        "N",
        "ARG44",
    )


def test_mol2_сохраняет_водороды() -> None:
    структура = read_mol2(ФИКСТУРА / "pocket.mol2")
    assert sum(1 for а in структура.atoms if а.element == "H") == 745


def test_элемент_берётся_из_типа_sybyl_а_не_из_имени_атома() -> None:
    # Имя атома в белке позиционное: у CB первая буква C, но это ни о чём не говорит
    # ни для CE1, ни для хлора лиганда. Элемент даёт только тип.
    структура = read_mol2(ФИКСТУРА / "ligand.mol2")
    assert {а.element for а in структура.atoms} <= {
        "C",
        "N",
        "O",
        "S",
        "H",
        "CL",
        "F",
        "BR",
        "I",
    }


def test_битый_mol2_не_подменяется_молча(tmp_path: Path) -> None:
    пустой = tmp_path / "пусто.mol2"
    пустой.write_text("@<TRIPOS>MOLECULE\nничего\n", encoding="utf-8")
    with pytest.raises(Mol2ReadError):
        read_mol2(пустой)
    with pytest.raises(Mol2ReadError):
        read_mol2(tmp_path / "нет-такого.mol2")


# --- шесть флагов атома ---------------------------------------------------------


def test_амидный_и_планарный_азот_акцепторами_не_считаются() -> None:
    # AFP::SetIsHA исключает N.pl3, N.am и N.4 дословно: у амидного азота
    # неподелённая пара уходит в сопряжение, и акцептором он не работает.
    assert not _акцептор(_атом("N.am", "N"))
    assert not _акцептор(_атом("N.pl3", "N"))
    assert not _акцептор(_атом("N.4", "N"))
    assert _акцептор(_атом("N.2", "N"))
    assert _акцептор(_атом("N.ar", "N"))


def test_кислород_акцептор_а_углерод_нет() -> None:
    assert _акцептор(_атом("O.2", "O"))
    assert _акцептор(_атом("O.3", "O"))
    assert not _акцептор(_атом("C.3", "C"))


def test_отрицательный_атом_акцептор_а_положительный_нет() -> None:
    # Первая ветка SetIsHA — GetFormalCharge()<0, без проверки элемента вообще.
    assert _акцептор(_атом("O.co2", "O"))
    assert not _акцептор(_атом("N.4", "N"))


def test_формальный_заряд_берётся_из_типа_sybyl() -> None:
    # Колонка зарядов mol2 несёт парциальные заряды MOE, а не формальные.
    assert _формальный_заряд("O.co2") == -1
    assert _формальный_заряд("N.4") == 1
    assert _формальный_заряд("C.cat") == 1
    assert _формальный_заряд("C.3") == 0


def test_ароматичность_не_читается_из_типа_атома() -> None:
    # MOE помечает `.ar` только шестичленные кольца: имидазол гистидина и пятичленное
    # кольцо триптофана записаны кекулевскими связями. Флаг берётся у перцепции.
    структура, _, ароматичность = _разобрать_mol2(ФИКСТУРА / "pocket.mol2")
    помечено = sum(1 for а in структура.atoms if а.sybyl.endswith(".ar"))
    воспринято = sum(ароматичность)
    assert помечено == 48
    assert воспринято == 66
    гистидин = [
        и
        for и, а in enumerate(структура.atoms)
        if а.residue == "HIS161" and а.name in ("CG", "ND1", "CD2", "CE1", "NE2")
    ]
    assert гистидин
    assert all(ароматичность[и] for и in гистидин)


def test_карман_разбирается_целиком() -> None:
    белок = pocket_groups(ФИКСТУРА / "pocket.mol2", "A")
    assert len(белок) == 84
    assert all(группы.hydrophobes or группы.donors for группы in белок.values())


def test_лиганд_разбирается_с_водородами_klifs() -> None:
    лиганд = ligand_groups(ФИКСТУРА / "ligand.mol2")
    assert лиганд.rings
    assert лиганд.donors
    assert any(д.hydrogens for д in лиганд.donors)


def test_донор_это_водород_при_кислороде_азоте_или_сере() -> None:
    структура, соседи, ароматичность = _разобрать_mol2(ФИКСТУРА / "pocket.mol2")
    индексы = [и for и, а in enumerate(структура.atoms) if а.residue == "LYS69"]
    группы = groups_from_mol2(структура, соседи, ароматичность, индексы)
    # У лизина донорами служат амидный азот остова и заряженная аминогруппа.
    assert len(группы.donors) >= 2


# --- геометрия ------------------------------------------------------------------


def _центр(z: float, нормаль: tuple[float, float, float]) -> AromaticCentre:
    точка = (0.0, 0.0, z)
    return AromaticCentre(centroid=точка, normal=нормаль, atoms=(точка,))


def test_угол_плоскостей_не_зависит_от_знака_нормали() -> None:
    assert _угол_плоскостей((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)) == pytest.approx(0.0)
    assert _угол_плоскостей((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)) == pytest.approx(90.0)


def test_нормаль_строится_по_двум_ароматическим_соседям() -> None:
    нормаль = _нормаль_ароматики((0.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    assert нормаль is not None
    assert abs(нормаль[2]) == pytest.approx(1.0)


def test_одного_соседа_для_нормали_мало() -> None:
    # В оригинале итератор в этом случае уходит за конец списка: воспроизводить нечего.
    assert _нормаль_ароматики((0.0, 0.0, 0.0), [(1.0, 0.0, 0.0)]) is None


def test_ароматические_биты_делятся_углом_а_не_расстоянием() -> None:
    остаток = Groups(rings=(_центр(0.0, (0.0, 0.0, 1.0)),))
    лицом = Groups(rings=(_центр(3.5, (0.0, 0.0, 1.0)),))
    ребром = Groups(rings=(_центр(3.5, (1.0, 0.0, 0.0)),))
    assert bits_for_residue(остаток, лицом)[1:3] == (True, False)
    assert bits_for_residue(остаток, ребром)[1:3] == (False, True)


def test_ароматический_центр_это_атом_а_не_центр_кольца(готовый_пакет) -> None:
    # При LcloRng=false центром свойства служит сам атом, поэтому «расстояние между
    # центрами» и «ближайшие атомы» — одна и та же величина, а не две разных.
    пары = ring_pairs(готовый_пакет)
    assert пары
    for пара in пары:
        assert пара.centroid_distance == pytest.approx(пара.min_atom_distance)


def test_водородная_связь_требует_и_расстояния_и_угла() -> None:
    донор = Donor(heavy=(0.0, 0.0, 0.0), hydrogens=((1.0, 0.0, 0.0),))
    близко_и_прямо = [(3.0, 0.0, 0.0)]
    близко_но_вбок = [(0.0, 3.0, 0.0)]
    далеко_но_прямо = [(4.0, 0.0, 0.0)]
    assert _есть_водородная_связь([донор], близко_и_прямо, ПРАВИЛА_KLIFS)
    assert not _есть_водородная_связь([донор], близко_но_вбок, ПРАВИЛА_KLIFS)
    assert not _есть_водородная_связь([донор], далеко_но_прямо, ПРАВИЛА_KLIFS)


def test_угол_связи_меряется_от_донора_а_не_от_водорода() -> None:
    # D-H вдоль X, акцептор под 40 градусов от донора: по правилу оригинала связь
    # есть (40 < 45). Если мерить D-H против H...A, угол выходит больше и бита нет.
    донор = Donor(heavy=(0.0, 0.0, 0.0), hydrogens=((1.0, 0.0, 0.0),))
    угол = radians(40.0)
    акцептор = [(3.0 * cos(угол), 3.0 * sin(угол), 0.0)]
    assert _есть_водородная_связь([донор], акцептор, ПРАВИЛА_KLIFS)


def test_ионный_бит_считается_между_атомами_по_порогу_из_исходника() -> None:
    остаток = Groups(anions=((0.0, 0.0, 0.0),))
    близко = Groups(cations=((0.0, 0.0, 3.9),))
    далеко = Groups(cations=((0.0, 0.0, 4.1),))
    assert bits_for_residue(остаток, близко)[6]
    assert not bits_for_residue(остаток, далеко)[6]


# --- отпечаток и сверка ---------------------------------------------------------


def test_порядок_бит_совпадает_с_типами_klifs() -> None:
    assert KLIFS_INTERACTION_TYPES == ("HYD", "F-F", "F-E", "DON", "ACC", "ION+", "ION-")


def test_длина_отпечатка_и_пустые_позиции(готовый_пакет) -> None:
    строка = fingerprint_from_prepared(готовый_пакет)
    assert len(строка) == 595
    assert set(строка) <= {"0", "1"}


def test_расчёт_совпадает_с_эталоном_бит_в_бит(готовый_пакет) -> None:
    итог = compare_prepared(готовый_пакет)
    assert итог.identical, f"расхождения: {итог.differences}"
    assert итог.reference_total == 17


def test_гидрофобные_биты_эталона_воспроизводятся_полностью(готовый_пакет) -> None:
    итог = compare_prepared(готовый_пакет)
    гидрофобные = next(с for с in итог.by_type if с.interaction == "HYD")
    assert гидрофобные.shared == гидрофобные.reference


def test_сверка_отвергает_строку_неверной_длины() -> None:
    with pytest.raises(KlifsRulesError):
        compare_with_reference("x", "1" * 10, "0" * 595)


def test_сверка_находит_позицию_и_тип_расхождения() -> None:
    наши = ["0"] * 595
    эталон = ["0"] * 595
    наши[7 * 16 + 3] = "1"
    итог = compare_with_reference("x", "".join(наши), "".join(эталон))
    assert len(итог.differences) == 1
    расхождение = итог.differences[0]
    assert (расхождение.position, расхождение.interaction) == (17, "DON")


def test_пакет_без_разметки_позиций_отвергается(tmp_path: Path) -> None:
    (tmp_path / "target.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(KlifsRulesError):
        prepare_package(tmp_path / "target.json", "0" * 595, "пусто")


def test_пакет_без_mol2_не_считается_по_запасному_пути(tmp_path: Path) -> None:
    # Запасного пути через PDB и SDF нет намеренно: без типов SYBYL правило
    # акцептора не воспроизвести, а водороды пришлось бы достраивать самим.
    (tmp_path / "target.json").write_text(
        json.dumps({"residue_to_position": {"ARG44.A": 1}}), encoding="utf-8"
    )
    with pytest.raises(KlifsRulesError):
        prepare_package(tmp_path / "target.json", "0" * 595, "без-mol2")


def test_пары_колец_несут_геометрию_и_бит_эталона(готовый_пакет) -> None:
    пары = ring_pairs(готовый_пакет)
    assert пары
    assert all(0.0 <= п.plane_angle <= 90.0 for п in пары)


def test_кандидаты_водородных_связей_обе_направленности(готовый_пакет) -> None:
    кандидаты = hbond_candidates(готовый_пакет)
    assert {к.direction for к in кандидаты} == {"DON", "ACC"}
    assert all(0.0 <= к.deviation <= 180.0 for к in кандидаты)


# --- разборщик PDB остался у правила apolar --------------------------------------


def test_разборщик_pdb_возвращает_водороды_и_имена() -> None:
    остатки = read_pdb_residues(ФИКСТУРА / "protein.pdb")
    assert "LYS69.A" in остатки
    assert any(а.element == "H" for атомы in остатки.values() for а in атомы)


def test_разборщик_pdb_сообщает_об_отсутствии_файла(tmp_path: Path) -> None:
    with pytest.raises(PdbReadError):
        read_pdb_residues(tmp_path / "нет-такого.pdb")


def test_выгрузка_в_реестр_несёт_сводку_и_типы() -> None:
    """Раздел 5.1 держится на трёх числах, и у них должен быть машинный источник.

    191 эталонный бит, 190 воспроизведённых и 143 гидрофобных до 19.09 находились
    сверкой только в прозе отчёта. Считаются
    они здесь, поэтому здесь же и записываются.
    """
    import importlib.util

    путь = Path(__file__).resolve().parents[1] / "scripts" / "klifs_rules.py"
    спецификация = importlib.util.spec_from_file_location("klifs_rules_cli", путь)
    assert спецификация and спецификация.loader
    модуль = importlib.util.module_from_spec(спецификация)
    спецификация.loader.exec_module(модуль)

    типы = KLIFS_INTERACTION_TYPES
    первая = FingerprintComparison(
        label="6tgu",
        differences=(),
        by_type=tuple(
            TypeCount(interaction=т, ours=5 if т == "HYD" else 0,
                      reference=5 if т == "HYD" else 0, shared=5 if т == "HYD" else 0)
            for т in типы
        ),
    )
    вторая = FingerprintComparison(
        label="8aoj",
        differences=(BitDifference(position=80, interaction="DON", ours=False, reference=True),),
        by_type=tuple(
            TypeCount(interaction=т, ours=0, reference=1 if т == "DON" else 0, shared=0)
            for т in типы
        ),
    )

    строки = модуль._измерения([первая, вторая], ПРАВИЛА_KLIFS, "проверка")
    по_ключу = {(с.target, с.variant, с.metric): с.value for с in строки}

    assert по_ключу[("e04b", "all", "reference_bits")] == "6"
    assert по_ключу[("e04b", "all", "shared_bits")] == "5"
    assert по_ключу[("e04b", "all", "structures")] == "2"
    assert по_ключу[("e04b", "all", "identical")] == "1"
    assert по_ключу[("e04b", "HYD", "shared")] == "5"
    assert по_ключу[("6tgu", "all", "reference_bits")] == "5"
    assert по_ключу[("8aoj", "all", "shared_bits")] == "0"
    assert all(с.notes.startswith("правила FingerPrintLib") for с in строки)
