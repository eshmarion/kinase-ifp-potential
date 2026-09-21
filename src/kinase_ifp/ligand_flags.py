"""Флаги атомов лиганда по правилам KLIFS, вычисленные из молекулы RDKit.

Зачем модуль нужен. `klifs_rules.ligand_groups` читает лиганд только из mol2 самого
KLIFS: там есть типы SYBYL, а правило акцептора (`AFP::SetIsHA`) и формальные заряды
оригинала записаны через них. Для сгенерированных молекул этого пути нет и быть
не может — в KLIFS их нет по определению, DiffSBDD отдаёт SDF. Поэтому те же шесть
флагов считаются здесь заново, из RDKit, и это **единственный** способ применить
правила KLIFS к порождённой молекуле.

Что заменено и почему это допустимо. Три типа SYBYL, которые правила упоминают
по имени, выражаются через перцепцию RDKit:

  * `N.4` — азот с четырьмя связями к тяжёлым атомам и водородам;
  * `N.am` — амидный азот: связан с углеродом, несущим двойную связь к O или S;
  * `N.pl3` — плоский трёхсвязный азот: гибридизация SP2 при трёх связях
    (анилиновый и пиррольный азот, азот в сопряжении).

Формальный заряд берётся у RDKit напрямую, а не по таблице трёх типов SYBYL, и это
не упрощение, а **расширение**: таблица `KLIFS_RULE_SYBYL_FORMAL_CHARGE` пропускает
любую кислотную группу, не записанную типом `O.co2` — тетразол 5a6n, фосфат 4oav
. У молекулы, прошедшей `protonate.normalize_charges`, заряды
расставлены по состоянию при pH 7.4, то есть тем же способом, каким их видит OEChem
в оригинале.

Ароматичность и здесь, и в `klifs_rules` берётся у RDKit — там это объявлено
единственной заведомой подменой инструмента (OEChem у нас нет), и подмена одна
и та же, а не две разные.

Совпадение двух путей проверяется измерением, а не рассуждением: у кристаллического
лиганда есть оба представления (`ligand.mol2` и `ligand.sdf` пакета мишени), и
`tests/test_ligand_flags.py` сверяет отпечатки, посчитанные через SYBYL и через RDKit.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from rdkit import Chem

from kinase_ifp.config import (
    KLIFS_RULE_DONOR_ELEMENTS,
    KLIFS_RULE_HYDROPHOBE_ELEMENTS,
    KLIFS_RULE_METAL_ELEMENTS,
)
from kinase_ifp.klifs_rules import AromaticCentre, Donor, Groups, Точка

# Двойная связь к этим элементам делает соседний азот амидным (`N.am` оригинала).
_АМИДНЫЕ_АКЦЕПТОРЫ: Final[tuple[str, ...]] = ("O", "S")

# Столько связей у азота типа `N.4`.
_СВЯЗЕЙ_N4: Final[int] = 4

# Столько связей у плоского трёхсвязного азота `N.pl3`.
_СВЯЗЕЙ_NPL3: Final[int] = 3

# Меньше двух ароматических соседей — нормаль в атоме не строится
# (`klifs_rules._нормаль_ароматики`, то же правило).
_АРОМАТИЧЕСКИХ_СОСЕДЕЙ: Final[int] = 2


class LigandFlagsError(RuntimeError):
    """Флаги посчитать нельзя: молекула без конформера или без явных водородов."""


def _точка(conf: Chem.Conformer, индекс: int) -> Точка:
    позиция = conf.GetAtomPosition(индекс)
    return (позиция.x, позиция.y, позиция.z)


def _амидный_азот(атом: Chem.Atom) -> bool:
    """Азот при карбонильном или тиокарбонильном углероде — `N.am` оригинала."""
    for сосед in атом.GetNeighbors():
        if сосед.GetSymbol() != "C":
            continue
        for связь in сосед.GetBonds():
            if связь.GetBondType() != Chem.BondType.DOUBLE:
                continue
            другой = связь.GetOtherAtom(сосед)
            if другой.GetSymbol() in _АМИДНЫЕ_АКЦЕПТОРЫ:
                return True
    return False


def _плоский_трёхсвязный(атом: Chem.Atom) -> bool:
    """Азот типа `N.pl3`: три связи при плоской геометрии."""
    связей = атом.GetDegree() + атом.GetTotalNumHs()
    return связей == _СВЯЗЕЙ_NPL3 and атом.GetHybridization() == Chem.HybridizationType.SP2


def _делокализованные_анионы(mol: Chem.Mol) -> set[int]:
    """Индексы кислородов, анионных по делокализации заряда — тип `O.co2` оригинала.

    Правило `AFP::SetIsAnn` считает анионом не только атом с отрицательным заряду,
    но и **любой** атом типа `O.co2`, а MOE ставит этот тип обоим кислородам
    карбоксилата: заряд в группе делокализован, и каждый кислород несёт половину.
    RDKit же пишет −1 на одном кислороде и нуль на втором, карбонильном.

    Расхождение не теоретическое: у лиганда 6tgu к каталитическому лизину обращён
    именно карбонильный кислород, и без этого правила бит `ION+` на позиции 17
    пропадает, а эталон KLIFS его ставит.

    Правило распространяется на любую кислотную группу с терминальными кислородами
    при одном центре — карбоксилат, сульфонат, фосфат. Центр с положительным
    формальным зарядом исключён: у нитрогруппы кислород тоже несёт −1, но группа
    в целом заряда не имеет и ионным контактом в оригинале не считается.
    """
    анионные: set[int] = set()
    for атом in mol.GetAtoms():
        if атом.GetSymbol() != "O" or атом.GetDegree() != 1:
            continue
        центр = next(iter(атом.GetNeighbors()))
        if центр.GetFormalCharge() > 0:
            continue
        терминальные = [
            сосед
            for сосед in центр.GetNeighbors()
            if сосед.GetSymbol() == "O" and сосед.GetDegree() == 1
        ]
        if any(кислород.GetFormalCharge() < 0 for кислород in терминальные):
            анионные.update(кислород.GetIdx() for кислород in терминальные)
    return анионные


def is_acceptor(атом: Chem.Atom) -> bool:
    """Флаг `IsHA` правил KLIFS для атома молекулы RDKit.

    Повторяет `AFP::SetIsHA`: отрицательный заряд — акцептор независимо от элемента,
    положительный — никогда, нейтральные кислород и азот — да, кроме азота типов
    `N.pl3`, `N.am` и `N.4`, которые здесь распознаются перцепцией.
    """
    заряд = атом.GetFormalCharge()
    if заряд < 0:
        return True
    if заряд > 0:
        return False
    if атом.GetSymbol() == "O":
        return True
    if атом.GetSymbol() != "N":
        return False
    связей = атом.GetDegree() + атом.GetTotalNumHs()
    if связей >= _СВЯЗЕЙ_N4:
        return False
    return not (_амидный_азот(атом) or _плоский_трёхсвязный(атом))


def groups_from_mol(mol: Chem.Mol) -> Groups:
    """Раскладывает атомы молекулы RDKit по шести флагам правил KLIFS.

    Принимает молекулу с явными водородами и одним конформером — такую, какую отдаёт
    `protonate.prepare_ligand`. Возвращает `Groups`, пригодную для
    `klifs_rules.bits_for_residue` наравне с разбором mol2.

    Поднимает `LigandFlagsError`, если конформера нет или водороды неявные: правило
    донора требует сам атом водорода, и без него бит `DON` не поставить — молча
    вернуть неполный отпечаток здесь хуже, чем отказаться.
    """
    if mol.GetNumConformers() == 0:
        raise LigandFlagsError("у молекулы нет конформера: координат для правил нет")

    водородов = sum(1 for атом in mol.GetAtoms() if атом.GetAtomicNum() == 1)
    неявных = sum(атом.GetTotalNumHs() for атом in mol.GetAtoms())
    if водородов == 0 and неявных > 0:
        raise LigandFlagsError(
            f"водороды неявные ({неявных} шт.): правило донора требует сам атом "
            "водорода, зови protonate.prepare_ligand до расчёта"
        )

    conf = mol.GetConformer()
    делокализованные = _делокализованные_анионы(mol)
    гидрофобы: list[Точка] = []
    акцепторы: list[Точка] = []
    катионы: list[Точка] = []
    анионы: list[Точка] = []
    ароматика: list[AromaticCentre] = []
    водороды_донора: dict[int, list[Точка]] = {}

    for атом in mol.GetAtoms():
        индекс = атом.GetIdx()
        символ = атом.GetSymbol()
        if атом.GetAtomicNum() == 1:
            # Донор в оригинале — сам водород, но расстояние меряется от тяжёлого
            # соседа, поэтому водороды собираются по атому, на котором сидят.
            for сосед in атом.GetNeighbors():
                if сосед.GetSymbol() in KLIFS_RULE_DONOR_ELEMENTS:
                    водороды_донора.setdefault(сосед.GetIdx(), []).append(
                        _точка(conf, индекс)
                    )
                    break
            continue

        точка = _точка(conf, индекс)
        if символ in KLIFS_RULE_HYDROPHOBE_ELEMENTS:
            гидрофобы.append(точка)
        if is_acceptor(атом):
            акцепторы.append(точка)
        заряд = атом.GetFormalCharge()
        if заряд > 0 and символ not in KLIFS_RULE_METAL_ELEMENTS:
            катионы.append(точка)
        if заряд < 0 or индекс in делокализованные:
            анионы.append(точка)
        if атом.GetIsAromatic():
            соседи = [
                _точка(conf, сосед.GetIdx())
                for сосед in атом.GetNeighbors()
                if сосед.GetIsAromatic()
            ]
            нормаль = _нормаль(точка, соседи)
            if нормаль is not None:
                ароматика.append(
                    AromaticCentre(centroid=точка, normal=нормаль, atoms=(точка,))
                )

    return Groups(
        hydrophobes=tuple(гидрофобы),
        rings=tuple(ароматика),
        donors=tuple(
            Donor(heavy=_точка(conf, и), hydrogens=tuple(в))
            for и, в in sorted(водороды_донора.items())
        ),
        acceptors=tuple(акцепторы),
        cations=tuple(катионы),
        anions=tuple(анионы),
    )


def _нормаль(центр: Точка, соседи: list[Точка]) -> Точка | None:
    """Нормаль к плоскости кольца в самом атоме — как `NAroFind_Local` оригинала."""
    if len(соседи) < _АРОМАТИЧЕСКИХ_СОСЕДЕЙ:
        return None
    первый = np.asarray(соседи[0], dtype=float) - np.asarray(центр, dtype=float)
    второй = np.asarray(соседи[1], dtype=float) - np.asarray(центр, dtype=float)
    нормаль = np.cross(первый, второй)
    длина = float(np.linalg.norm(нормаль))
    if длина == 0.0:
        return None
    return tuple((нормаль / длина).tolist())
