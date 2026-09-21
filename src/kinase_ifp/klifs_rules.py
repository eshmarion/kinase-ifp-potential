"""Воспроизведение эталонного отпечатка KLIFS по правилам самой FingerPrintLib.

Источник правил — **исходный код** библиотеки (`AFP.cc`, `AFP.hpp`,
`PropConstants.hpp`), которой KLIFS считает свои эталонные отпечатки, а не только
статья Marcou G., Rognan D. // J. Chem. Inf. Model. 2007. Vol. 47, № 1. P. 195–207.
Разница существенная: печатные таблицы 2 и 4 в трёх местах читаются не так, как
работает код, а руководство библиотеки приводит четыре разных набора умолчаний,
ни один из которых не совпадает с компилируемыми константами. Разбор — в
`docs/calibration.md`, раздел 7. Числа — в `config`, раздел `KLIFS_RULE_*`.

**Зачем модуль нужен и чем он не является.** Это не наш отпечаток и не замена ему.
Наш расчёт живёт в `fingerprint.compute_ifp`, идёт через ProLIF по нашим порогам
(`PROLIF_PARAMETERS`). Здесь считается другое: **что дал бы алгоритм KLIFS на тех же
координатах**. Проверка — совпадение бит в бит с `klifs_ifp_bits` пакета мишени.

**Обе стороны читаются из mol2 самого KLIFS, а не из PDB и не перцепцией RDKit.**
Флаги атома в оригинале определяются типом SYBYL и формальным зарядом: акцептором
не считается азот типов `N.pl3`, `N.am` и `N.4` (`AFP::SetIsHA` сравнивает
`GetType()` с этими строками дословно), ароматичность берётся у самого атома
(`X->IsAromatic()`, то есть тип `*.ar`), заряд — `GetFormalCharge()`. Ни того,
ни другого, ни третьего в PDB нет, и прежний разбор белка подставлял вместо них
наши шаблоны остатков — то есть нашу догадку вместо данных KLIFS. Теперь белок
читается из `pocket.mol2`, лиганд из `ligand.mol2`; оба файла KLIFS отдаёт сам,
оба готовит один и тот же конвейер MOE, и по ним же посчитан эталон.

**Водороды берутся оттуда же.** KLIFS отдаёт структуры уже протонированными
(у 6tgu в кармане 745 атомов водорода, у лиганда 9), и эталон посчитан по ним.
Достраивать свои нельзя: для вращающихся групп RDKit ставит водород по валентности,
ориентацию не оптимизируя, и бит водородной связи выходит по углу ложным.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import acos, degrees, dist
from pathlib import Path

import numpy as np
from rdkit import Chem

from kinase_ifp.config import (
    KLIFS_BITS_SHAPE,
    KLIFS_INTERACTION_TYPES,
    KLIFS_RULE_AROMATIC_DISTANCE,
    KLIFS_RULE_AROMATIC_FACE_ANGLE_DEG,
    KLIFS_RULE_DONOR_ELEMENTS,
    KLIFS_RULE_HBOND_DISTANCE,
    KLIFS_RULE_HBOND_MAX_DEVIATION_DEG,
    KLIFS_RULE_HYDROPHOBE_DISTANCE,
    KLIFS_RULE_HYDROPHOBE_ELEMENTS,
    KLIFS_RULE_IONIC_DISTANCE,
    KLIFS_RULE_METAL_ELEMENTS,
    KLIFS_RULE_NON_ACCEPTOR_NITROGEN_TYPES,
    KLIFS_RULE_SYBYL_FORMAL_CHARGE,
)
from kinase_ifp.mol2_reader import Mol2Atom, Mol2ReadError, Mol2Structure, read_mol2
from kinase_ifp.structure_io import split_residue_label

Точка = tuple[float, float, float]

_ВОДОРОД = "H"
_ПОЗИЦИЙ, _ТИПОВ = KLIFS_BITS_SHAPE
_АРОМАТИЧЕСКИХ_СОСЕДЕЙ = 2
# Файлы KLIFS, по которым считается отпечаток. Карман, а не белок целиком: в нём
# ровно те остатки, которым разметка сопоставляет позиции, и он вшестеро меньше.
_КАРМАН_MOL2 = "pocket.mol2"
_ЛИГАНД_MOL2 = "ligand.mol2"


class KlifsRulesError(RuntimeError):
    """Структуру нельзя разобрать по правилам KLIFS: нет файла, нет лиганда, нет разметки."""


@dataclass(frozen=True)
class AromaticCentre:
    """Ароматический центр свойства: сам атом и нормаль к плоскости кольца в нём.

    В оригинале центром служит **атом**, а не центр кольца: `AFP::LcloRng` по
    умолчанию `false`, и `SetDirections` собирает молекулку «атом плюс его
    ароматические соседи», первый атом которой (сам атом) становится центром.
    Поэтому расстояние между ароматическими центрами — это расстояние между
    ароматическими атомами, а буквальное прочтение таблицы 4 («центры колец
    ближе 4.0 Å») не воспроизводит на выборке ни одного ароматического бита:
    центры колец в кармане киназы расходятся на 4.5…6.3 Å.

    Поля `centroid` и `atoms` сохранены ради выгрузки геометрии (`ring_pairs`),
    где обе величины стоят рядом и сравнимы.
    """

    centroid: Точка
    normal: Точка
    atoms: tuple[Точка, ...] = ()


@dataclass(frozen=True)
class Donor:
    """Донор водородной связи: тяжёлый атом и водороды, которые на нём сидят."""

    heavy: Точка
    hydrogens: tuple[Точка, ...]


@dataclass(frozen=True)
class Groups:
    """Атомы одной стороны, разложенные по флагам таблицы 2.

    Одна и та же структура описывает и остаток белка, и лиганд целиком: правила
    таблицы 4 симметричны, и разводить две почти одинаковых записи незачем.
    """

    hydrophobes: tuple[Точка, ...] = ()
    rings: tuple[AromaticCentre, ...] = ()
    donors: tuple[Donor, ...] = ()
    acceptors: tuple[Точка, ...] = ()
    cations: tuple[Точка, ...] = ()
    anions: tuple[Точка, ...] = ()


@dataclass(frozen=True)
class RuleThresholds:
    """Числовые пороги правил таблицы 4.

    Вынесены в параметр, чтобы влияние каждого порога можно было измерить, а не
    обсуждать: два оставшихся расхождения на выборке лежат в пределах 0.1 Å
    от порогов, и вилку, которую допускают данные, показывает перебор
    (`scripts/klifs_rules.py --hbond ... --ionic ...`).
    """

    hydrophobe: float = KLIFS_RULE_HYDROPHOBE_DISTANCE
    hbond: float = KLIFS_RULE_HBOND_DISTANCE
    hbond_deviation: float = KLIFS_RULE_HBOND_MAX_DEVIATION_DEG
    ionic: float = KLIFS_RULE_IONIC_DISTANCE
    aromatic: float = KLIFS_RULE_AROMATIC_DISTANCE
    aromatic_angle: float = KLIFS_RULE_AROMATIC_FACE_ANGLE_DEG


# Набор по умолчанию — ровно числа статьи и исходника FingerPrintLib. Отдельного
# «измеренного» набора больше нет: после того как алгоритм стал совпадать с исходником,
# подгонять оказалось нечего.
ПРАВИЛА_KLIFS = RuleThresholds()

def _угол_векторов(первый: Точка, второй: Точка) -> float:
    """Угол между двумя векторами в градусах, 0…180."""
    a = np.asarray(первый, dtype=float)
    b = np.asarray(второй, dtype=float)
    длины = float(np.linalg.norm(a) * np.linalg.norm(b))
    if длины == 0.0:
        return 180.0
    косинус = float(np.dot(a, b)) / длины
    return degrees(acos(max(-1.0, min(1.0, косинус))))


def _вектор(начало: Точка, конец: Точка) -> Точка:
    """Вектор из одной точки в другую."""
    return (конец[0] - начало[0], конец[1] - начало[1], конец[2] - начало[2])


def _угол_плоскостей(первая: Точка, вторая: Точка) -> float:
    """Угол между плоскостями колец в градусах, 0…90.

    У нормали произвольный знак, поэтому угол между нормалями сам по себе не определён
    однозначно: 20° и 160° описывают одно и то же взаимное положение колец. Берётся
    острый угол — именно поэтому закрытые интервалы таблицы 4 (`[-π/6, π/6]` против
    `[π/6, 5π/6]`) сводятся к одной границе в 30°.
    """
    угол = _угол_векторов(первая, вторая)
    return min(угол, 180.0 - угол)


def _формальный_заряд(sybyl: str) -> int:
    """Формальный заряд атома по его типу SYBYL; ноль для типов, заряда не несущих."""
    return KLIFS_RULE_SYBYL_FORMAL_CHARGE.get(sybyl, 0)


def _акцептор(атом: Mol2Atom) -> bool:
    """Флаг `IsHA` оригинала (`AFP::SetIsHA`).

    Акцептор — это любой атом с отрицательным формальным зарядом (элемент при этом
    не важен, и это шире привычного «кислород или азот»), а из нейтральных —
    кислород и азот, **кроме** типов `N.pl3`, `N.am` и `N.4`. Положительно
    заряженный атом акцептором не считается никогда.
    """
    заряд = _формальный_заряд(атом.sybyl)
    if заряд < 0:
        return True
    if заряд > 0:
        return False
    if атом.element == "O":
        return True
    return атом.element == "N" and атом.sybyl not in KLIFS_RULE_NON_ACCEPTOR_NITROGEN_TYPES


def _нормаль_ароматики(центр: Точка, соседи: Sequence[Точка]) -> Точка | None:
    """Нормаль к плоскости кольца, построенная в самом атоме.

    Повторяет `NAroFind_Local`: векторное произведение связей к двум первым
    ароматическим соседям. Для плоского кольца это та же нормаль, что даёт кольцо
    целиком, но считается она локально, как в оригинале.

    `None`, если ароматических соседей меньше двух: в оригинале итератор в этом
    случае уходит за конец списка, то есть поведение не определено, и воспроизводить
    там нечего.
    """
    if len(соседи) < _АРОМАТИЧЕСКИХ_СОСЕДЕЙ:
        return None
    первый = np.asarray(_вектор(центр, соседи[0]), dtype=float)
    второй = np.asarray(_вектор(центр, соседи[1]), dtype=float)
    нормаль = np.cross(первый, второй)
    длина = float(np.linalg.norm(нормаль))
    if длина == 0.0:
        return None
    return tuple((нормаль / длина).tolist())


def _ароматичность(текст: str, атомов: int) -> tuple[bool, ...]:
    """Ароматичность каждого атома mol2 по перцепции RDKit.

    В оригинале флаг берётся у тулкита — `IsAr = X->IsAromatic()` (`AFP.hpp`), —
    то есть это результат модели ароматичности OEChem, а не запись из файла.
    Прочитать его из mol2 нельзя: MOE помечает типом `C.ar` только шестичленные
    кольца, а имидазол гистидина и пятичленное кольцо триптофана пишет кекулевскими
    связями. На кармане 6tgu это разница между 48 помеченными атомами и 66
    действительно ароматическими — три гистидина и триптофан целиком.

    OEChem у нас нет (лицензия), поэтому его модель заменяется моделью RDKit.
    На кольцах, которые встречаются в белке и в лигандах киназ, обе дают одно и то
    же: бензол, пиридин, пиррол, имидазол, индол, фуран, тиофен. Это единственное
    место правил, где мы заведомо подставляем другой инструмент, и оно объявлено.

    Порядок атомов RDKit сохраняет; расхождение числа атомов — ошибка, а не повод
    посчитать иначе.
    """
    молекула = Chem.MolFromMol2Block(текст, removeHs=False, sanitize=True)
    if молекула is None:
        raise KlifsRulesError(
            "RDKit не разобрал mol2 для перцепции ароматичности: без неё биты F-F "
            "и F-E не поставить, а тип SYBYL ароматичность не задаёт"
        )
    if молекула.GetNumAtoms() != атомов:
        raise KlifsRulesError(
            f"RDKit прочитал {молекула.GetNumAtoms()} атомов, в mol2 их {атомов}: "
            "сопоставить ароматичность по порядку нельзя"
        )
    return tuple(атом.GetIsAromatic() for атом in молекула.GetAtoms())


def _разобрать_mol2(
    path: Path,
) -> tuple[Mol2Structure, tuple[tuple[int, ...], ...], tuple[bool, ...]]:
    """Структура mol2, списки смежности и ароматичность атомов."""
    try:
        структура = read_mol2(path)
    except Mol2ReadError as ошибка:
        raise KlifsRulesError(str(ошибка)) from ошибка
    текст = path.read_text(encoding="utf-8")
    return структура, структура.neighbours(), _ароматичность(текст, len(структура.atoms))


def groups_from_mol2(
    структура: Mol2Structure,
    соседи: Sequence[Sequence[int]],
    ароматичность: Sequence[bool],
    индексы: Sequence[int],
) -> Groups:
    """Раскладывает подмножество атомов mol2 по шести флагам оригинала.

    Одна функция обслуживает и остаток белка, и лиганд целиком: флаги в оригинале
    ставятся поатомно и от того, чей это атом, не зависят. `индексы` задают, какие
    атомы структуры берутся; `соседи` — списки смежности из `Mol2Structure`.
    """
    атомы = структура.atoms
    гидрофобы: list[Точка] = []
    акцепторы: list[Точка] = []
    катионы: list[Точка] = []
    анионы: list[Точка] = []
    ароматика: list[AromaticCentre] = []
    # Донор в оригинале — сам водород, но расстояние меряется от тяжёлого соседа,
    # поэтому водороды собираются по тяжёлому атому, на котором сидят.
    водороды_донора: dict[int, list[Точка]] = {}

    for и in индексы:
        атом = атомы[и]
        if атом.element == _ВОДОРОД:
            for сосед in соседи[и]:
                if атомы[сосед].element in KLIFS_RULE_DONOR_ELEMENTS:
                    водороды_донора.setdefault(сосед, []).append(атом.xyz)
                    break
            continue
        if атом.element in KLIFS_RULE_HYDROPHOBE_ELEMENTS:
            гидрофобы.append(атом.xyz)
        if _акцептор(атом):
            акцепторы.append(атом.xyz)
        заряд = _формальный_заряд(атом.sybyl)
        if заряд > 0 and атом.element not in KLIFS_RULE_METAL_ELEMENTS:
            катионы.append(атом.xyz)
        if заряд < 0:
            анионы.append(атом.xyz)
        if ароматичность[и]:
            нормаль = _нормаль_ароматики(
                атом.xyz, [атомы[с].xyz for с in соседи[и] if ароматичность[с]]
            )
            if нормаль is not None:
                ароматика.append(
                    AromaticCentre(centroid=атом.xyz, normal=нормаль, atoms=(атом.xyz,))
                )

    return Groups(
        hydrophobes=tuple(гидрофобы),
        rings=tuple(ароматика),
        donors=tuple(
            Donor(heavy=атомы[и].xyz, hydrogens=tuple(в))
            for и, в in sorted(водороды_донора.items())
        ),
        acceptors=tuple(акцепторы),
        # Ионное расстояние меряется между самими заряженными атомами: в исходнике
        # `IonFind` кладёт в `IoGC` координаты атома, а не центр группы.
        cations=tuple(катионы),
        anions=tuple(анионы),
    )


def pocket_groups(path: Path, chain: str) -> dict[str, Groups]:
    """Разбирает `pocket.mol2`; ключ — остаток в нотации `residue_to_position`.

    Цепь в mol2 не хранится вовсе, поэтому подставляется аргументом — так же, как
    это делает `structure_io.write_pdb_from_mol2`.
    """
    структура, соседи, ароматичность = _разобрать_mol2(path)
    по_остаткам: dict[str, list[int]] = {}
    for и, атом in enumerate(структура.atoms):
        имя, номер, вставка = split_residue_label(атом.residue)
        по_остаткам.setdefault(f"{имя}{номер}{вставка}.{chain}", []).append(и)
    return {
        ключ: groups_from_mol2(структура, соседи, ароматичность, индексы)
        for ключ, индексы in по_остаткам.items()
    }


def ligand_groups(path: Path) -> Groups:
    """Разбирает `ligand.mol2` KLIFS целиком, сохраняя водороды MOE."""
    структура, соседи, ароматичность = _разобрать_mol2(path)
    return groups_from_mol2(
        структура, соседи, ароматичность, range(len(структура.atoms))
    )


def _есть_пара(левые: Iterable[Точка], правые: Sequence[Точка], порог: float) -> bool:
    """Есть ли хоть одна пара точек ближе порога.

    Неравенство строгое: в исходнике FingerPrintLib все сравнения расстояний —
    `OEGeom3DDistance(...) < порог`.
    """
    return any(dist(л, п) < порог for л in левые for п in правые)


def _есть_водородная_связь(
    доноры: Sequence[Donor], акцепторы: Sequence[Точка], пороги: RuleThresholds
) -> bool:
    """Правило таблицы 4: ‖DA‖ < 3.5 Å и отклонение от линейности меньше 45°.

    **Отклонение считается между связью D–H и направлением D→A**, а не между
    векторами D→H и H→A. Разница не косметическая: вторая величина примерно вдвое
    чувствительнее, и по ней те же комплексы требовали бы допуска 60° вместо
    напечатанных 45°. Так считает исходник FingerPrintLib (`AFP.cc`, `AFP::Property`:
    `U = HAGC − HDGC`, затем `scalar = HD · U` и `|acos(scalar) − Hangl| < Hdngl`
    при `Hangl = 0`, `Hdngl = π/4`).
    """
    for донор in доноры:
        for акцептор in акцепторы:
            if dist(донор.heavy, акцептор) >= пороги.hbond:
                continue
            da = _вектор(донор.heavy, акцептор)
            for водород in донор.hydrogens:
                dh = _вектор(донор.heavy, водород)
                if _угол_векторов(dh, da) < пороги.hbond_deviation:
                    return True
    return False


def _ароматические_биты(
    остаток: Groups, лиганд: Groups, пороги: RuleThresholds
) -> tuple[bool, bool]:
    """Биты `F-F` и `F-E` по правилам таблицы 4.

    Оба бита делят одно условие по расстоянию, а тип решает острый угол между
    плоскостями: не больше 30° — «плоскость к плоскости», не меньше 30° — «ребро
    к плоскости». Граница входит в оба интервала, потому что в таблице 4 они
    закрытые; случай ровно 30.000° при этом не встречается.

    **Расстояние меряется между ближайшими атомами колец, а не между их центрами.**
    В таблице 4 напечатано `‖ac₁ac₂‖ ≤ 4.0 Å`, где `ac` в сноске — геометрический
    центр кольца, но буквальное прочтение не воспроизводит на выборке ни одного
    из 14 ароматических бит эталона: расстояния между центрами в кармане киназы
    4.5…6.3 Å. По ближайшим атомам колец **то же число 4.0 Å и та же граница 30°**
    дают 14 бит из 14 при нуле ложных, причём разделение чистое — у сработавших
    позиций 3.46…4.07 Å, у ближайшей несработавшей 4.44 Å. Измерено 16.09.2026,
    `docs/calibration.md`, 7; выгрузка геометрии — `scripts/klifs_rules.py
    --dump-pairs`, где обе величины стоят рядом и вывод можно перепроверить.
    """
    лицом = ребром = False
    for первое in остаток.rings:
        for второе in лиганд.rings:
            расстояние = min(
                (dist(а, б) for а in первое.atoms for б in второе.atoms),
                default=float("inf"),
            )
            if расстояние >= пороги.aromatic:
                continue
            угол = _угол_плоскостей(первое.normal, второе.normal)
            # В исходнике: |acos(|n1·n2|) − PSangl| < PSdngl при PSangl = 0, PSdngl = π/6
            # для «лицом к лицу» и EFangl = π/2, EFdngl = 2π/6 для «ребром». На остром
            # угле это ровно граница в 30°, строгая с обеих сторон.
            if угол < пороги.aromatic_angle:
                лицом = True
            if угол > пороги.aromatic_angle:
                ребром = True
    return лицом, ребром


def bits_for_residue(
    residue: Groups, ligand: Groups, thresholds: RuleThresholds = ПРАВИЛА_KLIFS
) -> tuple[bool, ...]:
    """Семь бит одной позиции кармана в порядке KLIFS (`KLIFS_INTERACTION_TYPES`)."""
    лицом, ребром = _ароматические_биты(residue, ligand, thresholds)
    return (
        _есть_пара(residue.hydrophobes, ligand.hydrophobes, thresholds.hydrophobe),
        лицом,
        ребром,
        _есть_водородная_связь(residue.donors, ligand.acceptors, thresholds),
        _есть_водородная_связь(ligand.donors, residue.acceptors, thresholds),
        _есть_пара(residue.cations, ligand.anions, thresholds.ionic),
        _есть_пара(residue.anions, ligand.cations, thresholds.ionic),
    )


@dataclass(frozen=True)
class PreparedPackage:
    """Разобранный пакет мишени: обе стороны уже разложены по флагам таблицы 2.

    Существует затем, чтобы перебор порогов не перечитывал структуру: разбор белка
    и лиганда занимает почти всё время расчёта, а от порогов не зависит вовсе.
    """

    label: str
    protein: dict[str, Groups]
    ligand: Groups
    position_to_residue: dict[int, str]
    reference: str


def prepare_package(package_json: Path, reference: str, label: str) -> PreparedPackage:
    """Читает пакет мишени и раскладывает обе стороны по флагам оригинала.

    Оба входа — файлы mol2 самого KLIFS: `pocket.mol2` и `ligand.mol2`. Запасного
    пути через PDB и SDF больше нет намеренно. Он означал бы другую химию (без типов
    SYBYL нет ни правила акцептора, ни формальных зарядов) и другие водороды
    (достроенные нами вместо MOE), то есть тихую подмену правил на похожие.
    Отсутствие файла — ошибка, а не повод посчитать иначе.
    """
    пакет = json.loads(package_json.read_text(encoding="utf-8"))
    корень = package_json.parent
    разметка = пакет.get("residue_to_position") or {}
    if not разметка:
        raise KlifsRulesError(f"в пакете нет residue_to_position: {package_json}")
    # Цепь берётся из самой разметки: mol2 её не хранит, а все ключи разметки
    # относятся к одной цепи по построению пакета.
    цепь = next(iter(разметка)).rsplit(".", 1)[-1]
    return PreparedPackage(
        label=label,
        protein=pocket_groups(корень / _КАРМАН_MOL2, цепь),
        ligand=ligand_groups(корень / _ЛИГАНД_MOL2),
        position_to_residue={позиция: остаток for остаток, позиция in разметка.items()},
        reference=reference,
    )


def fingerprint_from_prepared(
    prepared: PreparedPackage, thresholds: RuleThresholds = ПРАВИЛА_KLIFS
) -> str:
    """595-битная строка по правилам KLIFS для уже разобранного пакета."""
    символы: list[str] = []
    for поз in range(1, _ПОЗИЦИЙ + 1):
        остаток = prepared.position_to_residue.get(поз)
        группы = prepared.protein.get(остаток) if остаток else None
        биты = (
            bits_for_residue(группы, prepared.ligand, thresholds)
            if группы
            else (False,) * _ТИПОВ
        )
        символы.extend("1" if бит else "0" for бит in биты)
    return "".join(символы)


def fingerprint_by_klifs_rules(
    package_json: Path, *, thresholds: RuleThresholds = ПРАВИЛА_KLIFS
) -> str:
    """595-битная строка, посчитанная по правилам KLIFS на пакете мишени.

    Формат совпадает с колонкой `interaction.fingerprint` базы: раскладка
    `(85 позиций, 7 типов)`. Позиции, которым разметка кармана не
    сопоставила остаток, дают семь нулей — как и в эталоне.
    """
    готовый = prepare_package(package_json, reference="", label=package_json.parent.name)
    return fingerprint_from_prepared(готовый, thresholds)


@dataclass(frozen=True)
class BitDifference:
    """Один расходящийся бит: позиция кармана, тип взаимодействия и чьё это значение."""

    position: int
    interaction: str
    ours: bool
    reference: bool


@dataclass(frozen=True)
class TypeCount:
    """Счёт по одному типу взаимодействия: наших бит, эталонных и общих."""

    interaction: str
    ours: int
    reference: int
    shared: int


@dataclass(frozen=True)
class FingerprintComparison:
    """Итог сверки расчёта по правилам KLIFS с эталоном одной структуры."""

    label: str
    differences: tuple[BitDifference, ...]
    by_type: tuple[TypeCount, ...]

    @property
    def identical(self) -> bool:
        """Совпал ли отпечаток с эталоном бит в бит."""
        return not self.differences

    @property
    def reference_total(self) -> int:
        """Сколько бит стоит у эталона."""
        return sum(счёт.reference for счёт in self.by_type)


def compare_with_reference(label: str, ours: str, reference: str) -> FingerprintComparison:
    """Сравнивает две 595-битные строки бит в бит и по типам взаимодействий."""
    ожидается = _ПОЗИЦИЙ * _ТИПОВ
    if len(ours) != ожидается or len(reference) != ожидается:
        raise KlifsRulesError(
            f"длины строк {len(ours)} и {len(reference)}, ожидается {ожидается}"
        )
    расхождения: list[BitDifference] = []
    наши = dict.fromkeys(KLIFS_INTERACTION_TYPES, 0)
    эталонные = dict.fromkeys(KLIFS_INTERACTION_TYPES, 0)
    общие = dict.fromkeys(KLIFS_INTERACTION_TYPES, 0)
    for индекс, (наш, эталон) in enumerate(zip(ours, reference, strict=True)):
        тип = KLIFS_INTERACTION_TYPES[индекс % _ТИПОВ]
        наш_бит = наш == "1"
        эталонный = эталон == "1"
        наши[тип] += наш_бит
        эталонные[тип] += эталонный
        общие[тип] += наш_бит and эталонный
        if наш_бит != эталонный:
            расхождения.append(
                BitDifference(
                    position=индекс // _ТИПОВ + 1,
                    interaction=тип,
                    ours=наш_бит,
                    reference=эталонный,
                )
            )
    return FingerprintComparison(
        label=label,
        differences=tuple(расхождения),
        by_type=tuple(
            TypeCount(тип, наши[тип], эталонные[тип], общие[тип])
            for тип in KLIFS_INTERACTION_TYPES
        ),
    )


def compare_prepared(
    prepared: PreparedPackage, thresholds: RuleThresholds = ПРАВИЛА_KLIFS
) -> FingerprintComparison:
    """Сверяет расчёт по правилам KLIFS с эталоном для уже разобранного пакета."""
    return compare_with_reference(
        prepared.label, fingerprint_from_prepared(prepared, thresholds), prepared.reference
    )


def compare_package(
    package_json: Path,
    reference: str,
    label: str,
    *,
    thresholds: RuleThresholds = ПРАВИЛА_KLIFS,
) -> FingerprintComparison:
    """Считает отпечаток по правилам KLIFS и сразу сверяет его с эталоном."""
    готовый = prepare_package(package_json, reference, label)
    return compare_prepared(готовый, thresholds)


def aggregate_types(
    comparisons: Sequence[FingerprintComparison],
) -> tuple[TypeCount, ...]:
    """Складывает счёт по типам со всех структур выборки."""
    return tuple(
        TypeCount(
            interaction=тип,
            ours=sum(с.by_type[номер].ours for с in comparisons),
            reference=sum(с.by_type[номер].reference for с in comparisons),
            shared=sum(с.by_type[номер].shared for с in comparisons),
        )
        for номер, тип in enumerate(KLIFS_INTERACTION_TYPES)
    )


@dataclass(frozen=True)
class RingPair:
    """Геометрия одной пары колец «остаток кармана — лиганд» рядом с битами эталона.

    Нужна для того же, для чего понадобился перебор правила `apolar`: понять, какой
    величиной KLIFS отделяет сработавший ароматический бит от несработавшего.
    Опубликованное правило (центры ближе 4.0 Å) на выборке не воспроизводит ни одного
    бита, а значит, KLIFS считает не тем, что напечатано в таблице 4.
    """

    label: str
    position: int
    residue: str
    centroid_distance: float
    plane_angle: float
    normal_to_centroid_angle: float
    min_atom_distance: float
    reference_face: bool
    reference_edge: bool


def ring_pairs(prepared: PreparedPackage) -> list[RingPair]:
    """Все пары «кольцо остатка — кольцо лиганда» с геометрией и битами эталона."""
    итог: list[RingPair] = []
    for поз in range(1, _ПОЗИЦИЙ + 1):
        остаток = prepared.position_to_residue.get(поз)
        группы = prepared.protein.get(остаток) if остаток else None
        if not группы or not группы.rings:
            continue
        начало = (поз - 1) * _ТИПОВ
        лицом = prepared.reference[начало + 1] == "1"
        ребром = prepared.reference[начало + 2] == "1"
        for кольцо_остатка in группы.rings:
            for кольцо_лиганда in prepared.ligand.rings:
                связь = _вектор(кольцо_остатка.centroid, кольцо_лиганда.centroid)
                итог.append(
                    RingPair(
                        label=prepared.label,
                        position=поз,
                        residue=остаток or "",
                        centroid_distance=dist(
                            кольцо_остатка.centroid, кольцо_лиганда.centroid
                        ),
                        plane_angle=_угол_плоскостей(
                            кольцо_остатка.normal, кольцо_лиганда.normal
                        ),
                        normal_to_centroid_angle=_угол_плоскостей(
                            кольцо_остатка.normal, связь
                        ),
                        min_atom_distance=min(
                            (
                                dist(а, б)
                                for а in кольцо_остатка.atoms
                                for б in кольцо_лиганда.atoms
                            ),
                            default=float("inf"),
                        ),
                        reference_face=лицом,
                        reference_edge=ребром,
                    )
                )
    return итог


@dataclass(frozen=True)
class HBondCandidate:
    """Лучшая по расстоянию пара «донор — акцептор» позиции рядом с битом эталона.

    `direction` — `DON`, когда донором выступает белок, и `ACC`, когда лиганд:
    имена типов KLIFS описывают роль белка, а не лиганда.
    """

    label: str
    position: int
    residue: str
    direction: str
    distance: float
    deviation: float
    reference: bool


def _лучшая_пара(
    доноры: Sequence[Donor], акцепторы: Sequence[Точка]
) -> tuple[float, float]:
    """Минимальное ‖DA‖ и отклонение от линейности у этой же пары."""
    лучшее = (float("inf"), 180.0)
    for донор in доноры:
        for акцептор in акцепторы:
            расстояние = dist(донор.heavy, акцептор)
            if расстояние >= лучшее[0]:
                continue
            # Та же величина, что в `_есть_водородная_связь`: угол между связью D-H
            # и направлением D->A. Считать здесь иначе значило бы выгружать не то,
            # по чему ставится бит.
            da = _вектор(донор.heavy, акцептор)
            отклонение = min(
                (_угол_векторов(_вектор(донор.heavy, в), da) for в in донор.hydrogens),
                default=180.0,
            )
            лучшее = (расстояние, отклонение)
    return лучшее


def hbond_candidates(prepared: PreparedPackage) -> list[HBondCandidate]:
    """Лучшие кандидаты в водородную связь по позициям кармана, с битами эталона."""
    итог: list[HBondCandidate] = []
    for поз in range(1, _ПОЗИЦИЙ + 1):
        остаток = prepared.position_to_residue.get(поз)
        группы = prepared.protein.get(остаток) if остаток else None
        if not группы:
            continue
        начало = (поз - 1) * _ТИПОВ
        for сдвиг, направление, доноры, акцепторы in (
            (3, "DON", группы.donors, prepared.ligand.acceptors),
            (4, "ACC", prepared.ligand.donors, группы.acceptors),
        ):
            расстояние, отклонение = _лучшая_пара(доноры, акцепторы)
            if расстояние == float("inf"):
                continue
            итог.append(
                HBondCandidate(
                    label=prepared.label,
                    position=поз,
                    residue=остаток or "",
                    direction=направление,
                    distance=расстояние,
                    deviation=отклонение,
                    reference=prepared.reference[начало + сдвиг] == "1",
                )
            )
    return итог
