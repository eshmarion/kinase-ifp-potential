"""Реконструкция правила `apolar` KLIFS по эталонным отпечаткам.

Первый бит каждой позиции эталонного отпечатка KLIFS (`apolar`) назначается правилом,
которое база нигде не опубликовала (`config.APOLAR_ATOM_ELEMENTS`, там же разбор
источников). Модуль восстанавливает правило перебором: для набора структур считает,
сколько эталонных бит воспроизводит каждое правило-кандидат и сколько даёт лишних.

Почему геометрия считается здесь заново, а не берётся у ProLIF: проверяемые правила
описываются набором химических элементов и порогом, тогда как ProLIF отбирает атомы
SMARTS-шаблоном и подменить своё правило не даёт. Сравнение с самим ProLIF получается
из обычного расчёта отпечатка (`fingerprint.compute_ifp`) и здесь не дублируется.

Водороды не учитываются ни одним правилом: их добавление даёт +2 бита и 69 ложных
срабатываний (`docs/calibration.md`, 3.2), то есть KLIFS считает по тяжёлым атомам.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import dist
from pathlib import Path

from kinase_ifp.config import (
    APOLAR_CONTACT_DISTANCE,
    KLIFS_BITS_SHAPE,
    KLIFS_INTERACTION_TYPES,
)
from kinase_ifp.pdb_reader import PdbReadError, read_pdb_residues

_ВОДОРОД = "H"
_ИНДЕКС_APOLAR = KLIFS_INTERACTION_TYPES.index("HYD")
_ПОЗИЦИЙ = KLIFS_BITS_SHAPE[0]
_ТИПОВ = KLIFS_BITS_SHAPE[1]


class ApolarRuleError(RuntimeError):
    """Структуру нельзя разобрать: нет файла, пустой лиганд, битая разметка позиций."""


@dataclass(frozen=True)
class Atom:
    """Тяжёлый атом: химический элемент и координаты в системе пакета мишени."""

    element: str
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class RuleScore:
    """Итог правила на наборе структур.

    `reproduced` — эталонные биты, которые правило ставит; `extra` — биты, которые
    правило ставит там, где у эталона ноль; `reference_total` — сколько эталонных
    бит было всего.
    """

    rule: str
    reproduced: int
    extra: int
    reference_total: int

    @property
    def recall(self) -> float:
        """Доля эталонных бит, которые правило воспроизвело."""
        return self.reproduced / self.reference_total if self.reference_total else 0.0

    @property
    def precision(self) -> float:
        """Доля поставленных бит, которые есть у эталона."""
        поставлено = self.reproduced + self.extra
        return self.reproduced / поставлено if поставлено else 0.0


@dataclass(frozen=True)
class StructureRuleScore:
    """Итог правила на одной структуре.

    `positions` — сколько позиций кармана рассматривалось; `reference_total` из них
    несут бит у эталона, остальные — отрицательные случаи, на которых проверяется
    точность.
    """

    label: str
    rule: str
    reproduced: int
    extra: int
    reference_total: int
    positions: int

    @property
    def recall(self) -> float:
        """Доля эталонных бит структуры, которые правило воспроизвело."""
        return self.reproduced / self.reference_total if self.reference_total else 0.0

    @property
    def negatives(self) -> int:
        """Позиции без бита у эталона — на них правило обязано молчать."""
        return self.positions - self.reference_total


def parse_pdb_atoms(path: Path) -> dict[str, list[Atom]]:
    """Тяжёлые атомы белка по остаткам; ключ — имя остатка, как в пакете мишени.

    Ключ собирается в том же виде, что и ключ `residue_to_position` из `target.json`
, поэтому сопоставление с позициями кармана не требует отдельной
    таблицы.

    Водороды отбрасываются здесь, а не в разборщике: их добавление даёт +2 бита
    и 69 ложных срабатываний, то есть KLIFS считает `apolar` по тяжёлым атомам.
    """
    try:
        по_остаткам = read_pdb_residues(path)
    except PdbReadError as ошибка:
        raise ApolarRuleError(str(ошибка)) from ошибка
    тяжёлые = {
        ключ: [Atom(а.element, а.xyz) for а in атомы if а.element != _ВОДОРОД]
        for ключ, атомы in по_остаткам.items()
    }
    return {ключ: атомы for ключ, атомы in тяжёлые.items() if атомы}


def parse_sdf_atoms(path: Path) -> list[Atom]:
    """Тяжёлые атомы лиганда из первой записи SDF.

    Блок V2000 читается напрямую, без RDKit: нужны только элемент и координаты,
    а санитизация здесь ничего не добавляет и может отвергнуть запись, которую
    KLIFS при расчёте эталона принял.
    """
    if not path.is_file():
        raise ApolarRuleError(f"нет файла лиганда: {path}")
    строки = path.read_text(encoding="utf-8").splitlines()
    if len(строки) < 4:
        raise ApolarRuleError(f"файл лиганда короче заголовка SDF: {path}")
    счётчик = строки[3]
    try:
        атомов = int(счётчик[:3])
    except ValueError as ошибка:
        raise ApolarRuleError(f"не читается строка счётчиков SDF: {счётчик!r}") from ошибка
    итог: list[Atom] = []
    for строка in строки[4 : 4 + атомов]:
        элемент = строка[31:34].strip().upper()
        if элемент == _ВОДОРОД:
            continue
        xyz = (float(строка[0:10]), float(строка[10:20]), float(строка[20:30]))
        итог.append(Atom(элемент, xyz))
    if not итог:
        raise ApolarRuleError(f"в лиганде нет тяжёлых атомов: {path}")
    return итог


def select_atoms(atoms: Iterable[Atom], elements: Sequence[str]) -> list[Atom]:
    """Отбор атомов по элементам; пустой набор означает «любой тяжёлый атом»."""
    if not elements:
        return list(atoms)
    разрешено = {э.upper() for э in elements}
    return [а for а in atoms if а.element in разрешено]


def has_contact(left: Sequence[Atom], right: Sequence[Atom], distance: float) -> bool:
    """Есть ли хоть одна пара атомов ближе порога."""
    for а in left:
        for б in right:
            if dist(а.xyz, б.xyz) <= distance:
                return True
    return False


def reference_apolar_bits(bits: str) -> dict[int, bool]:
    """Биты `apolar` эталона по позициям кармана; позиции нумеруются с единицы.

    Раскладка: позиция = индекс // 7, тип = индекс % 7.
    """
    ожидается = _ПОЗИЦИЙ * _ТИПОВ
    if len(bits) != ожидается:
        raise ApolarRuleError(f"эталон длиной {len(bits)}, ожидается {ожидается}")
    return {
        поз: bits[(поз - 1) * _ТИПОВ + _ИНДЕКС_APOLAR] == "1"
        for поз in range(1, _ПОЗИЦИЙ + 1)
    }


def apolar_bits_by_rule(
    package_json: Path,
    elements: Sequence[str],
    distance: float = APOLAR_CONTACT_DISTANCE,
) -> dict[int, bool]:
    """Биты `apolar`, которые даёт правило на пакете мишени."""
    пакет = json.loads(package_json.read_text(encoding="utf-8"))
    корень = package_json.parent
    белок = parse_pdb_atoms(корень / пакет["protein_pdb"])
    лиганд = select_atoms(parse_sdf_atoms(корень / пакет["ligand_sdf"]), elements)
    разметка = пакет.get("residue_to_position") or {}
    if not разметка:
        raise ApolarRuleError(f"в пакете нет residue_to_position: {package_json}")
    позиция_остатка = {позиция: остаток for остаток, позиция in разметка.items()}
    биты: dict[int, bool] = {}
    for поз in range(1, _ПОЗИЦИЙ + 1):
        остаток = позиция_остатка.get(поз)
        атомы = select_atoms(белок.get(остаток, []), elements) if остаток else []
        биты[поз] = bool(атомы) and has_contact(атомы, лиганд, distance)
    return биты


def score_rules_per_structure(
    packages: Mapping[Path, str],
    rules: Mapping[str, Sequence[str]],
    distance: float = APOLAR_CONTACT_DISTANCE,
) -> list[StructureRuleScore]:
    """Считает правила по каждой структуре отдельно.

    Разбивка нужна не для красоты отчёта: сводное число скрывает, что решающих
    наблюдений у части набора элементов единицы — хлор различает три бита, сера один,
    и видно это только по структурам.

    Метка структуры берётся из имени каталога пакета: `build_target_package` кладёт
    `target.json` в подкаталог с `pdb_id`.
    """
    итоги: list[StructureRuleScore] = []
    for путь, биты in packages.items():
        эталон = reference_apolar_bits(биты)
        эталонных = sum(эталон.values())
        позиций = len(эталон)
        for имя, элементы in rules.items():
            наши = apolar_bits_by_rule(путь, элементы, distance)
            воспроизведено = sum(1 for поз, есть in эталон.items() if есть and наши[поз])
            лишних = sum(1 for поз, есть in эталон.items() if наши[поз] and not есть)
            итоги.append(
                StructureRuleScore(
                    label=путь.parent.name,
                    rule=имя,
                    reproduced=воспроизведено,
                    extra=лишних,
                    reference_total=эталонных,
                    positions=позиций,
                )
            )
    return итоги


def aggregate(
    per_structure: Sequence[StructureRuleScore], rules: Iterable[str]
) -> list[RuleScore]:
    """Складывает разбивку по структурам в сводку по правилам.

    Вынесено отдельно, чтобы сводное число нельзя было посчитать вторым проходом
    и разойтись с разбивкой: и `score_rules`, и CLI зовут именно эту функцию.
    """
    итоги: list[RuleScore] = []
    for имя in rules:
        свои = [и for и in per_structure if и.rule == имя]
        итоги.append(
            RuleScore(
                rule=имя,
                reproduced=sum(и.reproduced for и in свои),
                extra=sum(и.extra for и in свои),
                reference_total=sum(и.reference_total for и in свои),
            )
        )
    return итоги


def score_rules(
    packages: Mapping[Path, str],
    rules: Mapping[str, Sequence[str]],
    distance: float = APOLAR_CONTACT_DISTANCE,
) -> list[RuleScore]:
    """Сводка по правилам на наборе пакетов: по одному итогу на правило."""
    return aggregate(score_rules_per_structure(packages, rules, distance), rules)


def compare_alternate_models(
    fingerprints: Mapping[int, str], structures: Sequence[tuple[int, str, str, str]]
) -> dict[str, int]:
    """Сравнивает эталоны альтернативных моделей одной пары «структура, цепь».

    `structures` — записи вида `(klifs_id, pdb_id, chain, alternate_model)`. Отвечает
    на вопрос, считает KLIFS отпечаток на модель или на структуру целиком: если
    у всех пар отпечатки моделей совпадают побитово, отпечаток один на структуру,
    и выбор модели влияет на наш расчёт, но не на эталон.
    """
    по_парам: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for klifs_id, pdb_id, цепь, модель in structures:
        биты = fingerprints.get(klifs_id)
        if биты is None:
            continue
        по_парам.setdefault((pdb_id, цепь), []).append((модель, биты))
    пар = совпало = разошлось = различий = 0
    for записи in по_парам.values():
        if len({модель for модель, _ in записи}) < 2:
            continue
        пар += 1
        if len({биты for _, биты in записи}) == 1:
            совпало += 1
            continue
        разошлось += 1
        первый, *прочие = [биты for _, биты in записи]
        for биты in прочие:
            различий += sum(1 for a, b in zip(первый, биты, strict=True) if a != b)
    return {
        "пар с несколькими моделями": пар,
        "отпечатки совпали": совпало,
        "отпечатки разошлись": разошлось,
        "различающихся бит всего": различий,
    }
