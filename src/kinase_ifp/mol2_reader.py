"""Чтение mol2 KLIFS с типами атомов SYBYL, зарядами и связями.

Нужен правилам `klifs_rules`: FingerPrintLib определяет флаги атома по типу SYBYL
и формальному заряду (`AFP.cc`, `AFP::SetIsHA` сравнивает `GetType()` со строками
`N.pl3`, `N.am`, `N.4`), а эти сведения есть только в mol2. В PDB нет ни типов,
ни зарядов, ни порядков связей, поэтому разбор белка из PDB требовал шаблонов
остатков — то есть нашей догадки вместо данных KLIFS.

**Почему свой разборщик, а не MDAnalysis или RDKit.** Обоим нужен разбор химии:
MDAnalysis теряет типы SYBYL и сведения о цепи (`structure_io`), RDKit санитизирует
молекулу и на белке из KLIFS отвергает то, что KLIFS при расчёте эталона принял.
Здесь же читаются ровно три колонки блока `@<TRIPOS>ATOM` и пары блока
`@<TRIPOS>BOND` — формат фиксированный, разбирать нечего.

Водороды сохраняются: без них не поставить ни один бит водородной связи.

**Ароматичность здесь не определяется.** Тип SYBYL её не задаёт: MOE помечает `.ar`
только шестичленные кольца, а имидазол гистидина и пятичленное кольцо триптофана
пишет в кекулевском виде (`C.2`, `N.2`, `N.pl3` и чередующиеся связи). В оригинале
флаг берётся у перцепции тулкита (`X->IsAromatic()`), поэтому и у нас он приходит
со стороны — см. `klifs_rules._ароматичность`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_БЛОК_АТОМОВ = "@<TRIPOS>ATOM"
_БЛОК_СВЯЗЕЙ = "@<TRIPOS>BOND"
_НАЧАЛО_БЛОКА = "@<TRIPOS>"
_ПОЛЕЙ_В_АТОМЕ = 6
_ПОЛЕЙ_В_СВЯЗИ = 4


class Mol2ReadError(RuntimeError):
    """mol2 нельзя разобрать: нет файла, нет блока атомов, битые поля."""


@dataclass(frozen=True)
class Mol2Atom:
    """Атом mol2: имя, тип SYBYL, координаты, заряд из файла и имя подструктуры.

    `element` выводится из типа SYBYL отсечением уточнения после точки: `N.am` → `N`,
    `C.ar` → `C`, `Cl` → `CL`. Имя атома для этого не годится — в белке оно позиционное
    (`CB`, `OD1`), и первая буква не всегда элемент.
    """

    name: str
    sybyl: str
    element: str
    xyz: tuple[float, float, float]
    charge: float
    residue: str


@dataclass(frozen=True)
class Mol2Structure:
    """Разобранный mol2: атомы в порядке файла и связи как пары индексов.

    Индексы связей уже переведены в нумерацию с нуля: в файле они с единицы.
    """

    atoms: tuple[Mol2Atom, ...]
    bonds: tuple[tuple[int, int], ...]

    def neighbours(self) -> tuple[tuple[int, ...], ...]:
        """Соседи каждого атома по связям, в порядке появления связей в файле.

        Порядок важен: ароматическую нормаль FingerPrintLib строит на двух **первых**
        ароматических соседях атома, а не на всех сразу.
        """
        соседи: list[list[int]] = [[] for _ in self.atoms]
        for левый, правый in self.bonds:
            соседи[левый].append(правый)
            соседи[правый].append(левый)
        return tuple(tuple(с) for с in соседи)


def _элемент(sybyl: str) -> str:
    """Химический элемент из типа SYBYL: часть до точки, в верхнем регистре."""
    return sybyl.split(".", 1)[0].upper()


def _границы(строки: Sequence[str], заголовок: str) -> tuple[int, int] | None:
    """Полуинтервал строк блока `@<TRIPOS>…`; `None`, если блока нет."""
    try:
        начало = next(i for i, с in enumerate(строки) if с.strip() == заголовок)
    except StopIteration:
        return None
    конец = next(
        (i for i in range(начало + 1, len(строки)) if строки[i].startswith(_НАЧАЛО_БЛОКА)),
        len(строки),
    )
    return начало + 1, конец


def read_mol2(path: Path) -> Mol2Structure:
    """Читает mol2: атомы с типами SYBYL и зарядами, связи как пары индексов."""
    if not path.is_file():
        raise Mol2ReadError(f"нет файла mol2: {path}")
    строки = path.read_text(encoding="utf-8").splitlines()

    границы = _границы(строки, _БЛОК_АТОМОВ)
    if границы is None:
        raise Mol2ReadError(f"в mol2 нет блока {_БЛОК_АТОМОВ}: {path}")
    атомы: list[Mol2Atom] = []
    for строка in строки[границы[0] : границы[1]]:
        поля = строка.split()
        if not поля:
            continue
        if len(поля) < _ПОЛЕЙ_В_АТОМЕ:
            raise Mol2ReadError(f"{path}: строка атома короче {_ПОЛЕЙ_В_АТОМЕ} полей: {строка!r}")
        try:
            xyz = (float(поля[2]), float(поля[3]), float(поля[4]))
        except ValueError as ошибка:
            raise Mol2ReadError(f"{path}: координаты не читаются: {строка!r}") from ошибка
        # Заряд и имя подструктуры необязательны: у лиганда из KLIFS они есть,
        # но формат их не требует, и отсутствие не повод ронять разбор.
        заряд = float(поля[8]) if len(поля) > 8 else 0.0
        подструктура = поля[7] if len(поля) > 7 else ""
        тип = поля[5]
        атомы.append(
            Mol2Atom(
                name=поля[1],
                sybyl=тип,
                element=_элемент(тип),
                xyz=xyz,
                charge=заряд,
                residue=подструктура,
            )
        )
    if not атомы:
        raise Mol2ReadError(f"в mol2 нет атомов: {path}")

    связи: list[tuple[int, int]] = []
    границы_связей = _границы(строки, _БЛОК_СВЯЗЕЙ)
    if границы_связей is not None:
        for строка in строки[границы_связей[0] : границы_связей[1]]:
            поля = строка.split()
            if not поля:
                continue
            if len(поля) < _ПОЛЕЙ_В_СВЯЗИ:
                raise Mol2ReadError(f"{path}: строка связи короче четырёх полей: {строка!r}")
            try:
                левый, правый = int(поля[1]) - 1, int(поля[2]) - 1
            except ValueError as ошибка:
                raise Mol2ReadError(
                    f"{path}: номера атомов связи не читаются: {строка!r}"
                ) from ошибка
            if not (0 <= левый < len(атомы) and 0 <= правый < len(атомы)):
                raise Mol2ReadError(f"{path}: связь ссылается на несуществующий атом: {строка!r}")
            связи.append((левый, правый))
    return Mol2Structure(atoms=tuple(атомы), bonds=tuple(связи))
