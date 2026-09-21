"""Чтение PDB пакета мишени по остаткам.

Ключ остатка собирается в том же виде, что и ключ `residue_to_position` из
`target.json`: `<RES><NUM><код вставки>.<CHAIN>`. Поэтому сопоставление
атомов с позициями кармана не требует отдельной таблицы.

Водороды **читаются и возвращаются**: правилам водородной связи KLIFS
(`klifs_rules`) донор определяется наличием явного водорода, и без них бит `DON`
не поставить вовсе. Кому водороды мешают, тот их отбрасывает сам — разборщик
решает за вызывающего только формат, но не состав.

Модуль намеренно не зависит ни от RDKit, ни от MDAnalysis: он читает колонки
фиксированной ширины по спецификации PDB, и этого достаточно. Разбор через RDKit
потребовал бы санитизации, которая на белке из KLIFS отвергает то, что KLIFS
при расчёте эталона принял.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_ЗАПИСИ_АТОМОВ = ("ATOM", "HETATM")


class PdbReadError(RuntimeError):
    """PDB нельзя разобрать: нет файла, нет ни одной записи атома, битые координаты."""


@dataclass(frozen=True)
class PdbAtom:
    """Атом из записи PDB: имя, химический элемент и координаты."""

    name: str
    element: str
    xyz: tuple[float, float, float]


def read_pdb_residues(path: Path) -> dict[str, list[PdbAtom]]:
    """Атомы PDB, сгруппированные по остаткам; ключ — как в `residue_to_position`.

    Возвращает все атомы, включая водороды. Элемент берётся из колонок 77–78,
    а при их отсутствии — из первого символа имени атома.
    """
    if not path.is_file():
        raise PdbReadError(f"нет файла структуры: {path}")
    по_остаткам: dict[str, list[PdbAtom]] = {}
    for номер_строки, строка in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not строка.startswith(_ЗАПИСИ_АТОМОВ):
            continue
        имя = строка[12:16].strip()
        элемент = (строка[76:78].strip() or имя[:1]).upper()
        остаток = строка[17:20].strip()
        цепь = строка[21].strip() or "A"
        номер = строка[22:26].strip()
        код_вставки = строка[26].strip()
        try:
            xyz = (float(строка[30:38]), float(строка[38:46]), float(строка[46:54]))
        except ValueError as ошибка:
            raise PdbReadError(
                f"{path}, строка {номер_строки}: координаты не читаются"
            ) from ошибка
        ключ = f"{остаток}{номер}{код_вставки}.{цепь}"
        по_остаткам.setdefault(ключ, []).append(PdbAtom(имя, элемент, xyz))
    if not по_остаткам:
        raise PdbReadError(f"в структуре нет атомов: {path}")
    return по_остаткам
