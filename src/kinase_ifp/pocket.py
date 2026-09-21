"""Карман связывания как объект: загрузка пакета мишени и белка.

Идентичность остатка (имя, номер, цепь) восстанавливается не здесь, а при сборке пакета:
`structure_io.write_pdb_from_mol2` пишет PDB по mol2 без потерь, и к моменту расчёта
файлы пакета уже пригодны. Здесь остаётся проверить, что это действительно так:
пакет, собранный старым способом, даёт пустой отпечаток молча, и такой случай отсекается
явной ошибкой, а не оставляется на потом.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import prolif

from kinase_ifp.klifs import KlifsDataError, validate_target_json

# Режим, в котором водороды белка не используются: типы ImplicitHBDonor/ImplicitHBAcceptor
# рассчитаны на структуры из PDB, где водородов нет вовсе.
IMPLICIT_PROTONATION: str = "implicit-prolif"


@dataclass(frozen=True)
class Pocket:
    """Карман мишени: пути к структурам и соответствие остатков позициям KLIFS."""

    klifs_structure_id: int
    pdb_id: str
    protein_path: Path
    ligand_path: Path
    residue_to_position: dict[str, int]
    reference_ifp: np.ndarray | None
    protonation: str


def load_pocket(target_json: Path) -> Pocket:
    """Читает `data/targets/<pdb_id>/target.json` и собирает `Pocket` по формату чтения кармана.

    Поднимает `KlifsDataError`, если пакет не соответствует формату пакета мишени — в том числе
    если объявлен неизвестный режим протонирования.
    """
    # Проверки формата пакета мишени уже написаны в klifs.py и здесь не дублируются.
    validate_target_json(target_json)

    package = json.loads(target_json.read_text(encoding="utf-8"))
    base = target_json.parent

    # Допустимость режима проверяет validate_target_json — здесь не дублируем.
    protonation = str(package["protonation"])

    bits = package["klifs_ifp_bits"]
    reference_ifp = None
    if bits is not None:
        # Импорт внутри функции: fingerprint.py типизируется через Pocket, и импорт
        # на уровне модуля замкнул бы два модуля друг на друга.
        from kinase_ifp.fingerprint import bits_to_ifp

        reference_ifp = bits_to_ifp(bits)

    # В режиме explicit берётся карман: взаимодействия считаются по 85 каноническим
    # позициям, и читать ради этого файл впятеро больше незачем. В режиме implicit-prolif
    # берётся `protein_noh.pdb` — готовый файл без водородов. Отфильтровать
    # водороды в памяти нельзя: `select_atoms("not element H")` оставляет связи с
    # удалёнными атомами, и ProLIF падает с access violation внутри
    # `split_mol_by_residues` — то есть роняет процесс, а не поднимает исключение.
    structure = (
        package["protein_noh_pdb"]
        if protonation == IMPLICIT_PROTONATION
        else package["pocket_pdb"]
    )

    return Pocket(
        klifs_structure_id=int(package["klifs_structure_id"]),
        pdb_id=str(package["pdb_id"]),
        protein_path=base / structure,
        ligand_path=base / package["ligand_sdf"],
        residue_to_position=dict(package["residue_to_position"]),
        reference_ifp=reference_ifp,
        protonation=protonation,
    )


def load_protein(pocket: Pocket) -> prolif.Molecule:
    """Загружает карман белка в виде молекулы ProLIF.

    Файл выбран в `load_pocket` по режиму протонирования; здесь он только читается.

    Поднимает `KlifsDataError`, если ни один остаток кармана не сопоставился с картой
    позиций: это признак пакета, собранного старым способом, и без проверки он дал бы не
    ошибку, а пустой отпечаток.
    """
    protein = _load_protein_cached(
        pocket.protein_path, pocket.protonation == IMPLICIT_PROTONATION
    )

    known = {str(residue.resid) for residue in protein}
    matched = sum(1 for residue in pocket.residue_to_position if residue in known)
    if matched == 0:
        raise KlifsDataError(
            f"{pocket.protein_path}: ни один из {len(pocket.residue_to_position)} остатков "
            f"кармана не найден в структуре. Так выглядит пакет мишени, собранный старым способом: "
            f"идентичность остатка потеряна при конверсии mol2 → PDB. Перестройте пакет "
            f"командой scripts/annotate_protonation.py, иначе отпечаток выйдет пустым"
        )
    return protein


# Карман читается один раз на прогон: для набора из 100 поз повторный разбор
# файла и построение молекулы ProLIF заняли бы больше времени, чем сам расчёт.
@lru_cache(maxsize=4)
def _load_protein_cached(path: Path, implicit_hydrogens: bool) -> prolif.Molecule:
    universe = mda.Universe(str(path))
    if universe.atoms.n_atoms == 0:
        raise KlifsDataError(f"{path}: в файле структуры нет ни одного атома")

    # force=True: MDAnalysis требует явного согласия на конверсию структуры без явных
    # порядков связей — для белка из PDB они выводятся по геометрии и именам атомов.
    #
    # NoImplicit=False обязателен для структуры без водородов и только для неё. По
    # умолчанию конвертер запрещает RDKit достраивать неявные водороды, и у атомов
    # депротонированного белка остаются незаполненные валентности; ProLIF на такой
    # молекуле падает access violation'ом внутри `split_mol_by_residues`, то есть
    # роняет процесс целиком, а не поднимает исключение. Для протонированной структуры
    # флаг не нужен: водороды в ней уже явные.
    #
    # Словарь вынесен в переменную с типом dict[str, Any] не для красоты: при распаковке
    # прямо в вызов mypy примеряет её к третьему параметру `selection: str | None`
    # и валится с arg-type, хотя на самом деле аргумент уходит в **kwargs конвертера.
    converter_kwargs: dict[str, Any] = {"NoImplicit": False} if implicit_hydrogens else {}
    return prolif.Molecule.from_mda(universe.atoms, force=True, **converter_kwargs)


def residue_ids_from_mapping(residue_to_position: dict[str, int]) -> list[str]:
    """Переводит карту остатков в формат DiffSBDD `<цепь>:<номер>` для `--pocket_ids`.

    Второй способ задать карман при генерации (`docs/questions.md`, В-13): вместо
    окружения кристаллического лиганда модели передаётся явный список остатков. Берётся
    он из `residue_to_position`, то есть из той же разметки KLIFS, по которой считается
    отпечаток, — карман для генерации и карман для скора совпадают по построению.

    Порядок — по канонической позиции KLIFS, чтобы список не зависел от порядка ключей
    в JSON и два запуска давали одинаковую строку аргументов.

    Принимает словарь, а не `Pocket`, потому что вызывается и из кода, работающего
    в облаке, где пакет мишени читается как обычный JSON.

    Формат взят из описания в В-13 и на живом DiffSBDD ещё не проверялся: клон модели
    в репозиторий не входит, а генерация идёт в Modal.
    """
    ids: list[str] = []
    for остаток, _ in sorted(residue_to_position.items(), key=lambda пара: пара[1]):
        имя, _, цепь = остаток.rpartition(".")
        номер = "".join(символ for символ in имя if символ.isdigit() or символ == "-")
        if not номер or not цепь:
            raise KlifsDataError(
                f"Остаток {остаток!r} не разбирается на номер и цепь: список для "
                f"--pocket_ids из него не построить"
            )
        ids.append(f"{цепь}:{номер}")
    return ids


def pocket_residue_ids(pocket: Pocket) -> list[str]:
    """Остатки кармана в формате DiffSBDD `<цепь>:<номер>`."""
    return residue_ids_from_mapping(pocket.residue_to_position)
