"""Перевод файлов структур между форматами без потери идентичности остатка.

KLIFS отдаёт координаты белка и кармана только в mol2, а формат пакета мишени требует PDB.
Прямая конвертация средствами MDAnalysis идентичность остатка теряет: в mol2 имя
и номер лежат в одном поле `subst_name` («ARG44»), MDAnalysis кладёт всю строку
в `resname`, а `resid` берёт из сквозного счётчика `subst_id`. На выходе получается
имя «ARG4» (обрезано до четырёх символов), номер 1 и цепь «X», которой в mol2 нет
вовсе. ProLIF адресует остатки как `RESNAME<номер>.<цепь>`, поэтому с ключами
`residue_to_position` из `target.json` не совпадает ни один остаток — отпечаток
выходит пустым, а код при этом не падает.

Измерено на мишени 6tgu 22.08.2026: со старой конвертацией ProLIF сопоставлял
0 остатков из 84, с восстановленной — 84 из 84.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import MDAnalysis as mda

# Имя подструктуры mol2: трёхбуквенное имя остатка, номер и необязательный код вставки
# («ARG44», «HIS161A»). Отрицательный номер допустим: в кристаллических структурах
# нумерация может начинаться до первого остатка последовательности.
#
# Минус KLIFS записывает **подчёркиванием**: у 2j90 (DAPK3) остатки тэга экспрессии
# идут «PHE_3, GLN_2, SER_1, MET0, VAL9», то есть −3, −2, −1, 0. Форма с дефисом была
# предусмотрена сразу, но источник её не порождает, и проверить это было не на чем:
# в 6tgu отрицательных номеров нет вовсе. Пока обе формы не принимались, структура
# с тэгом роняла сборку пакета целиком — карман при этом разбирается полностью,
# потому что разметка KLIFS начинается после стартового метионина.
RESIDUE_LABEL: Final[re.Pattern[str]] = re.compile(r"^([A-Za-z]{1,3})([-_]?\d+)([A-Za-z]?)$")

# Имена файлов внутри каталога мишени названы явно, поэтому они
# заданы здесь один раз, а не собираются из кусков в каждом вызове.
PROTEIN_PDB: Final[str] = "protein.pdb"
POCKET_PDB: Final[str] = "pocket.pdb"
PROTEIN_NOH_PDB: Final[str] = "protein_noh.pdb"


class StructureIOError(RuntimeError):
    """Файл структуры не разбирается или разбирается не так, как ожидает конвейер.

    Отдельный класс от `KlifsDataError`: то — «база отдала не те данные», это —
    «файл на диске не соответствует формату», и чинятся они разными способами.
    """


def split_residue_label(label: str) -> tuple[str, int, str]:
    """Разбирает имя подструктуры mol2 на имя остатка, номер и код вставки.

    `"ARG44"` → `("ARG", 44, "")`, `"HIS161A"` → `("HIS", 161, "A")`,
    `"SER_1"` → `("SER", -1, "")` — подчёркивание у KLIFS означает минус.
    Метку, которая не разбирается, функция не пропускает: молчаливый пропуск и есть
    механизм, которым терялась идентичность остатка.
    """
    match = RESIDUE_LABEL.match(label.strip())
    if match is None:
        raise StructureIOError(
            f"Имя подструктуры mol2 {label!r} не разбирается на имя остатка и номер: "
            f"восстановить нотацию ProLIF нельзя"
        )
    return match.group(1).upper(), int(match.group(2).replace("_", "-")), match.group(3).upper()


def write_pdb_from_mol2(mol2_path: Path, pdb_path: Path, chain: str) -> None:
    """Пишет PDB по mol2, восстанавливая имя, номер и код вставки остатка.

    `chain` проставляется один на весь файл: формат mol2 сведений о цепи не хранит,
    а KLIFS выравнивает карман в пределах одной цепи — так же и по той же причине
    подставляет цепь `build_residue_to_position` в `klifs.py`.
    """
    if not mol2_path.is_file():
        raise StructureIOError(f"Файл mol2 не найден: {mol2_path}")
    if not chain:
        raise StructureIOError(
            f"Для {mol2_path} не задана цепь: без неё нотация ProLIF неполна "
            f"и остатки не сопоставятся с позициями KLIFS"
        )

    universe = mda.Universe(str(mol2_path))
    parsed = [split_residue_label(label) for label in universe.residues.resnames]

    # Атрибутов цепи и кода вставки у прочитанного mol2 нет вовсе — их заводим, а не
    # присваиваем: присваивание несуществующему атрибуту MDAnalysis не создаёт.
    universe.add_TopologyAttr("chainIDs")
    universe.add_TopologyAttr("icodes")
    universe.residues.resnames = [name for name, _, _ in parsed]
    universe.residues.resids = [number for _, number, _ in parsed]
    universe.residues.icodes = [icode for _, _, icode in parsed]
    universe.atoms.chainIDs = chain

    universe.atoms.write(str(pdb_path))


def count_hydrogens(path: Path) -> int:
    """Считает атомы водорода в файле структуры.

    Признак того, что структура протонирована: ProLIF отличает донора водородной связи
    от акцептора по положению самого водорода, и без него находит одни ван-дер-ваальсовы
    касания. Ровно так сломался черновик (`source_files/мяу.py:2369-2371`).
    """
    if not path.is_file():
        raise StructureIOError(f"Файл структуры не найден: {path}")

    universe = mda.Universe(str(path))
    try:
        elements = universe.atoms.elements
    except AttributeError as error:
        raise StructureIOError(
            f"{path}: у атомов нет химических элементов, посчитать водороды нельзя"
        ) from error
    return int(sum(1 for element in elements if str(element).strip().upper() == "H"))


def write_without_hydrogens(source: Path, destination: Path) -> None:
    """Пишет копию структуры без атомов водорода.

    Нужен второму режиму сверки (`implicit-prolif`): типы
    `ImplicitHBDonor` и `ImplicitHBAcceptor` рассчитаны на структуры из PDB, где
    водородов нет, и сравнивать режимы надо на файле, где их действительно нет,
    а не полагаться на то, что ProLIF их проигнорирует.
    """
    universe = mda.Universe(str(source))
    heavy = universe.select_atoms("not element H")
    if heavy.n_atoms == 0:
        raise StructureIOError(f"{source}: после удаления водородов не осталось атомов")
    heavy.write(str(destination))


def rebuild_pdb_files(target_dir: Path, chain: str) -> dict[str, str]:
    """Перестраивает PDB-файлы пакета мишени из лежащих рядом mol2.

    Возвращает относительные пути для полей `protein_pdb`, `pocket_pdb`
    и `protein_noh_pdb` формата пакета мишени. Сеть не нужна: mol2 уже скачаны, поэтому
    той же функцией и собирается новый пакет, и чинится уже собранный.
    """
    for entity, pdb_name in (("protein", PROTEIN_PDB), ("pocket", POCKET_PDB)):
        write_pdb_from_mol2(target_dir / f"{entity}.mol2", target_dir / pdb_name, chain)

    write_without_hydrogens(target_dir / PROTEIN_PDB, target_dir / PROTEIN_NOH_PDB)

    return {
        "protein_pdb": PROTEIN_PDB,
        "pocket_pdb": POCKET_PDB,
        "protein_noh_pdb": PROTEIN_NOH_PDB,
    }
