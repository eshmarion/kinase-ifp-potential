"""Тесты загрузки кармана."""

from __future__ import annotations

import shutil
from pathlib import Path

import MDAnalysis as mda
import pytest

from kinase_ifp.config import KLIFS_IFP_SHAPE, N_KLIFS_POSITIONS
from kinase_ifp.klifs import KlifsDataError
from kinase_ifp.pocket import (
    IMPLICIT_PROTONATION,
    Pocket,
    load_pocket,
    load_protein,
    pocket_residue_ids,
)


def test_пакет_мишени_разбирается(pocket: Pocket) -> None:
    assert pocket.pdb_id == "6tgu"
    assert pocket.klifs_structure_id == 12448
    assert pocket.ligand_path.is_file()
    # Считаем по карману, а не по всему белку: 85 канонических позиций — это он и есть.
    assert pocket.protein_path.name == "pocket.pdb"
    assert pocket.protonation == "explicit"


def test_позиции_кармана_в_допустимом_диапазоне(pocket: Pocket) -> None:
    positions = list(pocket.residue_to_position.values())
    assert positions
    assert all(1 <= p <= N_KLIFS_POSITIONS for p in positions)
    # Позиция KLIFS адресует один остаток: повтор означал бы, что карман собран неверно.
    assert len(set(positions)) == len(positions)


def test_эталонный_отпечаток_имеет_форму_отпечатка(pocket: Pocket) -> None:
    assert pocket.reference_ifp is not None
    assert pocket.reference_ifp.shape == KLIFS_IFP_SHAPE


def test_остатки_белка_совпадают_с_картой_позиций(pocket: Pocket) -> None:
    """Главный стык: ProLIF должен называть остатки ровно так же, как пакет мишени.

    Идентичность остатка восстанавливает `structure_io.write_pdb_from_mol2`;
    здесь проверяется, что после неё сходятся все до одного.
    """
    protein = load_protein(pocket)
    prolif_residues = {str(residue.resid) for residue in protein}

    missing = set(pocket.residue_to_position) - prolif_residues
    assert not missing, f"остатки карты не найдены в белке: {sorted(missing)[:5]}"


def test_карман_читается_из_кэша(pocket: Pocket) -> None:
    """Для набора из 100 поз карман разбирается один раз, а не заново на каждую молекулу."""
    assert load_protein(pocket) is load_protein(pocket)


def test_белок_приходит_с_водородами(pocket: Pocket) -> None:
    """KLIFS отдаёт карман протонированным — без этого не определить донор H-связи."""
    protein = load_protein(pocket)
    hydrogens = sum(1 for atom in protein.GetAtoms() if atom.GetAtomicNum() == 1)
    assert hydrogens > 0


def test_в_режиме_implicit_берётся_файл_без_водородов(tmp_path: Path, target_json: Path) -> None:
    """Типы ImplicitHB* предполагают структуру без водородов — и именно файл, а не фильтр.

    Отфильтровать водороды в памяти нельзя: `select_atoms("not element H")` оставляет
    связи с удалёнными атомами, и ProLIF роняет процесс access violation'ом внутри
    `split_mol_by_residues`, а не поднимает исключение. Поэтому сборка пакета пишет отдельный
    `protein_noh.pdb`, и режим читает его.
    """
    shutil.copytree(target_json.parent, tmp_path / "6tgu")
    package = tmp_path / "6tgu" / "target.json"
    package.write_text(
        package.read_text(encoding="utf-8").replace('"explicit"', f'"{IMPLICIT_PROTONATION}"'),
        encoding="utf-8",
    )

    implicit = load_pocket(package)

    assert implicit.protein_path.name == "protein_noh.pdb"
    protein = load_protein(implicit)
    assert all(atom.GetAtomicNum() != 1 for atom in protein.GetAtoms())


def test_неизвестный_режим_протонирования_отвергается(tmp_path: Path, target_json: Path) -> None:
    """Проверку делает validate_target_json; здесь важно, что load_pocket её зовёт."""
    shutil.copytree(target_json.parent, tmp_path / "6tgu")
    broken = tmp_path / "6tgu" / "target.json"
    broken.write_text(
        broken.read_text(encoding="utf-8").replace('"explicit"', '"своё-какое-то"'),
        encoding="utf-8",
    )

    with pytest.raises(KlifsDataError, match="protonation="):
        load_pocket(broken)


def test_пакет_собранный_до_y16_отвергается(tmp_path: Path, target_json: Path) -> None:
    """Пакет с потерянной идентичностью остатка обязан падать, а не давать пустой отпечаток.

    Ровно так выглядела прямая конверсия mol2 → PDB: цепь терялась, и ни один остаток
    кармана не находился в структуре. Расчёт при этом не падал — и это опаснее падения.
    """
    shutil.copytree(target_json.parent, tmp_path / "6tgu")
    pocket_pdb = tmp_path / "6tgu" / "pocket.pdb"

    universe = mda.Universe(str(pocket_pdb))
    universe.atoms.chainIDs = "X"
    universe.atoms.write(str(pocket_pdb))

    with pytest.raises(KlifsDataError, match="старым способом"):
        load_protein(load_pocket(tmp_path / "6tgu" / "target.json"))


def test_остатки_кармана_в_формате_diffsbdd(pocket: Pocket) -> None:
    """Второй способ задать карман при генерации (В-13): явный список остатков."""
    ids = pocket_residue_ids(pocket)

    assert len(ids) == len(pocket.residue_to_position)
    assert all(идентификатор.startswith("A:") for идентификатор in ids)
    # Порядок — по канонической позиции KLIFS: два запуска обязаны дать одну строку.
    номера = [int(идентификатор.split(":")[1]) for идентификатор in ids]
    позиции = sorted(pocket.residue_to_position.values())
    assert len(номера) == len(позиции)
    assert ids == pocket_residue_ids(pocket)


def test_остаток_без_цепи_отвергается(pocket: Pocket) -> None:
    """Без цепи список неполон, а DiffSBDD примет его молча и возьмёт не тот карман."""
    сломанный = Pocket(**{**pocket.__dict__, "residue_to_position": {"ARG44": 1}})

    with pytest.raises(KlifsDataError, match="не разбирается"):
        pocket_residue_ids(сломанный)
