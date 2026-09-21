"""Тесты перевода структур mol2 → PDB и работы с водородами.

Главный тест здесь — `test_нотация_prolif_совпадает_с_ключом`: он падает на той
конвертации, которой пакет мишени собирался до 22.08.2026. Тогда MDAnalysis клал
«ARG44» целиком в имя остатка, а номер брал из сквозного счётчика, и ProLIF
не сопоставлял с позициями KLIFS ни одного остатка кармана.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kinase_ifp.structure_io import (
    POCKET_PDB,
    PROTEIN_NOH_PDB,
    PROTEIN_PDB,
    StructureIOError,
    count_hydrogens,
    rebuild_pdb_files,
    split_residue_label,
    write_pdb_from_mol2,
    write_without_hydrogens,
)

FIXTURES = Path(__file__).parent / "fixtures"
MINI_POCKET = FIXTURES / "mini_pocket.mol2"
MINI_POCKET_NOH = FIXTURES / "mini_pocket_noh.mol2"

# Столбцы PDB, в которых лежит идентичность остатка (нумерация с нуля).
RESNAME = slice(17, 20)
CHAIN = 21
RESID = slice(22, 26)


def atom_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith("ATOM")]


def prolif_residue(line: str) -> str:
    """Собирает нотацию ProLIF `RESNAME<номер>.<цепь>` из строки ATOM."""
    return f"{line[RESNAME].strip()}{int(line[RESID])}.{line[CHAIN]}"


class TestРазбораМетки:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("ARG44", ("ARG", 44, "")),
            ("HIS161A", ("HIS", 161, "A")),
            ("his161a", ("HIS", 161, "A")),
            # Нумерация кристаллической структуры может начинаться до первого остатка.
            ("GLY-5", ("GLY", -5, "")),
            ("  SER8  ", ("SER", 8, "")),
            # Так отрицательный номер пишет сам KLIFS: подчёркивание вместо минуса.
            # Форма взята из 2j90 (DAPK3), где тэг экспрессии идёт до стартового
            # метионина: «PHE_3, GLN_2, SER_1, MET0».
            ("SER_1", ("SER", -1, "")),
            ("PHE_3", ("PHE", -3, "")),
        ],
    )
    def test_разбирает_имя_и_номер(self, label: str, expected: tuple[str, int, str]) -> None:
        assert split_residue_label(label) == expected

    def test_подчёркивание_и_дефис_дают_один_номер(self) -> None:
        """Обе формы минуса означают одно: иначе остаток раздвоился бы по нотации ProLIF."""
        assert split_residue_label("SER_1") == split_residue_label("SER-1")

    @pytest.mark.parametrize("label", ["ARG", "44", "", "ARG44XY", "A1B2"])
    def test_не_пропускает_неразбираемое(self, label: str) -> None:
        with pytest.raises(StructureIOError):
            split_residue_label(label)


class TestКонвертацииВPdb:
    def test_восстанавливает_имя_номер_и_цепь(self, tmp_path: Path) -> None:
        out = tmp_path / "out.pdb"
        write_pdb_from_mol2(MINI_POCKET, out, "A")

        first = atom_lines(out)[0]
        assert first[RESNAME].strip() == "GLY"
        assert first[CHAIN] == "A"
        assert int(first[RESID]) == 7

    def test_нотация_prolif_совпадает_с_ключом(self, tmp_path: Path) -> None:
        # Ключи residue_to_position в target.json выглядят именно так; несовпадение
        # означает пустой отпечаток при исправном коде расчёта.
        out = tmp_path / "out.pdb"
        write_pdb_from_mol2(MINI_POCKET, out, "A")

        assert {prolif_residue(line) for line in atom_lines(out)} == {"GLY7.A", "SER8.A"}

    def test_число_атомов_не_меняется(self, tmp_path: Path) -> None:
        out = tmp_path / "out.pdb"
        write_pdb_from_mol2(MINI_POCKET, out, "A")

        assert len(atom_lines(out)) == 14

    def test_падает_без_цепи(self, tmp_path: Path) -> None:
        with pytest.raises(StructureIOError, match="цепь"):
            write_pdb_from_mol2(MINI_POCKET, tmp_path / "out.pdb", "")

    def test_падает_без_исходного_файла(self, tmp_path: Path) -> None:
        with pytest.raises(StructureIOError, match="не найден"):
            write_pdb_from_mol2(tmp_path / "нет.mol2", tmp_path / "out.pdb", "A")


class TestВодородов:
    def test_считает_водороды(self, tmp_path: Path) -> None:
        out = tmp_path / "out.pdb"
        write_pdb_from_mol2(MINI_POCKET, out, "A")

        assert count_hydrogens(out) == 4

    def test_структура_без_водородов_даёт_ноль(self, tmp_path: Path) -> None:
        out = tmp_path / "out.pdb"
        write_pdb_from_mol2(MINI_POCKET_NOH, out, "A")

        assert count_hydrogens(out) == 0

    def test_снимает_водороды_не_трогая_тяжёлые_атомы(self, tmp_path: Path) -> None:
        source = tmp_path / "out.pdb"
        stripped = tmp_path / "out_noh.pdb"
        write_pdb_from_mol2(MINI_POCKET, source, "A")

        write_without_hydrogens(source, stripped)

        assert count_hydrogens(stripped) == 0
        assert len(atom_lines(stripped)) == len(atom_lines(source)) - 4

    def test_падает_на_несуществующем_файле(self, tmp_path: Path) -> None:
        with pytest.raises(StructureIOError, match="не найден"):
            count_hydrogens(tmp_path / "нет.pdb")


class TestПересборкиПакета:
    def test_создаёт_три_файла_и_возвращает_их_имена(self, tmp_path: Path) -> None:
        for entity in ("protein", "pocket"):
            (tmp_path / f"{entity}.mol2").write_text(
                MINI_POCKET.read_text(encoding="utf-8"), encoding="utf-8"
            )

        paths = rebuild_pdb_files(tmp_path, "A")

        assert paths == {
            "protein_pdb": PROTEIN_PDB,
            "pocket_pdb": POCKET_PDB,
            "protein_noh_pdb": PROTEIN_NOH_PDB,
        }
        for name in paths.values():
            assert (tmp_path / name).is_file()
        assert count_hydrogens(tmp_path / PROTEIN_NOH_PDB) == 0

    def test_падает_если_mol2_нет(self, tmp_path: Path) -> None:
        with pytest.raises(StructureIOError, match="не найден"):
            rebuild_pdb_files(tmp_path, "A")
