"""Тесты фиксации режима протонирования в пакете мишени.

Проверяется не «код отработал без исключения», а то, что режим виден в данных:
поле `protonation` заполнено по факту подсчёта водородов, а не по умолчанию, и что
белок без водородов режимом `explicit` не помечается.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kinase_ifp.klifs import TARGET_JSON_KEYS, KlifsDataError, canonical_target_json
from kinase_ifp.protonation import KLIFS_PROTONATION_TOOL, annotate_protonation
from kinase_ifp.structure_io import PROTEIN_NOH_PDB, PROTEIN_PDB, count_hydrogens

# Помощник сборки пакета живёт рядом с тестами выбора мишени: пакет один и тот же, а две
# копии схемы разошлись бы при первой же правке формата пакета мишени.
from tests.test_klifs import make_target_package

FIXTURES = Path(__file__).parent / "fixtures"
MINI_POCKET = FIXTURES / "mini_pocket.mol2"
MINI_POCKET_NOH = FIXTURES / "mini_pocket_noh.mol2"


def make_package_with_structures(base: Path, mol2: Path = MINI_POCKET, **overrides: object) -> Path:
    """Пакет мишени, рядом с которым лежат mol2 белка и кармана."""
    target_json = make_target_package(base, **overrides)
    for entity in ("protein", "pocket"):
        (base / f"{entity}.mol2").write_text(mol2.read_text(encoding="utf-8"), encoding="utf-8")
    return target_json


class TestAnnotateProtonation:
    def test_проставляет_явный_режим_и_инструмент(self, tmp_path: Path) -> None:
        target_json = make_package_with_structures(tmp_path / "1abc")

        package = annotate_protonation(target_json)

        assert package["protonation"] == "explicit"
        assert package["protonation_tool"] == KLIFS_PROTONATION_TOOL

    def test_записывает_режим_в_файл_а_не_только_возвращает(self, tmp_path: Path) -> None:
        target_json = make_package_with_structures(tmp_path / "1abc")

        annotate_protonation(target_json)

        saved = json.loads(target_json.read_text(encoding="utf-8"))
        assert saved["protonation"] == "explicit"
        assert saved["protonation_tool"]

    def test_создаёт_белок_без_водородов_для_второго_режима(self, tmp_path: Path) -> None:
        base = tmp_path / "1abc"
        target_json = make_package_with_structures(base)

        package = annotate_protonation(target_json)

        assert package["protein_noh_pdb"] == PROTEIN_NOH_PDB
        assert count_hydrogens(base / PROTEIN_PDB) > 0
        assert count_hydrogens(base / PROTEIN_NOH_PDB) == 0

    def test_чинит_идентичность_остатков(self, tmp_path: Path) -> None:
        # До 22.08.2026 здесь оказывалось «GLY7» с номером 1 и цепью «X», и ключи
        # residue_to_position не совпадали ни с одним остатком.
        base = tmp_path / "1abc"
        target_json = make_package_with_structures(base)

        annotate_protonation(target_json)

        first = next(
            line
            for line in (base / "pocket.pdb").read_text(encoding="utf-8").splitlines()
            if line.startswith("ATOM")
        )
        assert first[17:20].strip() == "GLY"
        assert first[21] == "A"
        assert int(first[22:26]) == 7

    def test_белок_без_водородов_не_выдаётся_за_explicit(self, tmp_path: Path) -> None:
        target_json = make_package_with_structures(tmp_path / "1abc", mol2=MINI_POCKET_NOH)

        with pytest.raises(KlifsDataError, match="водорода 0"):
            annotate_protonation(target_json)

    def test_режим_implicit_сохраняет_мишень_без_водородов(self, tmp_path: Path) -> None:
        # Терять структуру целиком хуже, чем считать её
        # слабее. Водороды при этом никто не достраивает — их просто нет, поэтому
        # инструмент остаётся пустым.
        target_json = make_package_with_structures(tmp_path / "1abc", mol2=MINI_POCKET_NOH)

        package = annotate_protonation(target_json, on_missing_hydrogens="implicit")

        assert package["protonation"] == "implicit-prolif"
        assert package["protonation_tool"] is None

    def test_режим_implicit_не_понижает_протонированный_белок(self, tmp_path: Path) -> None:
        # Разрешение на понижение — не требование понизить: у структуры с водородами
        # режим обязан остаться explicit, иначе сверка потеряла бы 0.6 Танимото зря.
        target_json = make_package_with_structures(tmp_path / "1abc")

        package = annotate_protonation(target_json, on_missing_hydrogens="implicit")

        assert package["protonation"] == "explicit"

    def test_падает_на_негодном_значении_параметра(self, tmp_path: Path) -> None:
        target_json = make_package_with_structures(tmp_path / "1abc")

        with pytest.raises(ValueError, match="on_missing_hydrogens"):
            annotate_protonation(target_json, on_missing_hydrogens="как-нибудь")

    def test_падает_без_цепи(self, tmp_path: Path) -> None:
        target_json = make_package_with_structures(tmp_path / "1abc", chain="")

        with pytest.raises(KlifsDataError, match="chain"):
            annotate_protonation(target_json)

    def test_падает_без_пакета(self, tmp_path: Path) -> None:
        with pytest.raises(KlifsDataError, match="не найден"):
            annotate_protonation(tmp_path / "target.json")


def test_порядок_ключей_пакета_не_зависит_от_истории_сборки(tmp_path: Path) -> None:
    """Хэш пакета считается по байтам файла, а `json.dumps` хранит порядок вставки.

    Пакет, собранный до появления `protein_noh_pdb`, чинился на месте, и новый ключ
    уезжал в конец: тот же по содержанию пакет давал другой `package_sha256`, расчёт
    отказывался считать по «чужому» пакету, который на самом деле свой, и разбор этого
    занял вечер. Порядок ключей поэтому канонический, а не «как вышло».
    """
    target_json = make_package_with_structures(tmp_path)
    annotate_protonation(target_json, on_missing_hydrogens="implicit")
    эталон = target_json.read_bytes()

    пакет = json.loads(эталон.decode("utf-8"))
    кривой = {имя: значение for имя, значение in пакет.items() if имя != "protein_noh_pdb"}
    кривой["protein_noh_pdb"] = пакет["protein_noh_pdb"]
    assert list(кривой) != list(пакет), "перестановка не состоялась — тест ничего не проверяет"

    выправленный = json.dumps(canonical_target_json(кривой), indent=2, ensure_ascii=False)
    assert выправленный.encode("utf-8") == эталон


def test_незнакомый_ключ_пакета_не_теряется() -> None:
    """Молча выбросить поле хуже, чем оставить хэш зависимым от него."""
    выправленный = canonical_target_json({"pocket_sequence": "AA", "чужое": 1, "pdb_id": "6tgu"})

    assert list(выправленный) == ["pdb_id", "pocket_sequence", "чужое"]


def test_все_поля_собранного_пакета_перечислены_в_порядке(tmp_path: Path) -> None:
    """Иначе новое поле пакета молча уедет в хвост и снова разведёт хэши."""
    пакет = json.loads(make_package_with_structures(tmp_path).read_text(encoding="utf-8"))

    assert set(пакет) <= set(TARGET_JSON_KEYS), (
        f"поля пакета вне TARGET_JSON_KEYS: {sorted(set(пакет) - set(TARGET_JSON_KEYS))}"
    )
