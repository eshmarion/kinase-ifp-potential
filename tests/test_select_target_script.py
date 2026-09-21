"""Тесты CLI выбора мишени.

Проверяется одно: сборка пакета и простановка режима протонирования — один шаг,
а не два. Отдельной командой о режиме забывали, и 24.08 два пакета разошлись молча.

В KLIFS тест не ходит: сеть и сборка пакета подменены, важен порядок шагов.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from kinase_ifp.klifs import KlifsDataError

from .test_klifs import VALID_BITS, make_structure_row, make_target_package

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_target.py"


def load_script() -> Any:
    """Загружает `scripts/select_target.py` как модуль.

    Каталог `scripts/` намеренно не лежит на `pythonpath` (там тонкие CLI-входы,
    а не библиотека), поэтому модуль загружается по пути файла, а не импортом по имени.
    """
    spec = importlib.util.spec_from_file_location("select_target_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Готовит CLI к запуску: таблица структур на диске, сеть и сборка пакета подменены."""
    module = load_script()

    structures_csv = tmp_path / "structures.csv"
    pd.DataFrame([make_structure_row()]).to_csv(structures_csv, index=False)

    # Пакет приходит из сборки без режима — ровно то состояние, которое обязан
    # закрыть следующий шаг.
    target_json = make_target_package(tmp_path / "1abc", protonation=None)

    monkeypatch.setattr(module, "setup_remote", lambda: None)
    monkeypatch.setattr(module, "fetch_fingerprints", lambda session, ids: {1000: VALID_BITS})
    monkeypatch.setattr(module, "build_target_package", lambda *args, **kwargs: target_json)
    monkeypatch.setattr(
        module,
        "parse_args",
        # `kinase_group_other_than=[]` — умолчание ключа, заведённого 17.09 для
        # отбора второй мишени (пункт `C.2`) и ставшего повторяемым 18.09, когда
        # понадобилось отсеять сразу две группы. Здесь он пуст: проверяется поведение
        # обычного отбора, и без поля разбор аргументов упал бы на несуществующем.
        lambda: argparse.Namespace(
            structures=structures_csv,
            targets_dir=tmp_path,
            candidates=50,
            exclude_kinase=[],
            kinase_group_other_than=[],
        ),
    )
    return module


def test_режим_проставляется_той_же_командой(
    script: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called: list[dict[str, Any]] = []

    def fake_annotate(path: Path, *, on_missing_hydrogens: str) -> dict[str, Any]:
        called.append({"path": path, "on_missing_hydrogens": on_missing_hydrogens})
        return {"protonation": "explicit", "protonation_tool": "KLIFS/MOE"}

    monkeypatch.setattr(script, "annotate_protonation", fake_annotate)
    script.main()

    assert len(called) == 1, "сборка мишени обязана сама проставить режим протонирования"
    # Структуру без водородов теряем не мы, а падение: скрипт просит понизить режим.
    assert called[0]["on_missing_hydrogens"] == "implicit"

    out = capsys.readouterr().out
    assert "protonation: explicit" in out
    assert "KLIFS/MOE" in out
    assert "ВНИМАНИЕ" not in out, "протонированной мишени предупреждать не о чем"


def test_белок_без_водородов_не_теряется_но_предупреждает(
    script: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Цена режима названа числом: молча понижать
    # качество отпечатка нельзя — на этом строится вся сверка с эталоном.
    def fake_annotate(path: Path, *, on_missing_hydrogens: str) -> dict[str, Any]:
        return {"protonation": "implicit-prolif", "protonation_tool": None}

    monkeypatch.setattr(script, "annotate_protonation", fake_annotate)
    script.main()

    out = capsys.readouterr().out
    assert "protonation: implicit-prolif" in out
    assert "ВНИМАНИЕ" in out
    assert "0.400" in out and "1.000" in out


def test_исключение_киназы_убирает_её_структуры_из_отбора(
    script: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Вторая мишень берётся той же процедурой: меняется выборка, а не правило отбора."""
    monkeypatch.setattr(
        script,
        "annotate_protonation",
        lambda path, *, on_missing_hydrogens: {
            "protonation": "explicit",
            "protonation_tool": "KLIFS/MOE",
        },
    )
    аргументы = script.parse_args()
    аргументы.exclude_kinase = [make_structure_row()["kinase.klifs_name"]]
    monkeypatch.setattr(script, "parse_args", lambda: аргументы)

    # Единственная структура выборки принадлежит исключённой киназе, поэтому отбор
    # обязан остановиться с причиной, а не выбрать её вопреки ключу.
    with pytest.raises(KlifsDataError, match="После фильтрации не осталось"):
        script.main()

    assert "Исключены киназы" in capsys.readouterr().out
