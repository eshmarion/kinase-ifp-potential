"""Тесты CLI выгрузки эталонных отпечатков KLIFS.

Проверяется то, ради чего скрипт написан отдельным входом: файл эталонов дописывается,
а не переписывается, и уже выгруженные структуры второй раз в сеть не уходят.

В KLIFS тест не ходит: сессия подменена. Живая выгрузка проверяется запуском скрипта,
а не pytest — тест, зависящий от чужого сервера, падает от недоступности сети.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from kinase_ifp.config import KLIFS_IFP_LENGTH
from kinase_ifp.klifs import KlifsDataError

from .test_klifs import FakeSession, make_interactions, make_structures

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_klifs_ifp.py"


def load_script() -> Any:
    """Загружает `scripts/fetch_klifs_ifp.py` как модуль.

    Каталог `scripts/` намеренно не лежит на `pythonpath` (там тонкие CLI-входы,
    а не библиотека), поэтому модуль загружается по пути файла, а не импортом по имени.
    """
    spec = importlib.util.spec_from_file_location("fetch_klifs_ifp_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    klifs_ids: tuple[int, ...],
    limit: int,
) -> FakeSession:
    """Готовит вход на диске, подменяет сеть и выполняет скрипт. Возвращает сессию."""
    structures_csv = tmp_path / "structures.csv"
    make_structures(*klifs_ids).to_csv(structures_csv, index=False)

    session = FakeSession(make_interactions(*klifs_ids))
    monkeypatch.setattr(module, "setup_remote", lambda: session)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            limit=limit, structures=structures_csv, out=tmp_path / "klifs_ifp.csv"
        ),
    )
    module.main()
    return session


def test_файл_собирается_и_все_строки_валидны(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1, 2), limit=0)

    table = pd.read_csv(tmp_path / "klifs_ifp.csv", dtype={"bits": str})

    assert list(table.columns) == ["structure_id", "pdb_id", "bits"]
    assert len(table) == 2
    for bits in table["bits"]:
        assert len(bits) == KLIFS_IFP_LENGTH
        assert set(bits) <= {"0", "1"}


def test_limit_ограничивает_выборку(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_script()
    run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1, 2, 3), limit=2)

    table = pd.read_csv(tmp_path / "klifs_ifp.csv")

    assert len(table) == 2


def test_повторный_запуск_не_теряет_строк_и_не_ходит_в_сеть(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ради этого выгрузка и разбита на части: «сначала лучшие структуры, остальные
    # позже» обязано быть одной и той же командой, а не двумя процедурами.
    module = load_script()
    run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1,), limit=0)
    session = run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1, 2), limit=0)

    table = pd.read_csv(tmp_path / "klifs_ifp.csv")

    assert list(table["structure_id"]) == [1, 2]
    assert session.interactions.calls == [[2]], "структура 1 уже выгружена, запрашивать её незачем"


def test_структуры_вне_limit_остаются_в_файле(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Прогон с меньшим --limit не имеет права стереть то, что выгружено раньше:
    # перезапись данных запрещена, а выгрузка стоит времени и сети.
    module = load_script()
    run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1, 2, 3), limit=0)
    run(module, monkeypatch, tmp_path=tmp_path, klifs_ids=(1, 2, 3), limit=1)

    table = pd.read_csv(tmp_path / "klifs_ifp.csv")

    assert list(table["structure_id"]) == [1, 2, 3]


def test_печатает_сколько_структур_осталось_без_отпечатка(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Структура без отпечатка не ошибка, но и не пустое место: три числа в отчёте —
    # единственный способ заметить, что база отдала меньше, чем мы просили.
    module = load_script()
    structures_csv = tmp_path / "structures.csv"
    make_structures(1, 2).to_csv(structures_csv, index=False)

    monkeypatch.setattr(module, "setup_remote", lambda: FakeSession(make_interactions(1)))
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            limit=0, structures=structures_csv, out=tmp_path / "klifs_ifp.csv"
        ),
    )
    module.main()

    out = capsys.readouterr().out
    assert "Структур рассмотрено: 2" in out
    assert "С эталонным отпечатком: 1" in out
    assert "KLIFS не отдал отпечаток: 1" in out


def test_повреждённый_файл_останавливает_скрипт(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    out_path = tmp_path / "klifs_ifp.csv"
    pd.DataFrame([{"structure_id": 1, "pdb_id": "p1", "bits": "0101"}]).to_csv(
        out_path, index=False
    )
    structures_csv = tmp_path / "structures.csv"
    make_structures(1).to_csv(structures_csv, index=False)

    monkeypatch.setattr(module, "setup_remote", lambda: FakeSession(make_interactions(1)))
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(limit=0, structures=structures_csv, out=out_path),
    )

    with pytest.raises(KlifsDataError, match="595"):
        module.main()
