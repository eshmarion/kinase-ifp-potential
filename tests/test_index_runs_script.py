"""Тесты CLI пересборки реестра прогонов (`scripts/index_runs.py`).

Закрывает последнюю из четырёх веток входа. У входа
один отказ, и до 16.09 он не исполнялся ни разу: тестового файла у скрипта
не существовало. Отказ при этом содержательный — он объясняет человеку в свежем
клоне, откуда прогоны берутся, а не просто сообщает, что каталога нет.

Рабочий `runs/` не трогается: каталог прогонов всегда временный.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

КОРЕНЬ = Path(__file__).resolve().parents[1]
SCRIPT_PATH = КОРЕНЬ / "scripts" / "index_runs.py"


def load_script() -> Any:
    """Загружает вход как модуль: `scripts/` намеренно не лежит на `pythonpath`."""
    spec = importlib.util.spec_from_file_location("index_runs_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def запуск(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    модуль = load_script()

    def вызвать(*аргументы: str) -> None:
        monkeypatch.setattr(sys, "argv", ["index_runs.py", *аргументы])
        модуль.main()

    yield вызвать


def создать_прогон(runs_dir: Path, run_id: str, *, с_ранжированием: bool) -> Path:
    """Папка прогона с молекулами и паспортом; ранжирование делает её `scored`."""
    папка = runs_dir / run_id
    папка.mkdir(parents=True)
    (папка / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source": "local-poses",
                "created_utc": "2026-08-24T09:30:03Z",
                "target": {"pdb_id": "6tgu"},
                "sampling": {"n_requested": 4, "n_returned": 4, "seed": 0},
                "platform": "local-cpu",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (папка / "molecules.sdf").write_text("$$$$\n", encoding="utf-8")
    if с_ранжированием:
        (папка / "ranking.csv").write_text("mol_id,ifp_score\nm1,0.4\n", encoding="utf-8")
    return папка


def test_отсутствующий_каталог_прогонов_отвергается(запуск: Any, tmp_path: Path) -> None:
    """Сообщение объясняет, откуда берутся прогоны: человек в свежем клоне не знает."""
    with pytest.raises(SystemExit, match="pull_run.py"):
        запуск("--runs-dir", str(tmp_path / "нет-такого-каталога"))


def test_реестр_собирается_и_различает_статусы(
    запуск: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Прогон без ранжирования — `generated`, с ним — `scored`."""
    создать_прогон(tmp_path, "2026-08-24-6tgu-s0-n4", с_ранжированием=False)
    создать_прогон(tmp_path, "2026-08-24-6tgu-s0-n4-r1", с_ранжированием=True)

    запуск("--runs-dir", str(tmp_path))

    строки = (tmp_path / "index.csv").read_text(encoding="utf-8").splitlines()
    заголовок = строки[0].split(",")
    статусы = {
        строка.split(",")[заголовок.index("run_id")]: строка.split(",")[заголовок.index("status")]
        for строка in строки[1:]
    }
    assert статусы == {
        "2026-08-24-6tgu-s0-n4": "generated",
        "2026-08-24-6tgu-s0-n4-r1": "scored",
    }
    assert "всего прогонов: 2" in capsys.readouterr().out


def test_испорченная_папка_попадает_в_реестр_как_failed(запуск: Any, tmp_path: Path) -> None:
    """Молча исчезнувшая папка выглядит как «эксперимента не было»."""
    (tmp_path / "без-паспорта").mkdir()

    запуск("--runs-dir", str(tmp_path))

    содержимое = (tmp_path / "index.csv").read_text(encoding="utf-8")
    assert "без-паспорта" in содержимое
    assert "failed" in содержимое
