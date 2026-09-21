"""Тест CLI целевых метрик: печать обязана совпадать с таблицей.

Медиана `ifp_tanimoto` набора положений однажды была записана дважды и по-разному:
0.471 в выводе этой команды и 0.4575 в `results/before_after.csv`.
Величина одна, расходился способ
взять медиану: печать брала элемент по индексу `len // 2`, то есть верхнюю медиану,
а таблица берёт медиану с линейной интерполяцией.

Числа здесь не пересчитываются: прогоны не трогаются вовсе, читается
только уже записанный `metrics_per_molecule.csv`.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

КОРЕНЬ = Path(__file__).resolve().parents[1]
SCRIPT_PATH = КОРЕНЬ / "scripts" / "compute_metrics.py"
RUNS_DIR = КОРЕНЬ / "runs"


def load_script() -> Any:
    """Загружает `scripts/compute_metrics.py` как модуль.

    Каталог `scripts/` намеренно не лежит на `pythonpath` (там тонкие CLI-входы,
    а не библиотека), поэтому модуль загружается по пути файла.
    """
    spec = importlib.util.spec_from_file_location("compute_metrics_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def прогоны_с_метриками() -> list[Path]:
    """Папки прогонов, у которых есть посчитанная таблица метрик."""
    if not RUNS_DIR.is_dir():
        return []
    return sorted(
        папка
        for папка in RUNS_DIR.iterdir()
        if (папка / "metrics_per_molecule.csv").is_file()
    )


def читать_танимото(прогон: Path) -> list[float]:
    with (прогон / "metrics_per_molecule.csv").open(encoding="utf-8") as файл:
        return [
            float(строка["ifp_tanimoto"])
            for строка in csv.DictReader(файл)
            if строка["ifp_tanimoto"]
        ]


def test_сводка_отвергает_пустую_колонку() -> None:
    модуль = load_script()

    with pytest.raises(ValueError, match="Пустая колонка"):
        модуль.сводка_танимото([])


def test_медиана_печати_совпадает_с_медианой_таблицы() -> None:
    """Печать CLI и таблица обязаны давать одно число.

    `report.py` берёт медиану через `pandas.Series.median()`; сверяемся с ней самой,
    а не с её значением, записанным в документ, — документ источником не считается
.
    """
    модуль = load_script()
    прогоны = прогоны_с_метриками()
    assert прогоны, "В runs/ нет ни одного прогона с metrics_per_molecule.csv"

    расхождения = []
    for прогон in прогоны:
        значения = читать_танимото(прогон)
        if not значения:
            continue
        медиана, минимум, максимум = модуль.сводка_танимото(значения)
        ожидается = float(pd.Series(значения).median())
        if медиана != pytest.approx(ожидается):
            расхождения.append(f"{прогон.name}: печать {медиана}, таблица {ожидается}")
        assert минимум == min(значения)
        assert максимум == max(значения)

    assert not расхождения, "Печать CLI разошлась с таблицей: " + "; ".join(расхождения)


def test_верхняя_медиана_отличается_и_потому_запрещена() -> None:
    """Знаковый тест: способ, которым медиана бралась раньше, даёт другое число.

    Без него правка выглядела бы косметической. Расхождение измерено на прогоне
    набора поз: 0.4706 против 0.4575.
    """
    модуль = load_script()
    значения = читать_танимото(RUNS_DIR / "2026-08-24-6tgu-s0-n100")
    упорядоченные = sorted(значения)

    верхняя = упорядоченные[len(упорядоченные) // 2]
    медиана, _, _ = модуль.сводка_танимото(значения)

    assert верхняя == pytest.approx(0.4706, abs=1e-4)
    assert медиана == pytest.approx(0.4575, abs=1e-4)
    assert верхняя != pytest.approx(медиана)
