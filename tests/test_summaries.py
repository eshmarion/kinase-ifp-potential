"""Критерии значимости в реестре: обязательные величины и формат значений."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.measurements import read_measurements, upsert_measurements
from experiments.summaries import (
    OPTIMIZATION_STATS,
    SummaryError,
    optimization_stat_measurements,
)

СВОДКА = {"effect": 0.1016, "p_value": 2.31e-17, "pairs": 93.0}


def test_сводка_раскладывается_по_строкам() -> None:
    строки = optimization_stat_measurements(
        "6tgu", "2026-09-16-6tgu-s0-n100:all7:paired_score", {**СВОДКА, "ci_low": 0.0909}
    )

    величины = {и.metric: и.value for и in строки}
    assert величины["effect"] == "0.1016"
    assert величины["pairs"] == "93"
    assert {и.measurement for и in строки} == {OPTIMIZATION_STATS}
    assert {и.variant for и in строки} == {"2026-09-16-6tgu-s0-n100:all7:paired_score"}


def test_уровень_значимости_не_округляется_в_ноль() -> None:
    """«p < 0.001» обратно не восстановить, поэтому значение пишется как есть."""
    строки = optimization_stat_measurements("6tgu", "прогон:all7:paired_score", СВОДКА)

    значение = next(и.value for и in строки if и.metric == "p_value")
    assert значение == "2.31e-17"
    assert float(значение) == pytest.approx(2.31e-17)


def test_доля_не_уходит_в_экспоненту() -> None:
    """Доли молекул печатаются как доли: 0.906, а не 9.06e-01."""
    строки = optimization_stat_measurements(
        "6fnk",
        "прогон:all7:mcnemar_hbond",
        {**СВОДКА, "fraction_before": 0.74, "fraction_after": 0.906},
    )

    величины = {и.metric: и.value for и in строки}
    assert величины["fraction_after"] == "0.906"


@pytest.mark.parametrize("пропущена", ["effect", "p_value", "pairs"])
def test_сводка_без_обязательной_величины_отвергается(пропущена: str) -> None:
    """Уровень значимости без размера эффекта и числа пар не говорит, что сравнивалось."""
    урезанная = {имя: значение for имя, значение in СВОДКА.items() if имя != пропущена}

    with pytest.raises(SummaryError, match=пропущена):
        optimization_stat_measurements("6tgu", "прогон:all7:paired_score", урезанная)


def test_разные_тесты_одного_условия_не_вытесняют_друг_друга(tmp_path: Path) -> None:
    """Ключ несёт имя теста: без него `effect` Уилкоксона затёр бы `effect` Макнемара."""
    путь = tmp_path / "targets.csv"
    upsert_measurements(
        optimization_stat_measurements("6fnk", "прогон:all7:paired_score", СВОДКА), путь
    )
    upsert_measurements(
        optimization_stat_measurements(
            "6fnk", "прогон:all7:mcnemar_hbond", {**СВОДКА, "effect": 0.166}
        ),
        путь,
    )

    строки = read_measurements(путь)
    значения = {и.variant: и.value for и in строки if и.metric == "effect"}
    assert значения == {"прогон:all7:paired_score": "0.1016", "прогон:all7:mcnemar_hbond": "0.166"}


def test_повторный_расчёт_заменяет_свои_строки(tmp_path: Path) -> None:
    путь = tmp_path / "targets.csv"
    начальная = optimization_stat_measurements("6tgu", "прогон:all7:paired_score", СВОДКА)
    upsert_measurements(начальная, путь)
    upsert_measurements(
        optimization_stat_measurements(
            "6tgu", "прогон:all7:paired_score", {**СВОДКА, "effect": 0.2}
        ),
        путь,
    )

    строки = read_measurements(путь)
    assert len(строки) == len(СВОДКА)
    assert next(и.value for и in строки if и.metric == "effect") == "0.2"
