"""Тесты IFP-скора и ранжирования."""

from __future__ import annotations

import numpy as np
import pytest

from kinase_ifp.config import KLIFS_IFP_SHAPE, KLIFS_INTERACTION_TYPES
from kinase_ifp.scoring import (
    ScoredMolecule,
    ScoringError,
    ifp_score,
    rank_molecules,
    score_norm,
)

_HYD = KLIFS_INTERACTION_TYPES.index("HYD")
_DON = KLIFS_INTERACTION_TYPES.index("DON")
_ACC = KLIFS_INTERACTION_TYPES.index("ACC")


def _ifp(**биты: list[int]) -> np.ndarray:
    """Собирает отпечаток: имя типа → номера позиций (1..85)."""
    массив = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    for тип, позиции in биты.items():
        строка = KLIFS_INTERACTION_TYPES.index(тип.replace("_", "-"))
        for позиция in позиции:
            массив[строка, позиция - 1] = 1
    return массив


def test_полное_совпадение_даёт_единицу() -> None:
    эталон = _ifp(DON=[17, 81], ACC=[45])

    assert ifp_score(эталон, эталон) == pytest.approx(1.0)


def test_половина_взаимодействий_даёт_половину() -> None:
    эталон = _ifp(DON=[17, 81])
    молекула = _ifp(DON=[17])

    assert ifp_score(молекула, эталон) == pytest.approx(0.5)


def test_гидрофобные_в_скор_не_входят() -> None:
    """Скор считается по SCORING_INTERACTION_TYPES: HYD заглушил бы шесть специфичных типов."""
    эталон = _ifp(DON=[17], HYD=[3, 4, 11, 15, 36])
    молекула = _ifp(DON=[17])

    # Все гидрофобные эталона пропущены, но на скор это не влияет.
    assert ifp_score(молекула, эталон) == pytest.approx(1.0)


def test_лишние_взаимодействия_не_штрафуются() -> None:
    """alpha=1, beta=0: контакт, которого нет у эталона, сам по себе не дефект."""
    эталон = _ifp(DON=[17])
    молекула = _ifp(DON=[17], ACC=[45, 46, 47])

    assert ifp_score(молекула, эталон) == pytest.approx(1.0)


def test_пустой_эталон_даёт_ноль() -> None:
    assert ifp_score(_ifp(DON=[17]), _ifp(HYD=[3])) == pytest.approx(0.0)


def test_порядок_аргументов_важен() -> None:
    """Перепутанный порядок даёт другое число — это ловушка, а не эквивалентная запись."""
    эталон = _ifp(DON=[17, 81])
    молекула = _ifp(DON=[17])

    assert ifp_score(молекула, эталон) != pytest.approx(ifp_score(эталон, молекула))


def test_нормировка_делит_на_тяжёлые_атомы() -> None:
    assert score_norm(0.8, 20) == pytest.approx(0.04)


def test_нормировка_на_ноль_атомов_отвергается() -> None:
    with pytest.raises(ScoringError, match="тяжёлых атомов"):
        score_norm(0.8, 0)


def test_ранг_один_у_лучшего_скора() -> None:
    строки = rank_molecules(
        [
            ScoredMolecule("a", 0.2, 0.01),
            ScoredMolecule("b", 0.9, 0.05),
            ScoredMolecule("c", 0.5, 0.03),
        ],
        top_fraction=1.0,
    )

    ранги = {строка["mol_id"]: строка["rank"] for строка in строки}
    assert ранги == {"b": 1, "c": 2, "a": 3}


def test_топ_округляется_вверх() -> None:
    """На наборе из 10 молекул топ в 20 % — это 2 молекулы, а не 1.9 и не 0."""
    строки = rank_molecules(
        [ScoredMolecule(f"m{i}", i / 10, i / 100) for i in range(10)], top_fraction=0.2
    )

    assert sum(int(строка["selected"]) for строка in строки) == 2


def test_ничьи_сохраняют_порядок_появления() -> None:
    строки = rank_molecules(
        [ScoredMolecule("первая", 0.5, 0.02), ScoredMolecule("вторая", 0.5, 0.02)],
        top_fraction=1.0,
    )

    assert [строка["rank"] for строка in строки] == [1, 2]


def test_доля_топа_вне_диапазона_отвергается() -> None:
    with pytest.raises(ScoringError, match="Доля топа"):
        rank_molecules([ScoredMolecule("a", 1.0, 0.05)], top_fraction=0.0)


def test_пустой_набор_отвергается() -> None:
    with pytest.raises(ScoringError, match="ранжировать нечего"):
        rank_molecules([])
