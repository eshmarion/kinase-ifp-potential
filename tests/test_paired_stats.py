"""Парные критерии: точный биномиальный, Макнемара, знаковый и поправка Холма.

Главное, что проверяется, — что критерий смотрит на несогласованные пары, а не на доли:
выборка, где доля выросла с 0.0 до 1.0 на десяти молекулах, обязана дать то же p,
что десять подряд выпавших орлов, и ни одна согласованная пара на это число не влияет.
"""

from __future__ import annotations

import pytest

from evaluation.paired_stats import exact_binomial_p, holm, mcnemar, sign_test
from evaluation.stats import StatsError


def test_без_испытаний_различия_не_обнаружено() -> None:
    assert exact_binomial_p(0, 0) == 1.0


def test_десять_из_десяти_дают_две_тысячные() -> None:
    assert exact_binomial_p(10, 10) == pytest.approx(2 / 2**10)


def test_ровно_пополам_не_значимо() -> None:
    assert exact_binomial_p(5, 10) == 1.0


def test_макнемар_считает_по_несогласованным_парам() -> None:
    """Пятьдесят согласованных пар не меняют p: значимость дают только изменения."""
    до = [0] * 10 + [1] * 50
    после = [1] * 10 + [1] * 50
    итог = mcnemar(до, после)
    assert (итог.gained, итог.lost) == (10, 0)
    assert итог.p_value == pytest.approx(exact_binomial_p(10, 10))
    assert итог.before == pytest.approx(50 / 60)
    assert итог.after == pytest.approx(1.0)
    assert итог.ci[0] > 0


def test_макнемар_отвергает_недвоичные_значения() -> None:
    with pytest.raises(StatsError, match="нулей и единиц"):
        mcnemar([0, 1], [0, 2])


def test_макнемар_отвергает_ряды_разной_длины() -> None:
    with pytest.raises(StatsError, match="разной длины"):
        mcnemar([0, 1], [1])


def test_знаковый_критерий_отбрасывает_нулевые_разности() -> None:
    плюс, минус, p = sign_test([0.0] * 20 + [0.5] * 8 + [-0.5])
    assert (плюс, минус) == (8, 1)
    assert p == pytest.approx(exact_binomial_p(8, 9))


def test_холм_монотонен_и_сохраняет_порядок_ключей() -> None:
    """Третий тест сам по себе дал бы 0.03, но не может оказаться значимее второго."""
    исправленные = holm({"первый": 0.01, "второй": 0.02, "третий": 0.03})
    assert list(исправленные) == ["первый", "второй", "третий"]
    assert исправленные["первый"] == pytest.approx(0.03)
    assert исправленные["второй"] == pytest.approx(0.04)
    assert исправленные["третий"] == pytest.approx(0.04)


def test_холм_не_ослабляет_единственный_тест() -> None:
    assert holm({"один": 0.02})["один"] == pytest.approx(0.02)
