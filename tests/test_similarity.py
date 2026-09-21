"""Тесты мер сходства отпечатков.

Массивы здесь синтетические: сходство — арифметика над битами, и подмешивать сюда
ProLIF значило бы проверять две вещи сразу. Связь с настоящим отпечатком проверяется
единственным тестом на эталонной строке 6tgu.
"""

from __future__ import annotations

import numpy as np
import pytest

from kinase_ifp.config import (
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.pocket import Pocket
from kinase_ifp.similarity import select_types, tanimoto, tversky

_HYD = KLIFS_INTERACTION_TYPES.index("HYD")
_DON = KLIFS_INTERACTION_TYPES.index("DON")
_ACC = KLIFS_INTERACTION_TYPES.index("ACC")


def make_ifp(*bits: tuple[int, int]) -> np.ndarray:
    """Собирает отпечаток (7, 85) из пар «строка типа, номер позиции 1..85»."""
    ifp = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    for interaction_type, position in bits:
        ifp[interaction_type, position - 1] = 1
    return ifp


# Эталон и молекула, у которой к тем же двум взаимодействиям добавлено третье.
REFERENCE = make_ifp((_HYD, 1), (_DON, 5))
SUPERSET = make_ifp((_HYD, 1), (_DON, 5), (_ACC, 9))


class TestТанимото:
    def test_отпечаток_похож_на_себя_полностью(self) -> None:
        assert tanimoto(REFERENCE, REFERENCE) == 1.0

    def test_совпадает_с_ручным_счётом(self) -> None:
        """Один общий бит из трёх в объединении: 1/3, а не «функция не упала»."""
        other = make_ifp((_HYD, 1), (_ACC, 7))

        assert tanimoto(REFERENCE, other) == pytest.approx(1 / 3)

    def test_симметричен(self) -> None:
        assert tanimoto(REFERENCE, SUPERSET) == tanimoto(SUPERSET, REFERENCE)

    def test_нет_общих_битов_даёт_ноль(self) -> None:
        assert tanimoto(REFERENCE, make_ifp((_ACC, 40))) == 0.0

    def test_два_нулевых_отпечатка_дают_ноль(self) -> None:
        """Пустое объединение — 0.0, а не 1.0 и не деление на ноль."""
        empty = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)

        assert tanimoto(empty, empty) == 0.0

    def test_один_нулевой_отпечаток_даёт_ноль(self) -> None:
        assert tanimoto(REFERENCE, np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)) == 0.0


class TestTversky:
    def test_доля_битов_эталона(self) -> None:
        """При alpha=1, beta=0 первый аргумент — эталон: лишний бит молекулы не штрафуется."""
        assert tversky(REFERENCE, SUPERSET, 1, 0) == 1.0
        assert tanimoto(REFERENCE, SUPERSET) == pytest.approx(2 / 3)

    def test_несимметричен(self) -> None:
        """Перепутанный порядок аргументов отвечает на другой вопрос, а не ломается."""
        assert tversky(SUPERSET, REFERENCE, 1, 0) == pytest.approx(2 / 3)
        assert tversky(REFERENCE, SUPERSET, 1, 0) != tversky(SUPERSET, REFERENCE, 1, 0)

    def test_воспроизведена_половина_эталона(self) -> None:
        partial = make_ifp((_HYD, 1), (_ACC, 40))

        assert tversky(REFERENCE, partial, 1, 0) == pytest.approx(1 / 2)

    def test_единичные_веса_дают_танимото(self) -> None:
        """Проверка самой формулы: Танимото — частный случай Tversky при alpha=beta=1."""
        first, second = _random_pair()

        assert tversky(first, second, 1, 1) == pytest.approx(tanimoto(first, second))

    def test_половинные_веса_дают_дайса(self) -> None:
        first, second = _random_pair()
        common = np.count_nonzero(first & second)
        dice = 2 * common / (np.count_nonzero(first) + np.count_nonzero(second))

        assert tversky(first, second, 0.5, 0.5) == pytest.approx(dice)

    def test_пустые_отпечатки_дают_ноль(self) -> None:
        empty = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)

        assert tversky(empty, empty) == 0.0

    def test_отрицательные_веса_отвергаются(self) -> None:
        with pytest.raises(ValueError, match="отрицательными"):
            tversky(REFERENCE, SUPERSET, -1, 0)


class TestОтборТипов:
    def test_форма_среза(self) -> None:
        selected = select_types(REFERENCE, SCORING_INTERACTION_TYPES)

        assert selected.shape == (len(SCORING_INTERACTION_TYPES), KLIFS_IFP_SHAPE[1])

    def test_гидрофобный_контакт_исключён_из_скора(self) -> None:
        """Совпадает только HYD: по всем типам сходство есть, по типам скора его нет."""
        molecule = make_ifp((_HYD, 1), (_ACC, 9))

        assert tanimoto(REFERENCE, molecule) == pytest.approx(1 / 3)
        assert (
            tanimoto(
                select_types(REFERENCE, SCORING_INTERACTION_TYPES),
                select_types(molecule, SCORING_INTERACTION_TYPES),
            )
            == 0.0
        )

    def test_порядок_типов_соблюдается(self) -> None:
        selected = select_types(REFERENCE, ("DON", "HYD"))

        assert selected[0].sum() == 1 and selected[0][4] == 1
        assert selected[1][0] == 1

    def test_неизвестный_тип_отвергается(self) -> None:
        with pytest.raises(ValueError, match="Неизвестные типы"):
            select_types(REFERENCE, ("HYD", "ВОДОРОДНАЯ"))

    def test_пустой_список_типов_отвергается(self) -> None:
        with pytest.raises(ValueError, match="пуст"):
            select_types(REFERENCE, ())

    def test_повтор_типа_отвергается(self) -> None:
        """Дубль удваивал вклад типа, и мера объявляла его двумя битами."""
        with pytest.raises(ValueError, match="повторяются"):
            select_types(REFERENCE, ("HYD", "HYD"))

    def test_срезы_разных_типов_сравниваются_без_возражений(self) -> None:
        """Знаковый тест: мера видит форму, но не состав типов.

        Два среза разных типов с битами на одних позициях дают полное сходство,
        хотя общих взаимодействий у них нет. Защиты нет и не будет до правки
        формата мер сходства; тест фиксирует поведение как известное,
        чтобы молчаливым оно не было.
        """
        molecule = make_ifp((_HYD, 3), (_HYD, 7))
        reference = make_ifp((_ACC, 3), (_ACC, 7))

        assert tanimoto(
            select_types(molecule, ("HYD",)), select_types(reference, ("ACC",))
        ) == pytest.approx(1.0)


class TestПроверкаВхода:
    def test_перепутанные_оси_отвергаются(self) -> None:
        """(85, 7) — отменённая решением №19 раскладка: арифметика прошла бы молча."""
        with pytest.raises(ValueError, match="ожидалась форма"):
            tanimoto(REFERENCE.T, REFERENCE.T)

    def test_срез_и_полный_отпечаток_не_сравниваются(self) -> None:
        """Иначе numpy растянул бы срез по правилам broadcasting и вернул неверное число."""
        with pytest.raises(ValueError, match="разной формы"):
            tanimoto(select_types(REFERENCE, ("DON",)), REFERENCE)

    def test_небитовые_значения_отвергаются(self) -> None:
        counts = REFERENCE.copy()
        counts[_HYD, 0] = 3

        with pytest.raises(ValueError, match="значения 0 и 1"):
            tanimoto(counts, REFERENCE)

    def test_отбор_типов_по_срезу_отвергается(self) -> None:
        with pytest.raises(ValueError, match="полному отпечатку"):
            select_types(select_types(REFERENCE, ("DON", "HYD")), ("DON",))


def test_эталонный_отпечаток_klifs_похож_на_себя(pocket: Pocket) -> None:
    """Массив из `bits_to_ifp` годится мерам сходства как есть."""
    assert pocket.reference_ifp is not None

    assert tanimoto(pocket.reference_ifp, pocket.reference_ifp) == 1.0
    assert tversky(pocket.reference_ifp, pocket.reference_ifp) == 1.0


def _random_pair() -> tuple[np.ndarray, np.ndarray]:
    """Два непустых случайных отпечатка с фиксированным зерном — тест воспроизводим."""
    generator = np.random.default_rng(19)
    first = (generator.random(KLIFS_IFP_SHAPE) < 0.05).astype(np.uint8)
    second = (generator.random(KLIFS_IFP_SHAPE) < 0.05).astype(np.uint8)
    assert first.any() and second.any()
    return first, second
