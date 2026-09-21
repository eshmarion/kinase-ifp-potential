"""Тесты описательной статистики выгрузки KLIFS (пункты B.2 и B.4 плана подачи).

Проверяется не «функция что-то вернула», а те утверждения, ради которых модуль написан:
что знаменатель скора мал у подавляющего большинства структур базы, а не у одной нашей
мишени; что раскладка бит читается правильно; и что отказ наступает там, где данных
нет, а не подменяется нулём.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kinase_ifp.base_stats import (
    FEW_DIRECTED_BITS,
    BaseStatsError,
    _bit_positions,
    dataset_composition,
    reference_bits,
)
from kinase_ifp.config import KLIFS_IFP_LENGTH, SCORING_INTERACTION_TYPES


def test_состав_выгрузки_сужается_фильтрами() -> None:
    состав = dataset_composition()

    assert состав.downloaded == 14068
    assert состав.selected == 7465
    assert состав.kinases == 227
    assert состав.fingerprints == 7855
    assert состав.selected < состав.downloaded


def test_нуклеотидных_комплексов_примерно_восьмая_часть() -> None:
    """Знаменатель — структуры типа I с лигандом."""
    состав = dataset_composition()

    assert состав.with_ligand == 8450
    assert состав.nucleotide == 1003
    assert 0.11 < состав.nucleotide_share < 0.13


def test_пороги_отбора_стоят_рядом_с_квартилями() -> None:
    """Число без порога ничего не говорит, поэтому порог хранится в самом распределении."""
    состав = dataset_composition()

    assert состав.resolution.threshold == 3.0
    assert состав.quality.threshold == 6.0
    assert состав.resolution.passing < состав.resolution.count
    assert состав.resolution.minimum <= состав.resolution.median <= состав.resolution.maximum


def test_знаменатель_скора_мал_у_всей_базы_а_не_у_нашей_мишени() -> None:
    """Главное утверждение B.4: вырожденность — свойство схемы KLIFS для киназ."""
    эталоны = reference_bits()

    assert эталоны.structures == 7855
    assert эталоны.directed_median == 4
    assert эталоны.directed_minimum == 0
    assert эталоны.few_directed_share > 0.8


def test_у_шести_tgu_направленных_бит_не_меньше_медианы_базы() -> None:
    """Ответ на возражение «вам не повезло с мишенью»: пять бит — выше медианы базы."""
    эталоны = reference_bits()

    assert эталоны.directed_median <= FEW_DIRECTED_BITS


def test_гидрофобные_биты_составляют_основную_массу_базы() -> None:
    """То самое основание, по которому HYD исключён из скора."""
    эталоны = reference_bits()

    assert эталоны.hydrophobic_share > 0.75
    assert dict(эталоны.bits_by_type)["HYD"] > эталоны.total_bits / 2


def test_интерфейс_шире_чем_его_направленная_часть() -> None:
    эталоны = reference_bits()

    assert эталоны.positions_median > эталоны.directed_positions_median


def test_биты_по_типам_покрывают_все_семь_типов() -> None:
    эталоны = reference_bits()

    assert len(эталоны.bits_by_type) == 7
    assert sum(количество for _, количество in эталоны.bits_by_type) == эталоны.total_bits


def test_раскладка_строки_бит_позиционная() -> None:
    """Решение №19: позиция = i // 7, тип = i % 7. Первые семь бит — позиция 0."""
    строка = "1" + "0" * (KLIFS_IFP_LENGTH - 1)
    (позиция, тип), = _bit_positions(строка)

    assert позиция == 0
    assert тип == "HYD"

    сдвиг = "0" * 3 + "1" + "0" * (KLIFS_IFP_LENGTH - 4)
    (позиция, тип), = _bit_positions(сдвиг)
    assert позиция == 0
    assert тип in SCORING_INTERACTION_TYPES


def test_строка_не_той_длины_это_отказ() -> None:
    with pytest.raises(BaseStatsError, match="битовая строка длины"):
        _bit_positions("101")


def test_пропавшая_выгрузка_это_отказ_а_не_нули(tmp_path: Path) -> None:
    """Пустая таблица опаснее исключения: доли посчитались бы от нуля и выглядели бы живыми."""
    with pytest.raises(BaseStatsError, match="нет файла выгрузки"):
        dataset_composition(tmp_path)
