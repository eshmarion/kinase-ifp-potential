"""Тесты частотного анализа позиций шарнира.

Числа здесь заданы вручную, а не взяты из выгрузки KLIFS: доли вида «два из трёх»
проверяемы глазом, а на реальных 7855 отпечатках любая ошибка индексации выглядела бы
как правдоподобный результат — ровно так и жил черновик.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinase_ifp.config import (
    DATA_DIR,
    HINGE_POSITIONS,
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    N_KLIFS_POSITIONS,
)
from kinase_ifp.fingerprint import ifp_to_bits
from kinase_ifp.hinge import (
    compare_with_hinge,
    fingerprints_from_bits,
    frequency_ranks,
    hbond_frequencies,
    hinge_in_top,
    hinge_verdict,
    render_report,
)


def отпечаток(*биты: tuple[str, int]) -> np.ndarray:
    """Пустой отпечаток (7, 85) с проставленными парами «тип, позиция 1..85»."""
    массив = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    for тип, позиция in биты:
        массив[KLIFS_INTERACTION_TYPES.index(тип), позиция - 1] = 1
    return массив


def test_частота_считается_по_позициям_а_не_по_типам() -> None:
    """Связь на позиции 46 обязана дать 1.0 именно на 46-й, а не на 46-м элементе строки."""
    частоты, комплексов = hbond_frequencies([отпечаток(("DON", 46))])

    assert комплексов == 1
    assert частоты.shape == (N_KLIFS_POSITIONS,)
    assert частоты[45] == pytest.approx(1.0)
    assert частоты.sum() == pytest.approx(1.0)


def test_оба_типа_водородной_связи_засчитываются() -> None:
    """`DON` и `ACC` — одна и та же связь с разных сторон; различать их метрика не должна."""
    частоты, _ = hbond_frequencies([отпечаток(("DON", 17)), отпечаток(("ACC", 17))])

    assert частоты[16] == pytest.approx(1.0)


def test_прочие_типы_взаимодействий_в_частоту_не_попадают() -> None:
    """Ловит перепутанный номер типа: HYD и ароматика водородными связями не являются."""
    частоты, _ = hbond_frequencies(
        [отпечаток(("HYD", 46), ("F-F", 47), ("F-E", 48), ("ION+", 17), ("ION-", 24))]
    )

    assert частоты.sum() == pytest.approx(0.0)


def test_частота_это_доля_комплексов() -> None:
    """Два отпечатка из трёх со связью на позиции 46 дают 2/3, а не 2 и не 0.5."""
    частоты, комплексов = hbond_frequencies(
        [отпечаток(("DON", 46)), отпечаток(("ACC", 46)), отпечаток(("DON", 12))]
    )

    assert комплексов == 3
    assert частоты[45] == pytest.approx(2 / 3)
    assert частоты[11] == pytest.approx(1 / 3)


def test_две_связи_на_одной_позиции_считаются_за_один_комплекс() -> None:
    """Иначе доля превысила бы единицу: у одного комплекса могут стоять оба типа сразу."""
    частоты, _ = hbond_frequencies([отпечаток(("DON", 46), ("ACC", 46))])

    assert частоты[45] == pytest.approx(1.0)


def test_пустой_набор_не_даёт_молчаливых_нулей() -> None:
    """Массив нулей неотличим от честного «связей нет», поэтому здесь ошибка."""
    with pytest.raises(ValueError, match="пуст"):
        hbond_frequencies([])


def test_срез_по_типам_на_вход_не_принимается() -> None:
    """На срезе номера `DON` и `ACC` указали бы на другие типы, и арифметика не заметит."""
    срез = np.zeros((2, N_KLIFS_POSITIONS), dtype=np.uint8)

    with pytest.raises(ValueError, match="полный отпечаток"):
        hbond_frequencies([срез])


def test_ранг_растёт_от_частой_позиции_к_редкой() -> None:
    частоты = np.zeros(N_KLIFS_POSITIONS)
    частоты[0] = 0.9
    частоты[1] = 0.5

    ранги = frequency_ranks(частоты)

    assert ранги[0] == 1
    assert ранги[1] == 2
    assert ранги[2] == 3  # все нулевые позиции делят третье место


def test_равные_частоты_получают_равный_ранг() -> None:
    """Соревновательное ранжирование: иначе порядок задавал бы номер позиции."""
    частоты = np.zeros(N_KLIFS_POSITIONS)
    частоты[0] = 0.5
    частоты[1] = 0.5
    частоты[2] = 0.9

    ранги = frequency_ranks(частоты)

    assert ранги[2] == 1
    assert ранги[0] == ранги[1] == 2


def test_частоты_вне_диапазона_отвергаются() -> None:
    with pytest.raises(ValueError, match="от 0 до 1"):
        frequency_ranks(np.full(N_KLIFS_POSITIONS, 1.5))
    with pytest.raises(ValueError, match="формы"):
        frequency_ranks(np.zeros(7))


def test_сводка_собирает_шарнир_ключевые_позиции_и_топ() -> None:
    частоты = np.zeros(N_KLIFS_POSITIONS)
    for позиция in HINGE_POSITIONS:
        частоты[позиция - 1] = 0.8
    частоты[16] = 0.9  # каталитический лизин

    сводка = compare_with_hinge(частоты, n_complexes=100, top_n=4)

    assert сводка.n_complexes == 100
    assert [стат.position for стат in сводка.hinge] == list(HINGE_POSITIONS)
    assert сводка.top[0].position == 17
    assert [стат.position for стат in сводка.top[1:]] == list(HINGE_POSITIONS)
    assert dict(сводка.key)["catalytic_lys"].frequency == pytest.approx(0.9)


def test_вывод_различает_три_исхода() -> None:
    """Промежуточный исход обязан существовать: на реальной выгрузке получается именно он."""
    весь_шарнир = np.zeros(N_KLIFS_POSITIONS)
    for позиция in HINGE_POSITIONS:
        весь_шарнир[позиция - 1] = 0.4
    assert hinge_verdict(compare_with_hinge(весь_шарнир, n_complexes=10)) == "сходятся"

    часть = весь_шарнир.copy()
    часть[HINGE_POSITIONS[1] - 1] = 0.0
    # Восемь посторонних позиций частотнее — в топ-10 остаётся место ровно для двух
    # уцелевших позиций шарнира, а обнулённая туда не попадает.
    for позиция in range(60, 68):
        часть[позиция] = 0.9
    assert hinge_verdict(compare_with_hinge(часть, n_complexes=10)) == "сходятся частично"

    без_шарнира = np.zeros(N_KLIFS_POSITIONS)
    for позиция in range(60, 75):
        без_шарнира[позиция] = 0.9
    assert hinge_verdict(compare_with_hinge(без_шарнира, n_complexes=10)) == "расходятся"


def test_медиана_как_критерий_не_годится() -> None:
    """Фиксирует, почему порог — вхождение в топ: на реальном фоне медиана равна нулю.

    35 позиций из 85 не дают водородных связей ни разу, медиана выгрузки — 0.0004,
    и её проходит любая позиция с единственным битом. Тест держит замену критерия,
    чтобы «сравнить со средним» не вернулось как очевидное упрощение.
    """
    фон = np.zeros(N_KLIFS_POSITIONS)
    фон[46] = 0.009  # позиция 47: одна связь на сотню комплексов

    assert фон[46] > float(np.median(фон))
    assert hinge_verdict(compare_with_hinge(фон, n_complexes=7855)) == "сходятся частично"


def test_в_топ_попадают_только_позиции_шарнира_из_верхних_мест() -> None:
    частоты = np.zeros(N_KLIFS_POSITIONS)
    частоты[HINGE_POSITIONS[0] - 1] = 0.9
    частоты[HINGE_POSITIONS[1] - 1] = 0.5

    попали = hinge_in_top(compare_with_hinge(частоты, n_complexes=10, top_n=2))

    assert [стат.position for стат in попали] == list(HINGE_POSITIONS[:2])


def test_отчёт_содержит_все_85_позиций_и_условия_расчёта() -> None:
    сводка = compare_with_hinge(np.zeros(N_KLIFS_POSITIONS), n_complexes=42)

    текст = render_report(сводка, source="data/klifs/klifs_ifp.csv", filters="тип I")

    assert "42" in текст
    assert "data/klifs/klifs_ifp.csv" in текст
    assert "тип I" in текст
    for позиция in range(1, N_KLIFS_POSITIONS + 1):
        assert f"| {позиция} | " in текст


def test_битовая_строка_klifs_разбирается_генератором() -> None:
    """Скрипт кормит анализ строками из CSV — путь «строка → отпечаток» должен работать."""
    строка = ifp_to_bits(отпечаток(("DON", 46)))

    частоты, комплексов = hbond_frequencies(fingerprints_from_bits([строка]))

    assert комплексов == 1
    assert частоты[45] == pytest.approx(1.0)


@pytest.mark.skipif(
    not (DATA_DIR / "klifs" / "klifs_ifp.csv").is_file(),
    reason=(
        "klifs_ifp.csv собирает fetch_klifs_ifp.py и в клоне его может не быть: "
        "data/ коммитится, но выгрузка сделана локально и файл "
        "приедет её коммитом"
    ),
)
def test_на_реальной_выгрузке_результат_осмыслен() -> None:
    """Сквозная проверка на настоящих эталонах: форма, диапазон и непустой результат."""
    from kinase_ifp.klifs import read_fingerprint_table

    таблица = read_fingerprint_table(Path(DATA_DIR / "klifs" / "klifs_ifp.csv"))
    частоты, комплексов = hbond_frequencies(fingerprints_from_bits(таблица["bits"]))

    assert комплексов == len(таблица)
    assert частоты.shape == (N_KLIFS_POSITIONS,)
    assert ((частоты >= 0.0) & (частоты <= 1.0)).all()
    assert частоты.max() > 0.0
