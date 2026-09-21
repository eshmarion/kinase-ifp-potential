"""Веса бит отпечатка по частоте взаимодействия в базе KLIFS (вариант потенциала М2).

Зачем модуль нужен. Дискретный скор считает биты штуками: водородная связь
с каталитическим лизином и случайный ароматический контакт с краем кармана весят
одинаково. PADIF (Jasper, Humbeck, Brinkjost, Koch, J Cheminform 2018, 10:15) решает это
взвешиванием по частоте: взаимодействие, которое повторяется у множества эталонных
структур, признаётся более значимым, чем встретившееся однажды.

У нас источник частот лучше, чем у образца: `data/klifs/klifs_ifp.csv` — это 7855
отпечатков киназных комплексов, посчитанных самим KLIFS по тем же правилам, что и наш
эталон. То есть вес бита берётся не из нашего расчёта и не подгоняется под мишень,
а приходит из внешней базы.

**Правила расчёта отпечатка это не меняет.** Раскладка (85, 7), пороги и состав типов
остаются теми же; меняется только то, с каким коэффициентом бит входит
в сумму. Поэтому вариант М2 сравним с базовым бит в бит: при всех весах, равных
единице, его формула переходит в базовую, и это проверяется тестом.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from kinase_ifp.config import (
    BIT_WEIGHT_SMOOTHING_COUNT,
    KLIFS_IFP_CSV,
    KLIFS_IFP_LENGTH,
    KLIFS_IFP_SHAPE,
)
from kinase_ifp.fingerprint import bits_to_ifp


class BitWeightError(RuntimeError):
    """Веса построить нельзя: нет таблицы отпечатков или она испорчена."""


def bit_counts(csv_path: Path = KLIFS_IFP_CSV) -> np.ndarray:
    """Сколько структур базы KLIFS имеют каждый бит; массив формы (7, 85).

    Читает таблицу `structure_id, pdb_id, bits`, которую собирает
    `scripts/fetch_klifs_ifp.py`. Строка короче или длиннее 595 символов — отказ,
    а не пропуск: молчаливый пропуск испорченной строки занизил бы частоты,
    и заметить это по итоговым числам было бы нечем.
    """
    if not csv_path.is_file():
        raise BitWeightError(
            f"нет таблицы отпечатков {csv_path}: собери её "
            "командой python scripts/fetch_klifs_ifp.py"
        )
    сумма = np.zeros(KLIFS_IFP_SHAPE, dtype=float)
    структур = 0
    with csv_path.open(encoding="utf-8", newline="") as поток:
        for номер, запись in enumerate(csv.DictReader(поток), start=2):
            строка = (запись.get("bits") or "").strip()
            if len(строка) != KLIFS_IFP_LENGTH:
                raise BitWeightError(
                    f"{csv_path}, строка {номер}: длина отпечатка {len(строка)}, "
                    f"ожидалось {KLIFS_IFP_LENGTH}"
                )
            сумма += bits_to_ifp(строка)
            структур += 1
    if структур == 0:
        raise BitWeightError(f"{csv_path}: ни одного отпечатка")
    return сумма


def bit_weights(csv_path: Path = KLIFS_IFP_CSV) -> np.ndarray:
    """Веса бит в [0, 1] по частоте в базе KLIFS; массив формы (7, 85).

    Вес = (частота + eps) / (макс. частота + eps), где eps — вклад одной структуры
    (сглаживание Лапласа). Нормировка на максимум, а не на число структур: абсолютные
    частоты малы (самый частый бит встречается у меньшинства структур, потому что
    киназы связывают лиганды по-разному), и без нормировки все веса оказались бы
    близки к нулю, а вместе с ними и значения скора — формально это ничего не меняет,
    но числа стали бы нечитаемыми.
    """
    счёт = bit_counts(csv_path)
    if счёт.max() == 0:
        raise BitWeightError(f"{csv_path}: все биты нулевые, взвешивать нечем")
    eps = BIT_WEIGHT_SMOOTHING_COUNT
    веса: np.ndarray = (счёт + eps) / (счёт.max() + eps)
    return веса


@lru_cache(maxsize=4)
def cached_bit_weights(csv_path: Path = KLIFS_IFP_CSV) -> np.ndarray:
    """То же, что `bit_weights`, но таблица читается один раз за процесс.

    Нужна оптимизации позы: та зовёт потенциал тысячи раз на молекулу, а разбор
    4,7 МБ отпечатков занимает около секунды. Возвращённый массив помечен как
    неизменяемый — иначе один вызывающий мог бы испортить веса всем остальным.
    """
    веса = bit_weights(csv_path)
    веса.flags.writeable = False
    return веса
