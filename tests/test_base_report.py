"""Измерения по базе KLIFS ложатся в файл, на который может сослаться текст.

Смысл проверок один: число из этого измерения обязано существовать **в файле**,
а не в печати на экран. Иначе `scripts/check_numbers.py` не найдёт его в источниках,
оно останется неподтверждённым при сверке — независимо от того,
что измерение выполнено правильно.
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.base_report import ЗАГОЛОВОК, КОМАНДА, записать, строки
from experiments.layout import KLIFS_BASE_CSV, KLIFS_BASE_MD
from experiments.results_io import Строка, отчёт_markdown
from kinase_ifp.base_stats import DatasetComposition, Distribution, ReferenceBits

РАЗРЕШЕНИЕ = Distribution(
    name="resolution",
    count=10,
    minimum=0.83,
    q1=1.9,
    median=2.2,
    q3=2.6,
    maximum=35.0,
    threshold=3.0,
    passing=9,
)
КАЧЕСТВО = Distribution(
    name="quality",
    count=10,
    minimum=0.0,
    q1=7.6,
    median=8.0,
    q3=8.0,
    maximum=9.9,
    threshold=6.0,
    passing=8,
)
СОСТАВ = DatasetComposition(
    downloaded=14068,
    selected=7465,
    kinases=227,
    fingerprints=7855,
    with_ligand=8450,
    nucleotide=1003,
    resolution=РАЗРЕШЕНИЕ,
    quality=КАЧЕСТВО,
)
ЭТАЛОНЫ = ReferenceBits(
    structures=7855,
    directed_median=4.0,
    directed_mean=3.8253,
    directed_minimum=0,
    directed_maximum=13,
    few_directed=6670,
    bits_by_type=(("HYD", 112866), ("DON", 12890)),
    positions_median=15.0,
    directed_positions_median=3.0,
)


def _значения(собранные: list[Строка]) -> dict[str, float]:
    return {с.показатель: с.значение for с in собранные}


def test_ключевые_числа_подачи_попадают_в_отчёт() -> None:
    """Те самые величины, ради которых измерение затевалось (`B.2` и `B.4`)."""
    значения = _значения(строки(СОСТАВ, ЭТАЛОНЫ))
    assert значения["выгружено структур"] == 14068
    assert значения["прошло фильтры отбора"] == 7465
    assert значения["направленных бит: медиана"] == 4.0
    assert значения["позиций кармана в контакте: медиана"] == 15.0
    assert значения["из них с направленным контактом: медиана"] == 3.0


def test_производные_доли_берутся_у_структуры_а_не_считаются_заново() -> None:
    """Своей арифметики в отчёте нет: иначе доля разошлась бы с печатью."""
    значения = _значения(строки(СОСТАВ, ЭТАЛОНЫ))
    assert значения["доля нуклеотидных"] == СОСТАВ.nucleotide_share
    assert значения["их доля"] == ЭТАЛОНЫ.few_directed_share
    assert значения["доля гидрофобных бит"] == ЭТАЛОНЫ.hydrophobic_share
    assert значения["бит всего"] == ЭТАЛОНЫ.total_bits


def test_порог_отбора_записан_рядом_с_квартилями() -> None:
    """Медиана разрешения ничего не говорит, пока не сказано, каким был порог."""
    значения = _значения(строки(СОСТАВ, ЭТАЛОНЫ))
    assert значения["разрешение: медиана"] == 2.2
    assert значения["разрешение: порог отбора"] == 3.0
    assert значения["разрешение: доля прошедших"] == РАЗРЕШЕНИЕ.passing_share


def test_каждый_тип_взаимодействия_получает_строку() -> None:
    значения = _значения(строки(СОСТАВ, ЭТАЛОНЫ))
    assert значения["бит типа HYD"] == 112866
    assert значения["бит типа DON"] == 12890


def test_целое_печатается_целым() -> None:
    """«14068 структур», а не «14068.0»: иначе сверка чисел ищет не то значение."""
    текст = отчёт_markdown(строки(СОСТАВ, ЭТАЛОНЫ), ЗАГОЛОВОК, КОМАНДА)
    assert "| выгружено структур | 14068 |" in текст
    assert "14068.0" not in текст


def test_оба_файла_пишутся_и_читаются_машинно(tmp_path: Path) -> None:
    собранные = строки(СОСТАВ, ЭТАЛОНЫ)
    путь_md, путь_csv = записать(собранные, tmp_path)

    assert путь_md.name == KLIFS_BASE_MD
    assert путь_csv.name == KLIFS_BASE_CSV

    with путь_csv.open(encoding="utf-8", newline="") as файл:
        прочитанные = list(csv.DictReader(файл))
    assert len(прочитанные) == len(собранные)
    assert прочитанные[0]["показатель"] == "выгружено структур"
    assert прочитанные[0]["значение"] == "14068"


def test_csv_пишется_с_переводом_строки_а_не_с_возвратом_каретки(tmp_path: Path) -> None:
    """Решение №28: иначе файл, записанный кодом, отличался бы от сохранённого git."""
    _, путь_csv = записать(строки(СОСТАВ, ЭТАЛОНЫ), tmp_path)
    assert b"\r\n" not in путь_csv.read_bytes()
