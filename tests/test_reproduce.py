"""Тесты конвейера воспроизведения.

Главное здесь — не запуск всего конвейера, а то, что его объявление не врёт:
у каждого этапа есть исполнимый файл команды, метка из закрытого списка,
а обещанные выходы действительно лежат в репозитории. Ошибка в объявлении
проявилась бы иначе только при полном пересчёте, то есть через минуты счёта.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.reproduce import (
    МЕТКИ_СВЕРКИ,
    ЭТАПЫ,
    ReproduceError,
    _сравнить_csv,
    найти,
    список,
)

КОРЕНЬ = Path(__file__).resolve().parent.parent
МЕТКИ = frozenset({"cpu", "figures", "network", "heavy", "gpu"})


def test_имена_этапов_не_повторяются() -> None:
    имена = [э.имя for э in ЭТАПЫ]
    assert len(имена) == len(set(имена)), f"повтор имени этапа: {имена}"


@pytest.mark.parametrize("этап", ЭТАПЫ, ids=lambda э: э.имя)
def test_метка_из_закрытого_списка(этап) -> None:  # noqa: ANN001
    assert этап.метка in МЕТКИ, f"{этап.имя}: метка {этап.метка!r} неизвестна"


@pytest.mark.parametrize("этап", ЭТАПЫ, ids=lambda э: э.имя)
def test_команда_указывает_на_существующий_файл(этап) -> None:  # noqa: ANN001
    скрипт = КОРЕНЬ / этап.команда[0]
    assert скрипт.is_file(), f"{этап.имя}: нет файла {этап.команда[0]}"


@pytest.mark.parametrize("этап", [э for э in ЭТАПЫ if э.метка in МЕТКИ_СВЕРКИ], ids=lambda э: э.имя)
def test_выходы_этапов_сверки_лежат_в_репозитории(этап) -> None:  # noqa: ANN001
    """Этап, который сверяется, обязан иметь с чем сверяться."""
    for относительный in этап.выходы:
        assert (КОРЕНЬ / относительный).is_file(), f"{этап.имя}: нет {относительный}"


def test_в_сверку_входит_хотя_бы_один_этап() -> None:
    сверяемые = [э for э in ЭТАПЫ if э.метка in МЕТКИ_СВЕРКИ and э.выходы]
    assert сверяемые, "сверка не проверяет ничего — её пустой успех ничего не значит"


def test_неизвестный_этап_отвергается() -> None:
    with pytest.raises(ReproduceError, match="нет"):
        найти("такого-этапа-нет")


def test_опись_называет_все_этапы() -> None:
    текст = список()
    for э in ЭТАПЫ:
        assert э.имя in текст


def test_cli_list_отрабатывает() -> None:
    """Опись собирается и печатается.

    `PYTHONIOENCODING` задаётся явно: на Windows дочерний процесс пишет в кодировке
    консоли, и кириллица в выводе приходит нечитаемой, а `errors="replace"` без него
    превратил бы проверку в сравнение с мусором.
    """
    окружение = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    итог = subprocess.run(
        [sys.executable, "scripts/reproduce.py", "--list"],
        cwd=КОРЕНЬ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=окружение,
    )
    assert итог.returncode == 0, итог.stderr
    assert "Этапы конвейера" in итог.stdout


def test_сравнение_csv_видит_расхождение(tmp_path: Path) -> None:
    было = tmp_path / "a.csv"
    стало = tmp_path / "b.csv"
    было.write_text("метрика,значение\nскор,0.5\n", encoding="utf-8")
    стало.write_text("метрика,значение\nскор,0.7\n", encoding="utf-8")
    расхождение = _сравнить_csv(было, стало)
    assert расхождение is not None and "0.5" in расхождение


def test_сравнение_csv_терпит_шум_последней_цифры(tmp_path: Path) -> None:
    """Иначе сверка краснела бы на порядке суммирования, а не на ошибке."""
    было = tmp_path / "a.csv"
    стало = tmp_path / "b.csv"
    было.write_text("метрика,значение\nскор,0.30000000001\n", encoding="utf-8")
    стало.write_text("метрика,значение\nскор,0.3\n", encoding="utf-8")
    assert _сравнить_csv(было, стало) is None
