"""Нарезка расчёта на куски не меняет ответ.

Расчёт не проходил одной командой: контейнер вытесняли дважды,
третья попытка упёрлась в часовой предел. Лечится нарезкой, но нарезка имеет смысл
только при одном условии — сумма по кускам обязана совпадать с цельным расчётом
**точно**, а не приблизительно. Здесь это условие и проверяется.

Сети тесты не трогают: проверяется сложение разбивки, а не загрузка структур.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from kinase_ifp.apolar_rule import StructureRuleScore, aggregate

КОРЕНЬ = Path(__file__).resolve().parents[1]


def _модуль():
    """Импортирует точку входа Modal, не требуя установленного клиента.

    Файл — скрипт, а не пакет, поэтому берётся по пути. Если `modal` в окружении
    нет, тест пропускается: проверка про арифметику, а не про облако.
    """
    pytest.importorskip("modal")
    путь = КОРЕНЬ / "scripts" / "modal_apolar.py"
    spec = importlib.util.spec_from_file_location("modal_apolar_под_тестом", путь)
    assert spec is not None and spec.loader is not None
    модуль = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = модуль
    spec.loader.exec_module(модуль)
    return модуль


ПРАВИЛА = ("любой углерод", "углерод, сера, галогены")


def _разбивка() -> list[StructureRuleScore]:
    """Двенадцать структур, у каждой обе оценки правил, числа заведомо разные."""
    записи: list[StructureRuleScore] = []
    for номер in range(12):
        for сдвиг, правило in enumerate(ПРАВИЛА):
            записи.append(
                StructureRuleScore(
                    label=f"s{номер}",
                    rule=правило,
                    reproduced=номер + сдвиг,
                    extra=номер % 3,
                    reference_total=номер + 2,
                    positions=85,
                )
            )
    return записи


def test_сумма_по_кускам_совпадает_с_цельным_расчётом() -> None:
    """Главное утверждение: нарезка точна, а не приблизительна."""
    целиком = aggregate(_разбивка(), ПРАВИЛА)

    записи = _разбивка()
    частями: list[StructureRuleScore] = []
    for начало in range(0, len(записи), 5):
        частями += записи[начало : начало + 5]
    по_кускам = aggregate(частями, ПРАВИЛА)

    assert [(и.rule, и.reproduced, и.extra, и.reference_total) for и in целиком] == [
        (и.rule, и.reproduced, и.extra, и.reference_total) for и in по_кускам
    ]


def test_порядок_кусков_на_сводку_не_влияет() -> None:
    """Modal возвращает куски в порядке готовности, а не запуска."""
    записи = _разбивка()
    прямо = aggregate(записи, ПРАВИЛА)
    наоборот = aggregate(list(reversed(записи)), ПРАВИЛА)
    assert [(и.rule, и.recall, и.precision) for и in прямо] == [
        (и.rule, и.recall, и.precision) for и in наоборот
    ]


def test_разбивка_режется_без_потерь_и_без_повторов() -> None:
    модуль = _модуль()
    выборка = list(range(1, 98))
    части = модуль.куски(выборка, 40)

    assert [len(ч) for ч in части] == [40, 40, 17]
    склеенные = [и for часть in части for и in часть]
    assert склеенные == выборка


def test_кусок_меньше_единицы_отвергается() -> None:
    """Нулевой размер дал бы бесконечный список пустых кусков и молчаливое зависание."""
    модуль = _модуль()
    with pytest.raises(ValueError, match="положительным"):
        модуль.куски([1, 2, 3], 0)


def test_выборка_короче_куска_даёт_один_кусок() -> None:
    модуль = _модуль()
    assert модуль.куски([1, 2, 3], 40) == [[1, 2, 3]]


def test_пустая_выборка_не_даёт_ни_одного_куска() -> None:
    модуль = _модуль()
    assert модуль.куски([], 40) == []


def test_запись_ведёт_переводом_строки(tmp_path: Path) -> None:
    """Решение №28: CSV в этом проекте пишется с `\\n`, иначе diff врёт во всех строках."""
    модуль = _модуль()
    путь = tmp_path / "сводка.csv"
    модуль._записать(путь, ["a", "b"], [[1, 2], [3, 4]])
    assert b"\r\n" not in путь.read_bytes()
    assert путь.read_text(encoding="utf-8").splitlines()[0] == "a,b"
