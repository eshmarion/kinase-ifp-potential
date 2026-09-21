"""Критерии значимости в строки реестра: статистика оптимизации позы.

Зачем модуль нужен. `scripts/pose_optimization_stats.py` печатал свои числа в консоль
и не сохранял ничего. Числа верные, но машинного места у них нет, поэтому
сверка чисел их не подтверждает, и раздел,
написанный по таким числам 18.09, пришлось откатить
. Здесь уже посчитанные величины
раскладываются по строкам реестра измерений.

Разделение труда на 18.09: доли PoseBusters (вид `physics`) и сводки докинга
(вид `docking`) пишет соседняя сессия, здесь их нет намеренно — две реализации одного
вида измерения в одном файле разошлись бы значениями и ключами.

Отдельный вид, а не колонки внутри `pose_optimization`: у сводки прогона и у критериев
значимости разный смысл строки. Сводка описывает выборку («средний скор стал 0.813»),
критерий описывает сравнение двух состояний одной выборки («прирост значим, p = 2·10⁻¹⁷»),
и путать их в одном ключе нельзя — при пересчёте сводки критерий остался бы от прежнего
расчёта и выглядел бы свежим.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from experiments.measurements import Measurement

# Виды измерений, которые заводит этот модуль.
OPTIMIZATION_STATS: Final[str] = "pose_optimization_stats"

# Величины, которые обязана нести строка критерия значимости. Проверяются на месте:
# уровень значимости без размера эффекта и без числа пар — это число, по которому
# нельзя сказать, что именно сравнивалось, а такие числа в текст и не пускают.
REQUIRED_STAT_METRICS: Final[tuple[str, ...]] = ("effect", "p_value", "pairs")


class SummaryError(RuntimeError):
    """Сводку нельзя разложить по строкам реестра: не хватает обязательных величин."""


def _строки(
    target: str,
    measurement: str,
    variant: str,
    stats: Mapping[str, float],
    *,
    units: Mapping[str, str] | None = None,
    measured_utc: str = "",
    source: str = "",
    notes: str = "",
) -> list[Measurement]:
    единицы = units or {}
    return [
        Measurement(
            target=target,
            measurement=measurement,
            variant=variant,
            metric=имя,
            # Формат `:.6g` держит и уровни значимости (2.31e-17), и доли (0.908),
            # не переводя первые в нули, а вторые в экспоненту.
            value=f"{значение:.6g}",
            unit=единицы.get(имя, ""),
            measured_utc=measured_utc,
            source=source,
            notes=notes,
        )
        for имя, значение in stats.items()
    ]


def _проверить_обязательные(stats: Mapping[str, float], где: str) -> None:
    нет = [имя for имя in REQUIRED_STAT_METRICS if имя not in stats]
    if нет:
        raise SummaryError(
            f"{где}: в сводке нет обязательных величин {нет}. "
            f"Уровень значимости без размера эффекта и числа пар в текст не идёт"
        )


def optimization_stat_measurements(
    target: str,
    variant: str,
    stats: Mapping[str, float],
    *,
    measured_utc: str = "",
    source: str = "",
    notes: str = "",
) -> list[Measurement]:
    """Критерии значимости оптимизации позы строками реестра.

    Принимает мишень, ключ условия (`<прогон>:<метка>`, см. `measurements.variant_from_poses`)
    и словарь «величина → число»: обязательно `effect` (размер эффекта), `p_value`
    и `pairs` (число ненулевых пар), далее любые — границы интервала, доли, счётчики.
    Возвращает строки для `upsert_measurements`.

    Уровни значимости пишутся как есть, без округления до «< 0.001»: порог сравнения
    выбирает читатель, а округлённое значение обратно не восстановить.
    """
    _проверить_обязательные(stats, f"{target} {variant}")
    return _строки(
        target,
        OPTIMIZATION_STATS,
        variant,
        stats,
        measured_utc=measured_utc,
        source=source,
        notes=notes,
    )


