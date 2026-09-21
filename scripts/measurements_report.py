"""CLI: сводка реестра измерений по мишеням.

    docker compose run --rm dev python scripts/measurements_report.py
    docker compose run --rm dev python scripts/measurements_report.py \
        --measurement pose_optimization
    docker compose run --rm dev python scripts/measurements_report.py \
        --measurement calibration_smoke --md

Реестр (`data/measurements/targets.csv`) хранит по строке на одно измеренное число, поэтому
глазами его читать неудобно, а числа нужны и в отчётах. Здесь он разворачивается
в таблицу «строка на мишень, колонка на величину» — по любому виду измерения и любому
числу мишеней.

Своих чисел скрипт не считает: он только читает реестр и печатает его.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.measurements import (  # noqa: E402
    markdown_table,
    read_measurements,
    select,
)
from kinase_ifp.config import MEASUREMENTS_CSV  # noqa: E402


def величины(измерения: tuple, вид: str) -> list[tuple[str, str]]:
    """Величины данного вида измерения в алфавитном порядке, как колонки таблицы.

    Набор колонок берётся из самих данных, а не из списка в коде: вид измерения,
    появившийся позже, обязан печататься без правки этого скрипта.
    """
    имена = sorted({измерение.metric for измерение in select(измерения, measurement=вид)})
    return [(имя, имя) for имя in имена]


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument(
        "--measurements", type=Path, default=MEASUREMENTS_CSV, help="реестр измерений"
    )
    разбор.add_argument(
        "--measurement",
        default=None,
        help="вид измерения; без ключа печатается опись всех видов с числом строк",
    )
    разбор.add_argument(
        "--target", default=None, help="показать только одну мишень (по умолчанию все)"
    )
    разбор.add_argument(
        "--md",
        action="store_true",
        help="печатать готовую markdown-таблицу, а не выравненные колонки",
    )
    аргументы = разбор.parse_args()

    измерения = read_measurements(аргументы.measurements)
    if not измерения:
        print(f"реестр {аргументы.measurements} пуст или отсутствует", file=sys.stderr)
        return 1

    if аргументы.measurement is None:
        виды: dict[str, set[str]] = {}
        for измерение in измерения:
            виды.setdefault(измерение.measurement, set()).add(измерение.target)
        print(f"реестр {аргументы.measurements}: строк {len(измерения)}")
        for вид in sorted(виды):
            мишени = ", ".join(sorted(виды[вид]))
            строк = len(select(измерения, measurement=вид))
            print(f"  {вид:28s} строк {строк:4d}  мишени: {мишени}")
        return 0

    отобранные = select(измерения, measurement=аргументы.measurement, target=аргументы.target)
    if not отобранные:
        print(
            f"в реестре нет измерений вида {аргументы.measurement!r}"
            + (f" по мишени {аргументы.target!r}" if аргументы.target else ""),
            file=sys.stderr,
        )
        return 1

    таблица = markdown_table(
        отобранные,
        аргументы.measurement,
        величины(отобранные, аргументы.measurement),
        variant_header="Вариант",
        date_header="Посчитано",
    )
    if аргументы.md:
        print("\n".join(таблица))
        return 0

    # Без `--md` таблица печатается без разделительной строки markdown: в терминале
    # она только мешает читать.
    for номер, строка in enumerate(таблица):
        if номер != 1:
            print(строка)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
