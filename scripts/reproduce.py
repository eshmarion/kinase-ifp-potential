"""CLI конвейера воспроизведения: опись этапов, запуск одного, сверка с репозиторием.

    python scripts/reproduce.py --list
    python scripts/reproduce.py --stage klifs-base
    python scripts/reproduce.py --check

`--check` пересчитывает лёгкие этапы в копии репозитория и сравнивает результат
с тем, что лежит в `results/`. Копия нужна потому, что скрипты пишут по абсолютным
путям, и счёт «на месте» затёр бы то, с чем сверяемся.
"""

from __future__ import annotations

import argparse
import sys

from experiments.reproduce import МЕТКИ_СВЕРКИ, ReproduceError, выполнить, найти, сверить, список


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    группа = parser.add_mutually_exclusive_group(required=True)
    группа.add_argument("--list", action="store_true", help="перечислить этапы")
    группа.add_argument("--stage", metavar="ИМЯ", help="выполнить один этап")
    группа.add_argument("--check", action="store_true", help="пересчитать и сверить")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(МЕТКИ_СВЕРКИ),
        help=f"какие метки брать в сверку (по умолчанию {' '.join(МЕТКИ_СВЕРКИ)})",
    )
    аргументы = parser.parse_args()

    if аргументы.list:
        print(список())
        return 0

    try:
        if аргументы.stage:
            выполнить(найти(аргументы.stage))
            print(f"этап {аргументы.stage!r} выполнен")
            return 0
        расхождения = сверить(аргументы.labels)
    except ReproduceError as ошибка:
        print(f"ОШИБКА: {ошибка}", file=sys.stderr)
        return 1

    if расхождения:
        print("Пересчёт разошёлся с тем, что лежит в репозитории:", file=sys.stderr)
        for строка in расхождения:
            print(f"  {строка}", file=sys.stderr)
        return 1
    print("Сверка пройдена: пересчёт совпал с тем, что лежит в репозитории.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
