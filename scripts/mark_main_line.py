"""CLI: пометить прогон как представляющий работу в таблице «было/стало».

Запуск:
    docker compose run --rm dev python scripts/mark_main_line.py --run runs/<id>
    docker compose run --rm dev python scripts/mark_main_line.py --run runs/<id> --off

Метка живёт в паспорте прогона (ключ `main_line`) и читается правилом отбора
`experiments.registry.main_line_runs`: помеченный прогон представляет свой источник
независимо от полноты и свежести. Без метки правило работает по-прежнему.

Зачем это нужно: отбор пересчитывается при каждом запуске,
и прогон, поставленный ради сужения доверительных интервалов, молча занял место того,
по которому написан раздел «Результаты». Ставить метку или нет — решение авторов
работы, а не механизма; здесь только способ её записать.

Логики тут нет: разбор аргументов и вызов `runs.record_main_line`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runs import RunError, record_main_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="папка прогона в runs/")
    parser.add_argument(
        "--off",
        action="store_true",
        help="снять метку. Смена метки разрешена: она говорит о представлении работы, "
        "а не о том, чем прогон считали",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        паспорт = record_main_line(args.run, main_line=not args.off)
    except RunError as ошибка:
        raise SystemExit(f"метка не записана: {ошибка}") from ошибка

    состояние = "снята" if args.off else "поставлена"
    print(f"{args.run.name}: метка «основной» {состояние}")
    print(f"источник: {паспорт.get('source')}, мишень: {паспорт.get('target', {}).get('pdb_id')}")
    print("проверить отбор: python scripts/build_report.py")


if __name__ == "__main__":
    main()
