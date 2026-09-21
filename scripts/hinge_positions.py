"""CLI: частота водородных связей по позициям кармана в эталонах KLIFS.

Запуск:
    docker compose run --rm dev python scripts/hinge_positions.py

Читает `data/klifs/klifs_ifp.csv` и пишет отчёт о позициях шарнира:
таблицу частот по всем 85 позициям и сравнение с `HINGE_POSITIONS`.

Логики здесь нет: разбор аргументов, вызов `kinase_ifp.hinge` и печать сводки.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kinase_ifp.config import (
    DATA_DIR,
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET_INHIBITOR_TYPES,
)
from kinase_ifp.hinge import (
    TOP_POSITIONS,
    compare_with_hinge,
    fingerprints_from_bits,
    hbond_frequencies,
    hinge_verdict,
    render_report,
)
from kinase_ifp.klifs import read_fingerprint_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ifp",
        type=Path,
        default=DATA_DIR / "klifs" / "klifs_ifp.csv",
        help="таблица эталонных отпечатков KLIFS",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "hinge.md",
        help="куда записать отчёт",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=TOP_POSITIONS,
        help="сколько самых частых позиций показать отдельной таблицей",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ifp.is_file():
        raise SystemExit(
            f"Нет таблицы отпечатков {args.ifp}. "
            "Собрать: python scripts/fetch_klifs_ifp.py --limit 0"
        )

    таблица = read_fingerprint_table(args.ifp)
    if таблица.empty:
        raise SystemExit(f"{args.ifp}: таблица пуста, считать частоты не по чему")

    частоты, комплексов = hbond_frequencies(fingerprints_from_bits(таблица["bits"]))
    сводка = compare_with_hinge(частоты, комплексов, top_n=args.top)

    фильтры = (
        f"тип ингибирования {', '.join(TARGET_INHIBITOR_TYPES)}, "
        f"разрешение не хуже {MAX_RESOLUTION} Å, оценка качества не ниже {MIN_QUALITY_SCORE}"
    )
    # Путь пишется относительно корня проекта: внутри контейнера он начинается
    # с /workspace, и абсолютный путь в файле под git читался бы как чужой.
    источник = args.ifp.resolve()
    if источник.is_relative_to(PROJECT_ROOT):
        источник = источник.relative_to(PROJECT_ROOT)
    args.out.write_text(
        render_report(сводка, источник.as_posix(), фильтры), encoding="utf-8"
    )

    print(f"Комплексов учтено: {комплексов}")
    print("Позиции шарнира (доля комплексов с H-связью, ранг из 85):")
    for стат in сводка.hinge:
        print(f"  {стат.position}: {стат.frequency:.3f}, ранг {стат.rank}")
    print("Ключевые позиции:")
    for имя, стат in сводка.key:
        print(f"  {стат.position} ({имя}): {стат.frequency:.3f}, ранг {стат.rank}")
    верх = ", ".join(f"{стат.position} ({стат.frequency:.3f})" for стат in сводка.top)
    print(f"Топ-{len(сводка.top)} позиций: {верх}")
    print(f"Разметка KLIFS и наблюдаемые контакты: {hinge_verdict(сводка)}")
    print(f"Записано: {args.out}")


if __name__ == "__main__":
    main()
