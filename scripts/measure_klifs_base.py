"""CLI: состав выгрузки KLIFS и вырожденность скора по всей базе (пункты B.2 и B.4).

Запуск:
    docker compose run --rm dev python scripts/measure_klifs_base.py

Печатает два раздела. Первый — состав набора данных для «Материалов и методов»:
сколько структур выгружено, сколько прошло фильтры, сколько киназ и эталонов, какова
доля нуклеотидных комплексов, как распределены разрешение и оценка качества
относительно порогов отбора. Второй — знаменатель скора по всем эталонам базы: он
отвечает на вопрос, свойство ли это нашей мишени или схемы KLIFS вообще.

Логики здесь нет: разбор аргументов и вызов `kinase_ifp.base_stats`.
"""

from __future__ import annotations

import argparse

from kinase_ifp.base_stats import (
    FEW_DIRECTED_BITS,
    BaseStatsError,
    Distribution,
    dataset_composition,
    reference_bits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        choices=("состав", "эталоны"),
        help="печатать только один раздел; по умолчанию печатаются оба",
    )
    return parser.parse_args()


def печать_распределения(распределение: Distribution, сторона: str) -> None:
    """Печатает квартили колонки и сколько значений остаётся по нужную сторону порога."""
    print(
        f"   {распределение.name:18} "
        f"мин {распределение.minimum:6.2f} · Q1 {распределение.q1:6.2f} · "
        f"медиана {распределение.median:6.2f} · Q3 {распределение.q3:6.2f} · "
        f"макс {распределение.maximum:6.2f}"
    )
    print(
        f"   {'':18} порог {сторона} {распределение.threshold}: "
        f"проходит {распределение.passing} из {распределение.count} "
        f"({распределение.passing_share:.1%})"
    )


def печать_состава() -> None:
    состав = dataset_composition()
    print("1. СОСТАВ ВЫГРУЗКИ")
    print(f"   структур выгружено           {состав.downloaded}")
    print(f"   прошло фильтры отбора        {состав.selected}")
    print(f"   киназ в выгрузке             {состав.kinases}")
    print(f"   эталонных отпечатков         {состав.fingerprints}")
    print(
        f"   тип I с лигандом             {состав.with_ligand}, "
        f"из них нуклеотид {состав.nucleotide} ({состав.nucleotide_share:.1%})"
    )
    печать_распределения(состав.resolution, "не хуже")
    печать_распределения(состав.quality, "не ниже")


def печать_эталонов() -> None:
    эталоны = reference_bits()
    print("2. ЗНАМЕНАТЕЛЬ СКОРА ПО ВСЕЙ БАЗЕ")
    print(f"   эталонов                     {эталоны.structures}")
    print(
        f"   направленных бит             медиана {эталоны.directed_median:.0f} · "
        f"среднее {эталоны.directed_mean:.2f} · "
        f"от {эталоны.directed_minimum} до {эталоны.directed_maximum}"
    )
    print(
        f"   не больше {FEW_DIRECTED_BITS} бит              у {эталоны.few_directed} структур "
        f"({эталоны.few_directed_share:.1%})"
    )
    print(f"   всего бит                    {эталоны.total_bits}")
    for тип, количество in эталоны.bits_by_type:
        доля = количество / эталоны.total_bits if эталоны.total_bits else 0.0
        print(f"      {тип:5} {количество:7}  {доля:6.1%}")
    print(
        f"   позиций кармана с контактом  медиана {эталоны.positions_median:.0f}, "
        f"из них с направленным {эталоны.directed_positions_median:.0f}"
    )


def main() -> None:
    args = parse_args()
    try:
        if args.only != "эталоны":
            печать_состава()
        if args.only is None:
            print()
        if args.only != "состав":
            печать_эталонов()
    except BaseStatsError as ошибка:
        raise SystemExit(f"Посчитать не удалось: {ошибка}") from ошибка


if __name__ == "__main__":
    main()
