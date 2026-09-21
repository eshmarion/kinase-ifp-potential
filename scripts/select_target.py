"""CLI: выбор мишени и сборка пакета мишени `target.json`.

Запуск (после scripts/fetch_klifs.py):
    uv run python scripts/select_target.py

Мишень выбирается процедурой, а не именем киназы. Результат —
`data/targets/<pdb_id>/target.json` по формату пакета мишени плюс скачанные файлы структуры.

Режим протонирования проставляется здесь же, вторым шагом: отдельной командой о нём
забывали, и 24.08 два пакета разошлись режимами молча.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from opencadd.databases.klifs import setup_remote

from kinase_ifp.config import DATA_DIR
from kinase_ifp.klifs import (
    INHIBITOR_TYPE_COLUMN,
    build_target_package,
    fetch_fingerprints,
    fetch_kinase_groups,
    filter_structures,
    rank_structures,
    select_target,
)
from kinase_ifp.protonation import (
    ON_MISSING_H_IMPLICIT,
    PROTONATION_EXPLICIT,
    annotate_protonation,
)

# Колонки таблицы структур, по которым киназа сопоставляется со своей группой.
# Сама группа в таблице структур пуста и запрашивается отдельно.
NAME_COLUMN = "kinase.klifs_name"
SPECIES_COLUMN = "species.klifs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--structures",
        type=Path,
        default=DATA_DIR / "klifs" / "structures.csv",
        help="таблица структур, созданная scripts/fetch_klifs.py",
    )
    parser.add_argument(
        "--targets-dir",
        type=Path,
        default=DATA_DIR / "targets",
        help="каталог, в котором создаётся папка мишени",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=50,
        help="для скольких лучших структур запрашивать эталонные отпечатки KLIFS",
    )
    parser.add_argument(
        "--exclude-kinase",
        action="append",
        default=[],
        metavar="ИМЯ",
        help=(
            "не рассматривать структуры этой киназы (kinase.klifs_name); ключ можно "
            "повторить. Так отбирается вторая мишень: та же процедура, что дала первую"
        ),
    )
    parser.add_argument(
        "--kinase-group-other-than",
        action="append",
        default=[],
        metavar="ГРУППА",
        help=(
            "отбирать только среди киназ, чья группа KLIFS отличается от указанной; "
            "ключ можно повторить. Процедура отбора при этом не меняется (пункт C.2 "
            "плана подачи). Повторяемым он стал 18.09: у работы уже две группы — CMGC "
            "и TK, — и третья мишень отбирается среди тех, кто не принадлежит ни одной"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.structures.is_file():
        raise SystemExit(
            f"Нет файла {args.structures}. Сначала запустите scripts/fetch_klifs.py"
        )

    # Фильтры применяются повторно: таблица на диске могла быть собрана другой версией
    # кода или отредактирована руками, а мишень обязана удовлетворять порогам.
    structures = filter_structures(pd.read_csv(args.structures))

    # Исключение киназы — способ получить вторую мишень **той же процедурой**, а не
    # назвать её от себя: правило отбора не меняется, из выборки уходит
    # только киназа, на которой уже стоит первая мишень. Фильтр стоит здесь, а не
    # внутри `select_target`: правило отбора едино для всех, а состав выборки —
    # дело вызывающего.
    if args.exclude_kinase:
        было = len(structures)
        structures = structures.loc[~structures["kinase.klifs_name"].isin(args.exclude_kinase)]
        print(
            f"Исключены киназы {', '.join(args.exclude_kinase)}: "
            f"структур {было} -> {len(structures)}"
        )

    # Вторая мишень для C.2 отбирается той же процедурой, но среди киназ другой
    # группы. Условие объявляется ключом, а не зашивается: довод «переносимость
    # проверена между семействами» держится на том, что киназу выбрал код, а
    # не человек. Наша 6tgu — CK2a2, группа CMGC. Ключей два, и они
    # независимы: `--exclude-kinase` убирает названную киназу, `--kinase-group-other-than`
    # — целые группы, по разу на ключ. Какой из двух остаётся процедурой работы,
    # выбирается отдельно.
    session = setup_remote()

    if args.kinase_group_other_than:
        свои = {str(группа) for группа in args.kinase_group_other_than}
        названия = ", ".join(sorted(свои))
        было_всего = len(structures)
        # Группы приходится спрашивать у базы отдельно: в таблице структур колонка
        # `kinase.group` пуста во всех строках (17.09.2026). Ключ парный — имя и вид:
        # одно имя KLIFS носят киназы разных видов, и группы у них разные.
        группы = fetch_kinase_groups(session, structures[NAME_COLUMN].tolist())
        print(f"Групп получено для {len(группы)} пар «киназа, вид»")
        своё = [
            группы.get((str(имя), str(вид))) not in свои
            for имя, вид in zip(structures[NAME_COLUMN], structures[SPECIES_COLUMN], strict=True)
        ]
        structures = structures.loc[своё]
        if structures.empty:
            raise SystemExit(
                f"после отсева групп {названия!r} не осталось ни одной структуры "
                f"из {было_всего}: проверьте написание групп"
            )
        print(f"Отсеяны группы {названия}: осталось {len(structures)} структур из {было_всего}")

    # Отпечатки запрашиваются не для всей выборки, а для лучших кандидатов: запрос идёт
    # в KLIFS по сети, а мишень всё равно берётся из начала списка.
    candidates = rank_structures(structures).head(args.candidates)
    fingerprints = fetch_fingerprints(session, [int(i) for i in candidates["structure.klifs_id"]])
    print(f"Кандидатов рассмотрено: {len(candidates)}, из них с эталонным отпечатком: "
          f"{len(fingerprints)}")

    target = select_target(candidates, fingerprints)
    target_json = build_target_package(
        session, target, args.targets_dir, fingerprints[int(target["structure.klifs_id"])]
    )

    print(f"Мишень: {target['structure.pdb_id']} ({target['kinase.klifs_name']})")
    print(f"Разрешение: {target['structure.resolution']} A, "
          f"качество: {target['structure.qualityscore']}")
    print(f"Конформация: DFG-{target['structure.dfg']}, aC-{target['structure.ac_helix']} "
          f"— тип {target[INHIBITOR_TYPE_COLUMN]}")
    print(f"Альтернативная модель: {target['structure.alternate_model']}, "
          f"KLIFS ID {target['structure.klifs_id']}")
    print(f"Пакет мишени: {target_json}")

    # Белок без водородов — не ошибка сборки, а структура не из конвейера KLIFS.
    # Такую мишень не терять, а считать в режиме
    # 'implicit-prolif' — он законный и уже поддержан ProLIF-конфигом. Своих водородов
    # не достраиваем: корректного инструмента в окружении нет.
    package = annotate_protonation(target_json, on_missing_hydrogens=ON_MISSING_H_IMPLICIT)

    print(f"protonation: {package['protonation']}")
    print(f"protonation_tool: {package['protonation_tool']}")

    if package["protonation"] != PROTONATION_EXPLICIT:
        print(
            "\nВНИМАНИЕ: у белка нет ни одного водорода, мишень переведена в режим "
            "'implicit-prolif'.\nОтпечаток в нём заметно слабее: по типам скора Танимото "
            "0.400 против 1.000 у 'explicit'\n(измерено сверкой). "
            "Отпечатки разных режимов между собой несравнимы —\nв одной таблице мишени "
            "с разными режимами сравнивать нельзя."
        )


if __name__ == "__main__":
    main()
