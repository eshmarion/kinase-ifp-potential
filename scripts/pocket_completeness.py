"""CLI: полнота кармана KLIFS у мишени и у выборки сверки.

Запуск:
    uv run python scripts/pocket_completeness.py --target data/targets/6tgu/target.json
    uv run python scripts/pocket_completeness.py --sample

Отпечаток считается по 85 позициям выравнивания кармана, и позиция без остатка даёт
ноль, неотличимый от честного «взаимодействия здесь нет». Скрипт
отвечает на два вопроса: сколько позиций занято и приходится ли дыра на позицию,
где у эталона KLIFS стоят биты. Второе и есть **систематически невоспроизводимое
взаимодействие**: остатка в структуре нет, значит наш расчёт не поставит там бита
никогда, сколько бы верной ни была метрика.

`--sample` ходит в KLIFS по сети: пакеты выборки собираются во временный каталог,
как в `calibrate_sample`, и в `data/targets/` не попадают. ProLIF при этом не зовётся —
полнота считается по паспорту, а не по структуре, и прогон идёт минутами, а не часами.

Логики здесь нет: разбор аргументов, печать и запись раздела отчёта.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from opencadd.databases.klifs import setup_remote

from experiments.markdown import merge_sections
from kinase_ifp.completeness import (
    check_sequence_agreement,
    harmful_missing_positions,
    missing_positions_with_reference_bits,
    pocket_completeness,
    pocket_gap_statistics,
)
from kinase_ifp.config import CALIBRATION_KINASES, DATA_DIR, N_KLIFS_POSITIONS, RESULTS_DIR
from kinase_ifp.klifs import (
    MISSING_VALUE_MARKERS,
    build_target_package,
    filter_structures,
    read_fingerprint_table,
    select_best_per_kinase,
)
from kinase_ifp.reselect import read_reselected_sample, select_reselected

# Заголовок раздела в отчёте сверки. Отчёт собирается слиянием по заголовкам, поэтому
# раздел узнаётся по номеру и переживает прогон, который его не считал.
COMPLETENESS_SECTION = "## 6. Полнота кармана"

# Сколько самых частых пустующих позиций показывать в фоне по базе.
TOP_GAPS = 3

REPORT = RESULTS_DIR / "calibration.md"

# Шапка — только на случай создания файла с нуля; у существующего отчёта преамбула
# принадлежит тому, кто его вёл.
PREAMBLE = (
    "# Сверка отпечатка с эталоном KLIFS",
    "",
    "Собрано `scripts/pocket_completeness.py`.",
)


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов отдельной функцией: по ней тесты сверяют набор ключей."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=Path,
        default=DATA_DIR / "targets" / "6tgu" / "target.json",
        help="пакет мишени, для которого считается полнота кармана",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="посчитать полноту по выборке киназ CALIBRATION_KINASES (по сети)",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        default=DATA_DIR / "klifs" / "structures.csv",
        help="таблица структур, созданная scripts/fetch_klifs.py",
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=DATA_DIR / "klifs" / "klifs_ifp.csv",
        help="таблица эталонных отпечатков, созданная scripts/fetch_klifs_ifp.py",
    )
    parser.add_argument(
        "--reselected",
        type=Path,
        default=DATA_DIR / "klifs" / "calibration_diffsbdd_ready.csv",
        help="состав перевыбранной выборки под условия DiffSBDD, от "
        "scripts/classify_complexes.py --reselect-calibration",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"записать результат разделом «{COMPLETENESS_SECTION}» в {REPORT}",
    )
    return parser


def описать_пакет(package: dict[str, Any], имя: str) -> list[str]:
    """Строки отчёта по одному пакету мишени."""
    полнота = pocket_completeness(package)
    блоки = missing_positions_with_reference_bits(package)
    вредные = harmful_missing_positions(package)

    строки = [
        f"- **{имя}**: занято {полнота.filled} позиций из {N_KLIFS_POSITIONS}",
    ]
    if полнота.is_complete:
        строки[0] += " — карман полон"
        return строки

    строки[0] += f", пустуют {', '.join(str(п) for п in полнота.missing)}"
    if not блоки:
        строки.append("  - эталона у структуры нет: сверять дыры не с чем")
        return строки
    for позиция in полнота.missing:
        блок = блоки[позиция]
        вердикт = "эталон несёт биты" if позиция in вредные else "эталон пуст"
        строки.append(f"  - позиция {позиция}: `{блок}` — {вердикт}")
    return строки


def прогон_по_мишени(args: argparse.Namespace) -> list[str]:
    """Считает полноту одного пакета и печатает её."""
    if not args.target.is_file():
        raise SystemExit(f"Нет файла {args.target}. Сначала scripts/select_target.py")

    package = json.loads(args.target.read_text(encoding="utf-8"))
    # Сверка карты позиций со строкой кармана делается и здесь, а не только при сборке:
    # пакет мог быть собран раньше, и молча считать по разъехавшимся
    # сторонам нельзя — числа вышли бы правдоподобными и неверными.
    check_sequence_agreement(package, MISSING_VALUE_MARKERS)

    строки = описать_пакет(package, str(package.get("pdb_id", args.target.parent.name)))
    for строка in строки:
        print(строка)
    return строки


def прогон_по_выборке(args: argparse.Namespace) -> list[str]:
    """Собирает пакеты выборки во временный каталог и считает полноту каждого."""
    if not args.structures.is_file():
        raise SystemExit(f"Нет файла {args.structures}. Сначала scripts/fetch_klifs.py")
    if not args.fingerprints.is_file():
        raise SystemExit(f"Нет файла {args.fingerprints}. Сначала scripts/fetch_klifs_ifp.py")

    структуры = filter_structures(pd.read_csv(args.structures))
    таблица = read_fingerprint_table(args.fingerprints)
    отпечатки = {
        int(sid): str(bits)
        for sid, bits in zip(таблица["structure_id"], таблица["bits"], strict=True)
    }
    # Состав тот же, что у полной сверки: перевыбранная выборка, если файл есть
    #, иначе прежняя. Считать полноту на другом составе,
    # чем считалось согласие с эталоном, бессмысленно — числа не встали бы рядом.
    if args.reselected.is_file():
        коды = read_reselected_sample(args.reselected)
        выборка = select_reselected(
            структуры, коды, отпечатки.keys(), expect_kinases=CALIBRATION_KINASES
        )
        состав = "перевыбранная выборка"
    else:
        выборка = select_best_per_kinase(структуры, CALIBRATION_KINASES, отпечатки.keys())
        состав = "прежняя выборка"

    print(f"Выборка: {len(выборка)} структур — {', '.join(выборка['structure.pdb_id'])}")

    строки: list[str] = []
    неполные = 0
    вредные_всего = 0
    задетых_структур = 0
    session = setup_remote()
    with TemporaryDirectory(ignore_cleanup_errors=True) as времянка:
        for _, структура in выборка.iterrows():
            klifs_id = int(структура["structure.klifs_id"])
            путь = build_target_package(
                session, структура, Path(времянка), отпечатки.get(klifs_id)
            )
            package = json.loads(путь.read_text(encoding="utf-8"))
            полнота = pocket_completeness(package)
            вредные = harmful_missing_positions(package)
            if not полнота.is_complete:
                неполные += 1
            if вредные:
                задетых_структур += 1
                вредные_всего += sum(блок.count("1") for блок in вредные.values())
            имя = f"{структура['structure.pdb_id']} ({структура['kinase.klifs_name']})"
            блок_строк = описать_пакет(package, имя)
            for строка in блок_строк:
                print(строка)
            строки.extend(блок_строк)

    итог = (
        f"Из {len(выборка)} структур карман неполон у {неполные}; "
        f"дыры задевают биты эталона у {задетых_структур} структур "
        f"({вредные_всего} бит всего)."
    )
    print(итог)
    return [f"Состав: {состав}.", "", *строки, "", итог]


def фон_по_базе(structures: Path) -> list[str]:
    """Насколько неполный карман — обычное дело в KLIFS. Считается без сети.

    Нужен, чтобы число по нашей мишени читалось верно. «У 6tgu карман неполон»
    звучит как дефект отбора; на фоне базы видно, что это норма выравнивания.
    """
    if not structures.is_file():
        return []

    строки_кармана = pd.read_csv(structures)["structure.pocket"].dropna().astype(str)
    статистика = pocket_gap_statistics(строки_кармана, MISSING_VALUE_MARKERS)
    if not статистика.total:
        return []

    перечень = ", ".join(
        f"{позиция} ({сколько / статистика.total:.1%})"
        for позиция, сколько in статистика.most_common(TOP_GAPS)
    )
    return [
        f"Фон по выгрузке KLIFS ({статистика.total} структур со строкой кармана): "
        f"карман неполон у {статистика.incomplete} из них — "
        f"{статистика.incomplete_share:.1%}. Чаще всего пустуют позиции {перечень}.",
        "",
        "Неполный карман — норма выравнивания, а не дефект кристалла. Позиция 50, "
        "самая частая дыра и единственная у нашей мишени, размечена KLIFS как "
        "`linker.50` — середина линкера между шарниром (46–48) и спиралью αD (53–54). "
        "Линкер у разных киназ разной длины, поэтому вставочная позиция у короткого "
        "линкера не занята **в принципе**, а не потеряна при съёмке.",
        "",
        "Отсюда предел, который важнее самой полноты: эталон KLIFS считается по тем же "
        "позициям, что и наш отпечаток, поэтому дыра не создаёт расхождения между нами "
        "и эталоном — она одинаково слепа с обеих сторон. Взаимодействие, которое "
        "пришлось бы на незанятую позицию, невидимо **и нам, и эталону**; сверка такую "
        "потерю не покажет никогда.",
    ]


def собрать_раздел(строки: list[str]) -> list[str]:
    """Раздел отчёта с заголовком и датой измерения."""
    дата = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    return [
        COMPLETENESS_SECTION,
        "",
        f"Посчитано {дата} командой `scripts/pocket_completeness.py`.",
        "",
        "Позиция без остатка даёт в отпечатке ноль, неотличимый от честного",
        "«взаимодействия здесь нет». Опасна не всякая дыра, а та, на которой",
        "у эталона KLIFS стоят биты: там наш расчёт не поставит бита никогда,",
        "и расхождение с эталоном объясняется отсутствующим остатком, а не методом.",
        "",
        *строки,
    ]


def main() -> None:
    args = build_parser().parse_args()

    строки = прогон_по_выборке(args) if args.sample else прогон_по_мишени(args)
    фон = фон_по_базе(args.structures)
    for строка in фон:
        print(строка)

    if args.write:
        прежний = REPORT.read_text(encoding="utf-8") if REPORT.is_file() else None
        REPORT.write_text(
            merge_sections(прежний, [собрать_раздел(строки + ["", *фон])], preamble=PREAMBLE),
            encoding="utf-8",
            newline="\n",
        )
        print(f"Записано в {REPORT}")


if __name__ == "__main__":
    main()
