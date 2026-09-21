"""CLI: сверка нашего отпечатка с эталоном KLIFS.

Запуск:
    uv run python scripts/calibrate_ifp.py --target 6tgu      # дымовая сверка
    uv run python scripts/calibrate_ifp.py --sample           # выборка структур

Дымовая сверка отвечает на вопрос «не сломан ли расчёт очевидным образом» и считается
на одной мишени. Полная сверка (`--sample`) меряет величину расхождения на выборке
киназ `CALIBRATION_KINASES` и пишет медиану, IQR и разбивку по типам взаимодействий
**на одном и том же наборе структур**: посчитанные на разных выборках, эти числа
в текст не идут.

Логики здесь нет: разбор аргументов, печать и запись отчёта.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from opencadd.databases.klifs import setup_remote

from experiments.markdown import merge_sections
from experiments.measurements import (
    Measurement,
    markdown_table,
    select,
    targets,
    upsert_measurements,
)
from kinase_ifp.calibration import (
    CalibrationResult,
    SampleSummary,
    StructureCalibration,
    calibrate_sample,
    calibrate_target,
    gross_failures,
    read_structure_calibration_csv,
    summarize_calibration,
    write_structure_calibration_csv,
)
from kinase_ifp.config import CALIBRATION_KINASES, DATA_DIR, MEASUREMENTS_CSV, RESULTS_DIR
from kinase_ifp.fingerprint import compute_ifp
from kinase_ifp.klifs import (
    filter_structures,
    read_fingerprint_table,
    select_best_per_kinase,
)
from kinase_ifp.molecule_io import read_mol
from kinase_ifp.pocket import Pocket, load_pocket
from kinase_ifp.protonate import prepare_ligand
from kinase_ifp.reselect import read_reselected_sample, select_reselected

# Режим, в котором приводится подробная разбивка и по которому ведётся основной вывод.
# Режим на весь проект задаётся явно; здесь он только не подразумевается молча.
MAIN_PROTONATION = "explicit"

# Заголовок раздела полной сверки. Отчёт собирается слиянием по заголовкам
# (`experiments.markdown`), поэтому раздел узнаётся по своему номеру и переживает
# любой прогон, который его не считал.
SAMPLE_SECTION = "## 2. Полная сверка на выборке структур"

# Шапка отчёта. Пишется **только при создании файла с нуля**: в существующем документе
# преамбула принадлежит тому, кто его вёл, а дата измерения стоит внутри каждого раздела.
PREAMBLE = (
    "# Сверка отпечатка с эталоном KLIFS",
    "",
    "Совпадения 1:1 не ждём: эталон считает сторонняя программа (FingerPrintLib)",
    "по своим правилам. Порог приёмки не назначается — работа",
    "считается сделанной по факту измерения, и в текст идёт измеренное число",
    "с разбором расхождений.",
)

# Заголовок раздела сверки на перевыбранной выборке. Номер раздела задан константой,
# а не ключом командной строки: раздел определяется составом выборки, а ключом можно
# было бы записать числа перевыбора под видом прежних: требуется
# ровно обратного, оба набора рядом.
RESELECTED_SECTION = "## 4. Сверка на перевыбранной выборке под условия DiffSBDD"

# Куда ложатся по-структурные числа. Путей два, потому что прежний файл служит колонкой
# «было»: прогон на новом составе с прежним умолчанием затёр бы то, с чем сравнивает.
SAMPLE_CSV = DATA_DIR / "calibration" / "e04b_per_structure.csv"
RESELECTED_CSV = DATA_DIR / "calibration" / "e04b_reselected_per_structure.csv"


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов отдельной функцией: по ней тесты сверяют свой набор ключей.

    Хелпер тестов строит `Namespace` руками, и забытый в нём ключ ронял бы прогон
    на `AttributeError` посреди расчёта, а не на разборе аргументов.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="6tgu", help="pdb_id мишени для дымовой сверки")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="посчитать полную сверку на выборке киназ CALIBRATION_KINASES",
    )
    parser.add_argument(
        "--targets-dir", type=Path, default=DATA_DIR / "targets", help="каталог с пакетами мишеней"
    )
    parser.add_argument(
        "--structures",
        type=Path,
        default=DATA_DIR / "klifs" / "structures.csv",
        help="таблица структур KLIFS (scripts/fetch_klifs.py)",
    )
    parser.add_argument(
        "--kinases",
        type=Path,
        default=DATA_DIR / "klifs" / "kinases.csv",
        help="сводка по киназам: из неё берётся группа киназы",
    )
    parser.add_argument(
        "--fingerprints",
        type=Path,
        default=DATA_DIR / "klifs" / "klifs_ifp.csv",
        help="эталонные отпечатки KLIFS (scripts/fetch_klifs_ifp.py)",
    )
    parser.add_argument(
        "--report", type=Path, default=RESULTS_DIR / "calibration.md", help="куда записать числа"
    )
    parser.add_argument(
        "--reselected",
        type=Path,
        default=None,
        help="состав выборки от scripts/classify_complexes.py --reselect-calibration; "
        "с ним сверка считается на перевыбранных структурах и пишется разделом 4",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=SAMPLE_CSV,
        help="прежние по-структурные числа: из них берётся колонка «было» раздела 4",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="куда записать по-структурные числа выборки; по умолчанию путь выбирается "
        "по тому, задан ли --reselected",
    )
    parser.add_argument(
        "--measurements",
        type=Path,
        default=MEASUREMENTS_CSV,
        help="реестр измерений по мишеням: из него собирается раздел 1 по всем мишеням",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _куда_писать_csv(args: argparse.Namespace) -> Path:
    """Путь по-структурного файла: явный ключ, иначе — по составу выборки.

    Прежний файл нельзя перезаписывать прогоном на новом составе: он и есть колонка
    «было». Ошибка была бы молчаливой — числа исчезли бы, а прогон отработал бы успешно.
    """
    if args.csv is not None:
        return Path(args.csv)
    return RESELECTED_CSV if args.reselected else SAMPLE_CSV


def _посчитано() -> str:
    """Дата прогона строкой — она стоит внутри раздела, а не в шапке документа.

    Разделы отчёта считаются порознь: пересчёт выборки не трогает дымовую сверку
    и наоборот. Одна дата на весь файл после первого же такого прогона относилась бы
    только к части чисел, и определить, к какой именно, было бы нечем.
    """
    return f"Посчитано {datetime.now(timezone.utc).date().isoformat()}."


def _таблица_типов(результат: CalibrationResult | SampleSummary) -> list[str]:
    строки = [
        "| Тип | Наш | KLIFS | Общих | Пропущено | Лишних | Находим от KLIFS |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in результат.by_type:
        доля = f"{100 * t.shared / t.reference:.0f} %" if t.reference else "—"
        строки.append(
            f"| `{t.interaction_type}` | {t.ours} | {t.reference} | {t.shared} "
            f"| {t.missed} | {t.extra} | {доля} |"
        )
    return строки


# Вид измерения в реестре (`experiments.measurements`). Числа дымовой сверки хранятся
# там, а не только в markdown: раздел отчёта узнаётся слиянием по номеру, поэтому прогон
# по другой мишени стирал раздел предыдущей. Реестр заменяет строки только
# своей мишени, и отчёт собирается из него целиком — сразу по всем посчитанным мишеням.
SMOKE_MEASUREMENT = "calibration_smoke"
SMOKE_BY_TYPE_MEASUREMENT = "calibration_smoke_by_type"

# Колонки сводной таблицы раздела 1: имя величины в реестре и заголовок колонки.
SMOKE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("tanimoto_all", "Танимото, все типы"),
    ("tanimoto_scoring", "Танимото, типы скора"),
    ("bits_ours", "Наших бит"),
    ("bits_reference", "Эталон"),
    ("bits_common", "Общих"),
    ("bits_extra", "Лишних"),
)


def _измерения_мишени(
    результаты: Sequence[CalibrationResult],
    карман: Pocket,
    поломки: Sequence[str],
    pdb_id: str,
) -> list[Measurement]:
    """Числа дымовой сверки одной мишени строками реестра.

    Принимает результаты по режимам протонирования, разобранный карман, найденные грубые
    поломки и код мишени; возвращает строки для `upsert_measurements`. Разбивка по типам
    пишется только для режима, объявленного в пакете: именно в нём считаются прогоны.
    """
    когда = _посчитано().removeprefix("Посчитано ").rstrip(".")
    команда = f"python scripts/calibrate_ifp.py --target {pdb_id}"
    общее = {
        "target": карман.pdb_id,
        "measured_utc": когда,
        "source": команда,
    }
    строки: list[Measurement] = []
    for результат in результаты:
        значения = {
            "tanimoto_all": f"{результат.tanimoto_all:.3f}",
            "tanimoto_scoring": f"{результат.tanimoto_scoring:.3f}",
            "bits_ours": str(результат.ours),
            "bits_reference": str(результат.reference),
            "bits_common": str(результат.shared),
            "bits_extra": str(результат.extra),
        }
        строки += [
            Measurement(
                measurement=SMOKE_MEASUREMENT,
                variant=результат.protonation,
                metric=имя,
                value=значение,
                unit="bit" if имя.startswith("bits_") else "",
                notes=f"KLIFS ID {карман.klifs_structure_id}",
                **общее,
            )
            for имя, значение in значения.items()
        ]

    основной = next(r for r in результаты if r.protonation == карман.protonation)
    for t in основной.by_type:
        строки += [
            Measurement(
                measurement=SMOKE_BY_TYPE_MEASUREMENT,
                variant=f"{карман.protonation}:{t.interaction_type}",
                metric=имя,
                value=str(значение),
                unit="bit",
                **общее,
            )
            for имя, значение in (
                ("bits_ours", t.ours),
                ("bits_reference", t.reference),
                ("bits_common", t.shared),
                ("bits_missed", t.missed),
                ("bits_extra", t.extra),
            )
        ]

    строки.append(
        Measurement(
            measurement=SMOKE_MEASUREMENT,
            variant=карман.protonation,
            metric="gross_failures",
            value=str(len(поломки)),
            notes="; ".join(поломки),
            **общее,
        )
    )
    return строки


def _подраздел_типов(измерения: Sequence[Measurement], мишень: str, номер: int) -> list[str]:
    """Разбивка по типам одной мишени, собранная из реестра."""
    по_типам = select(измерения, measurement=SMOKE_BY_TYPE_MEASUREMENT, target=мишень)
    if not по_типам:
        return []
    режим = по_типам[0].variant.split(":")[0]
    строки = [
        "",
        f"### 1.{номер}. Разбивка по типам: {мишень}, режим `{режим}`",
        "",
        "| Тип | Наш | KLIFS | Общих | Пропущено | Лишних | Находим от KLIFS |",
        "|---|---|---|---|---|---|---|",
    ]
    типы = sorted({измерение.variant.split(":", 1)[1] for измерение in по_типам})
    for тип in типы:
        значения = {
            измерение.metric: int(измерение.value)
            for измерение in по_типам
            if измерение.variant.endswith(f":{тип}")
        }
        эталон = значения.get("bits_reference", 0)
        доля = f"{100 * значения.get('bits_common', 0) / эталон:.0f} %" if эталон else "—"
        строки.append(
            f"| `{тип}` | {значения.get('bits_ours', 0)} | {эталон} "
            f"| {значения.get('bits_common', 0)} | {значения.get('bits_missed', 0)} "
            f"| {значения.get('bits_extra', 0)} | {доля} |"
        )
    поломки = select(измерения, measurement=SMOKE_MEASUREMENT, target=мишень)
    строка_поломок = next((и for и in поломки if и.metric == "gross_failures"), None)
    if строка_поломок is not None:
        текст = строка_поломок.notes or "ни одна из трёх не подтвердилась"
        строки += ["", f"Грубые поломки: {текст}."]
    return строки


def _раздел_мишени(
    target_json: Path, pdb_id: str, measurements_csv: Path
) -> tuple[list[str], list[str]]:
    """Дымовая сверка: считает мишень, пополняет реестр и собирает раздел по всем мишеням.

    Принимает `target.json` мишени, её код и путь к реестру измерений; возвращает строки
    раздела 1 и список грубых поломок посчитанной мишени. Раздел собирается из реестра,
    поэтому числа прежних мишеней в нём остаются, даже если прогон их не считал.
    """
    результаты = calibrate_target(target_json)
    карман = load_pocket(target_json)

    # Грубые поломки проверяются на режиме, объявленном в пакете: именно в нём считаются
    # все прогоны, и именно он должен быть исправен.
    # `read_mol` вместо `MolFromMolFile`: путь уходит в C++ через ANSI, и кириллица
    # в нём на Windows даёт «Bad input file» (`kinase_ifp.molecule_io`).
    сырой = read_mol(карман.ligand_path)
    if сырой is None:
        raise SystemExit(f"{карман.ligand_path}: RDKit не разобрал лиганд мишени")
    лиганд = prepare_ligand(сырой)
    отпечаток = compute_ifp(карман, лиганд)
    основной = next(r for r in результаты if r.protonation == карман.protonation)
    поломки = gross_failures(основной, отпечаток, карман.reference_ifp)

    for результат in результаты:
        print(
            f"{результат.protonation:16s} все типы {результат.tanimoto_all:.3f} | "
            f"типы скора {результат.tanimoto_scoring:.3f} | наших {результат.ours}, "
            f"эталон {результат.reference}, общих {результат.shared}, лишних {результат.extra}"
        )

    измерения = upsert_measurements(
        _измерения_мишени(результаты, карман, поломки, pdb_id), measurements_csv
    )

    строки = [
        "## 1. Дымовая сверка по мишеням",
        "",
        "Числа берутся из реестра измерений `data/measurements/targets.csv`, поэтому раздел",
        "показывает **все** посчитанные мишени, а не последнюю (`docs/questions.md`,",
        "Прогон по одной мишени обновляет только её строки.",
        "",
        "Команда: `python scripts/calibrate_ifp.py --target <pdb_id>`; последним считался "
        f"`{pdb_id}`.",
        "",
    ]
    строки += markdown_table(
        измерения,
        SMOKE_MEASUREMENT,
        SMOKE_COLUMNS,
        date_header="Посчитано",
    )
    for номер, мишень in enumerate(targets(измерения, SMOKE_BY_TYPE_MEASUREMENT), start=1):
        строки += _подраздел_типов(измерения, мишень, номер)
    return строки, поломки


def _раздел_выборки(rows: list[StructureCalibration]) -> list[str]:
    """Полная сверка на выборке: таблица по структурам, медианы и разбивка по типам."""
    режимы = sorted({row.protonation for row in rows})
    сводки = {
        режим: summarize_calibration([row for row in rows if row.protonation == режим])
        for режим in режимы
    }
    основной = сводки.get(MAIN_PROTONATION, сводки[режимы[0]])
    строки_основного = [row for row in rows if row.protonation == основной.protonation]

    строки = [
        SAMPLE_SECTION,
        "",
        _посчитано(),
        "",
        f"Выборка: {основной.n_structures} киназ {_число_групп(строки_основного)}, "
        "по одной структуре на киназу.",
        "Список киназ задан константой `CALIBRATION_KINASES` (`src/kinase_ifp/config.py`),",
        "структура внутри киназы выбирается кодом — лучшая по `rank_structures` из тех,",
        "у которых есть эталонный отпечаток KLIFS. Тот же набор киназ, на котором измерено",
        "исключение `HYD` из скора (`docs/metrics.md`, 3.9): одна выборка на два утверждения.",
        "",
        "Команда: `python scripts/calibrate_ifp.py --sample`. Пакеты мишеней собираются",
        "во временный каталог, `data/targets/` не меняется; по-структурные числа —",
        "в `data/calibration/e04b_per_structure.csv`.",
        "",
        "### Медиана и межквартильный размах",
        "",
        "| Режим | Танимото, все типы | IQR | Танимото, типы скора | IQR | Структур |",
        "|---|---|---|---|---|---|",
    ]
    for режим in режимы:
        сводка = сводки[режим]
        med_a, q1_a, q3_a = сводка.tanimoto_all
        med_s, q1_s, q3_s = сводка.tanimoto_scoring
        строки.append(
            f"| `{режим}` | **{med_a:.3f}** | {q1_a:.3f}–{q3_a:.3f} | **{med_s:.3f}** "
            f"| {q1_s:.3f}–{q3_s:.3f} | {сводка.n_structures} |"
        )

    if основной.n_scoring_undefined:
        учтено = основной.n_structures - основной.n_scoring_undefined
        строки += [
            "",
            f"Медиана по типам скора посчитана по {учтено} структурам из "
            f"{основной.n_structures}: у остальных эталон состоит",
            "из одних гидрофобных бит, и воспроизводить по типам скора нечего —",
            "величина не определена, а не равна нулю",
            "(`docs/metrics.md`, 3.9). Медиана по всем семи типам считается по всей выборке.",
        ]

    строки += [
        "",
        f"### По структурам, режим `{основной.protonation}`",
        "",
        "| PDB | Киназа | Группа | Все типы | Типы скора |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(строки_основного, key=lambda r: r.result.tanimoto_all, reverse=True):
        скор = (
            f"{row.result.tanimoto_scoring:.3f}"
            if row.result.scoring_defined
            else "— (эталон только `HYD`)"
        )
        строки.append(
            f"| {row.pdb_id} | {row.kinase} | {row.group or '—'} "
            f"| {row.result.tanimoto_all:.3f} | {скор} |"
        )

    строки += [
        "",
        f"### Разбивка по типам на той же выборке, режим `{основной.protonation}`",
        "",
    ]
    строки += _таблица_типов(основной)
    return строки


def _раздел_перевыбора(
    rows: list[StructureCalibration], baseline: list[StructureCalibration], состав: Path
) -> list[str]:
    """Сверка на перевыбранной выборке рядом с прежними числами, а не вместо них.

    Прежние числа приходят из `--baseline-csv` и сводятся **той же** функцией, что новые:
    переписанные из текста, они отличались бы форматированием, и эта разница читалась бы
    как разница выборок.

    Соединение идёт по имени киназы, а не по коду структуры: шесть кодов из двенадцати
    сменились, и соединение по PDB дало бы шесть пустых строк вместо сравнения.
    """
    режимы = sorted({row.protonation for row in rows})
    сводки = {
        режим: summarize_calibration([row for row in rows if row.protonation == режим])
        for режим in режимы
    }
    основной = сводки.get(MAIN_PROTONATION, сводки[режимы[0]])
    новые = {row.kinase: row for row in rows if row.protonation == основной.protonation}
    прежние = {row.kinase: row for row in baseline if row.protonation == основной.protonation}
    было = summarize_calibration(list(прежние.values())) if прежние else None

    строки = [
        RESELECTED_SECTION,
        "",
        _посчитано(),
        "",
        "Половина прежней выборки описывала связывание, которого модель воспроизвести",
        "не может: у шести структур из двенадцати в кармане ион металла, добавка",
        "кристаллизации или ковалентная связь лиганда с белком",
        "`docs/pocket-content.md`). Выборка перевыбрана тем же правилом `rank_structures`",
        "с третьим условием — годностью кармана — и осталась полной, 12 киназ из 12.",
        "",
        f"Состав: `{состав.as_posix()}`. Команда:",
        "",
        "```",
        f"python scripts/calibrate_ifp.py --sample --reselected {состав.as_posix()}",
        "```",
        "",
        "**Прежние числа раздела 2 остаются в силе и здесь не заменяются** — какой набор",
        "пойдёт в текст, решается отдельно.",
        "",
        f"### Медиана и межквартильный размах, режим `{основной.protonation}`",
        "",
        "| Набор | Танимото, все типы | IQR | Танимото, типы скора | IQR | Структур |",
        "|---|---|---|---|---|---|",
    ]
    for имя, сводка in (("прежняя выборка", было), ("перевыбранная", основной)):
        if сводка is None:
            строки.append(f"| {имя} | — | — | — | — | — |")
            continue
        med_a, q1_a, q3_a = сводка.tanimoto_all
        med_s, q1_s, q3_s = сводка.tanimoto_scoring
        учтено = сводка.n_structures - сводка.n_scoring_undefined
        строки.append(
            f"| {имя} | **{med_a:.3f}** | {q1_a:.3f}–{q3_a:.3f} | **{med_s:.3f}** "
            f"| {q1_s:.3f}–{q3_s:.3f} | {сводка.n_structures} (скор: {учтено}) |"
        )

    if было is not None and было.n_scoring_undefined != основной.n_scoring_undefined:
        строки += [
            "",
            "**Знаменатель медианы по типам скора сменился.** Величина не определена там,",
            "где эталон состоит из одних гидрофобных бит: воспроизводить по типам скора",
            "нечего, и это не ноль (`docs/metrics.md`, 3.9). Прежде таких структур —",
            f"{было.n_scoring_undefined}, теперь — {основной.n_scoring_undefined}.",
            "Без этой оговорки сдвиг медианы смешался бы со сменой знаменателя.",
        ]

    строки += [
        "",
        f"### По киназам, режим `{основной.protonation}`",
        "",
        "| Киназа | Было | Стало | Все типы: было → стало | Типы скора: было → стало |",
        "|---|---|---|---|---|",
    ]
    for имя in sorted(новые, key=lambda k: новые[k].result.tanimoto_all, reverse=True):
        новая = новые[имя]
        прежняя = прежние.get(имя)
        строки.append(
            f"| {имя} | {прежняя.pdb_id if прежняя else '—'} | {новая.pdb_id} "
            f"| {_пара(прежняя, новая, скор=False)} | {_пара(прежняя, новая, скор=True)} |"
        )

    строки += ["", f"### Разбивка по типам, режим `{основной.protonation}`", ""]
    строки += _таблица_типов_сравнение(было, основной)
    return строки


def _пара(
    прежняя: StructureCalibration | None, новая: StructureCalibration, *, скор: bool
) -> str:
    """Клетка «было → стало». Неопределённая величина показывается прочерком, а не нулём."""

    def одно(row: StructureCalibration | None) -> str:
        if row is None:
            return "—"
        if скор:
            return f"{row.result.tanimoto_scoring:.3f}" if row.result.scoring_defined else "н/о"
        return f"{row.result.tanimoto_all:.3f}"

    слева, справа = одно(прежняя), одно(новая)
    return справа if прежняя is None else f"{слева} → {справа}"


def _таблица_типов_сравнение(было: SampleSummary | None, стало: SampleSummary) -> list[str]:
    """Доля найденных бит эталона по типам: прежняя выборка рядом с новой."""
    прежние = {t.interaction_type: t for t in было.by_type} if было else {}
    строки = [
        "| Тип | Наш | KLIFS | Общих | Лишних | Находим от KLIFS: было → стало |",
        "|---|---|---|---|---|---|",
    ]
    for t in стало.by_type:
        старый = прежние.get(t.interaction_type)
        было_доля = (
            f"{100 * старый.shared / старый.reference:.0f} %"
            if старый and старый.reference
            else "—"
        )
        стало_доля = f"{100 * t.shared / t.reference:.0f} %" if t.reference else "—"
        строки.append(
            f"| `{t.interaction_type}` | {t.ours} | {t.reference} | {t.shared} "
            f"| {t.extra} | {было_доля} → {стало_доля} |"
        )
    return строки


def _число_групп(rows: list[StructureCalibration]) -> str:
    """Словами: сколько групп киназ в выборке. Пустые группы не считаются."""
    группы = {row.group for row in rows if row.group}
    return f"из {len(группы)} групп" if группы else "(группы киназ в данных не заполнены)"


def _проставить_группы(выборка: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Берёт группу киназы из сводки `kinases.csv`.

    В таблице структур KLIFS колонка `kinase.group` не заполнена — база отдаёт её
    только запросом по имени киназы, и сводка `kinases.csv` уже
    собрана с ней. Группа нужна тексту: выборка описывается числом групп киназ,
    и проверить это утверждение надо по таблице, а не на слово.
    """
    if not path.is_file():
        print(f"Нет файла {path}: группы киназ в отчёте останутся пустыми")
        return выборка

    сводка = pd.read_csv(path)
    группы = dict(zip(сводка["kinase.klifs_name"], сводка["kinase.group"], strict=True))
    результат = выборка.copy()
    результат["kinase.group"] = [группы.get(имя, "") for имя in результат["kinase.klifs_name"]]
    return результат


def _собрать_выборку(args: argparse.Namespace) -> list[StructureCalibration]:
    """Отбирает структуры выборки и считает сверку каждой (по сети — только координаты)."""
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
    # Состав берётся из файла перевыбора, а не пересобирается `select_best_per_kinase`:
    # та функция про годность кармана не знает и вернула бы прежние структуры,
    # половина которых условиям DiffSBDD не отвечает.
    if args.reselected:
        коды = read_reselected_sample(Path(args.reselected))
        отобранные = select_reselected(
            структуры, коды, отпечатки.keys(), expect_kinases=CALIBRATION_KINASES
        )
    else:
        отобранные = select_best_per_kinase(структуры, CALIBRATION_KINASES, отпечатки.keys())
    выборка = _проставить_группы(отобранные, args.kinases)
    группы = sorted({str(g) for g in выборка["kinase.group"] if str(g).strip()})
    print(f"Выборка: {len(выборка)} структур — {', '.join(выборка['structure.pdb_id'])}")
    print(f"Групп киназ: {len(группы)} — {', '.join(группы)}")

    session = setup_remote()
    with TemporaryDirectory(ignore_cleanup_errors=True) as времянка:
        rows = calibrate_sample(session, выборка, отпечатки, Path(времянка))

    for row in rows:
        if row.protonation != MAIN_PROTONATION:
            continue
        скор = (
            f"{row.result.tanimoto_scoring:.3f}" if row.result.scoring_defined else "не определён"
        )
        print(
            f"{row.pdb_id} {row.kinase:8s} все типы {row.result.tanimoto_all:.3f} | "
            f"типы скора {скор}"
        )
    return rows


def _прежние_числа(args: argparse.Namespace) -> list[StructureCalibration]:
    """Прежняя сверка для колонки «было». Её отсутствие — не отказ.

    Прежние числа в разделе 4 — иллюстрация к измерению, а не само измерение: без них
    новые числа остаются полными и верными, поэтому пропажа файла печатается и работа
    продолжается.
    """
    путь = Path(args.baseline_csv)
    if not путь.is_file():
        print(f"Нет прежних чисел {путь}: колонка «было» останется пустой")
        return []
    return read_structure_calibration_csv(путь)


def main() -> None:
    args = parse_args()
    if args.reselected and not args.sample:
        raise SystemExit(
            "--reselected задаёт состав выборки и без --sample ничего не делает. "
            "Добавьте --sample либо уберите --reselected"
        )
    target_json = args.targets_dir / args.target / "target.json"
    if not target_json.is_file():
        raise SystemExit(f"Нет пакета мишени {target_json}. Сначала scripts/select_target.py")

    строки_мишени, поломки = _раздел_мишени(target_json, args.target, Path(args.measurements))
    разделы = [строки_мишени]

    if args.sample:
        rows = _собрать_выборку(args)
        if args.reselected:
            разделы.append(
                _раздел_перевыбора(rows, _прежние_числа(args), Path(args.reselected))
            )
        else:
            разделы.append(_раздел_выборки(rows))
        куда = _куда_писать_csv(args)
        write_structure_calibration_csv(rows, куда)
        print(f"По структурам: {куда}")

    # Отчёт обновляется по разделам, а не переписывается целиком: раздел 3 дописан
    # руками, и тотальная запись стирала его при каждом прогоне.
    старый = args.report.read_text(encoding="utf-8") if args.report.is_file() else None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        merge_sections(старый, разделы, preamble=PREAMBLE), encoding="utf-8"
    )
    print(f"\nОтчёт: {args.report}")

    if поломки:
        перечень = "\n".join(f"  - {п}" for п in поломки)
        raise SystemExit(f"Сверка нашла грубые расхождения:\n{перечень}")


if __name__ == "__main__":
    main()
