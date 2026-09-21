"""Рисунки для раздела «Результаты и обсуждение».

Рисунки строятся кодом из CSV прогонов, а не собираются руками: пересчёт данных
не должен означать перерисовку. Здесь только логика, разбор командной строки —
в `scripts/make_figures.py`.

Рисование идёт объектным API matplotlib (`Figure` плюс `savefig`), а не через
`pyplot`. Причина техническая: `pyplot` выбирает backend при импорте, и в образе
без дисплея ему нужен `matplotlib.use("Agg")`, то есть исполняемый код между
импортами. `Figure` сохраняет файл сам и ни от какого backend не зависит.

Три рисунка не равноправны по доступности данных. Рисунок 1 (сходство) и рисунок 3
(схема) строятся уже сейчас, рисунок 2 (контрольные метрики) — только по прогону
`source=diffsbdd`: на наборе поз QED и SA вырождены и по формату таблицы метрик остаются
пустыми. Отсутствие данных для рисунка 2 — это исключение с названной причиной,
а не пустая картинка.
"""

from __future__ import annotations

import csv
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from experiments.layout import (
    CSV_EOL,
    FIGURE_FACTS_CSV,
    FIGURES_ENVIRONMENT_JSON,
    FIGURES_README,
    INDEX_CSV,
    METRICS_CSV,
    PANELS_CSV,
    RANKING_CSV,
    RESCORE_SUMMARY_CSV,
)
from experiments.measurements import (
    MEASUREMENTS_CSV,
    POSES_PREFIX,
    MeasurementError,
    read_measurements,
    select,
)
from experiments.registry import main_line_runs
from experiments.run_io import read_rmsd
from experiments.runs import read_passport
from kinase_ifp.base_stats import KLIFS_DIR, BaseStatsError, selection_values
from kinase_ifp.config import (
    KLIFS_INTERACTION_TYPES,
    MAIN_TARGET_PDB_ID,
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
    RESULTS_DIR,
    RMSD_BIN_EDGES_A,
    RMSD_BIN_LABELS,
    TARGETS_DIR,
)
from kinase_ifp.klifs_rules import (
    KlifsRulesError,
    compare_with_reference,
    fingerprint_by_klifs_rules,
)
from kinase_ifp.scoring import DEFAULT_TOP_FRACTION, ScoredMolecule, rank_molecules

# Разрешение растра. 200 dpi — компромисс: в тексте курсовой рисунок шириной 16 см
# печатается без видимых пикселей, а файл остаётся в сотнях килобайт, то есть проходит
# ограничение на размер файла в git.
FIGURE_DPI: Final[int] = 200

# Минимальный кегль любой надписи на рисунке. Положение ФББ, Приложение 3, п. 1.9:
# «не рекомендуется использовать шрифт меньше 6 пт». Держится тестом
# tests/test_figures.py, а не только договорённостью: подписи осей и делений
# matplotlib берёт из rcParams, и уменьшение фигуры их незаметно не масштабирует.
MIN_FONT_SIZE_PT: Final[float] = 6.0

# Кегли надписей. Заданы явно, потому что дефолт matplotlib зависит от версии,
# а нижняя граница у нас нормативная (см. MIN_FONT_SIZE_PT).
AXIS_LABEL_PT: Final[float] = 10.0
TICK_LABEL_PT: Final[float] = 9.0
BLOCK_TEXT_PT: Final[float] = 8.5

# Формат подписи под рисунком. Положение ФББ, Приложение 3, п. 1.8: слово «Рисунок»,
# номер, тире, название с заглавной буквы; подпись располагается ПОД рисунком.
# Отсюда же следует, что названия внутри картинки (suptitle) быть не должно — иначе
# оно дублирует подпись и расходится с ней при первой же правке.
FIGURE_CAPTION_PREFIX: Final[str] = "Рисунок"

# Как разбиты молекулы на рисунке 1.
GROUPING_CONDITION: Final[str] = "условия эксперимента"
GROUPING_RMSD: Final[str] = "бины RMSD к кристаллической позе"
GROUPING_SELECTION: Final[str] = "источник и отбор по IFP-скору"

# Как называется группа отобранных молекул. Полное имя группы — «<источник>» либо
# «<источник>, топ»: это ровно колонки таблицы «было/стало», и совпадать они обязаны
# не для красоты: у числа должно быть одно место, откуда его берут.
SELECTION_TOP_SUFFIX: Final[str] = ", топ по IFP-скору"

# Контрольные метрики рисунка 2 (docs/metrics.md, разделы 3.3 и 3.4).
CONTROL_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("qed", "QED"),
    ("sa_score", "SA score"),
    ("n_heavy_atoms", "Число тяжёлых атомов"),
)


class FiguresError(RuntimeError):
    """Рисунок нельзя построить по имеющимся данным; в тексте — почему именно."""


@dataclass(frozen=True)
class FigureEntry:
    """Готовый рисунок: файл, название, пояснение и прогоны, из которых он собран.

    Название и пояснение хранятся раздельно, потому что по положению ФББ подпись
    строится как «Рисунок N – Название», а номер известен только на сборке: если
    рисунок 2 не построился из-за отсутствия данных, следующий за ним получает
    номер 2, а не 3. Собирает подпись `full_caption`.
    """

    path: Path
    title: str
    caption: str
    run_ids: tuple[str, ...]
    # Сколько молекул на каждой панели: пара «название разбивки — поголовье».
    # Пусто у рисунков без панелей (схема) и у тех, где разбивок нет.
    panels: tuple[tuple[str, int], ...] = ()
    # Числа, которые называет подпись: пара «что измерено — значение». Подпись
    # попадает в подпись дословно, а подпись сверяется с источником, поэтому у её чисел
    # обязан быть машинный источник. Пусто там, где подпись чисел не называет.
    facts: tuple[tuple[str, float], ...] = ()


def full_caption(entry: FigureEntry, number: int) -> str:
    """Подпись под рисунком по положению ФББ: «Рисунок N – Название. Пояснение».

    Номер сквозной по всей работе (Приложение 3, п. 1.8) и присваивается по порядку
    фактически построенных рисунков, а не по имени файла: пропуск номера в тексте —
    дефект оформления, а несобранный рисунок 2 — штатная ситуация до первого прогона
    `source=diffsbdd`.
    """
    название = entry.title.rstrip(".")
    return f"{FIGURE_CAPTION_PREFIX} {number} – {название}. {entry.caption}"


def _apply_font_sizes(ax: Any) -> None:
    """Кегли подписей осей и делений — не мельче `MIN_FONT_SIZE_PT` (п. 1.9)."""
    ax.xaxis.label.set_fontsize(AXIS_LABEL_PT)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_PT)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_PT)


def runs_with_metrics(runs_dir: Path, pdb_id: str | None = MAIN_TARGET_PDB_ID) -> list[Path]:
    """Прогоны основной линии, у которых посчитан `metrics_per_molecule.csv`.

    **Рисунки строятся по одной мишени** (`pdb_id`, по умолчанию мишень работы).
    Реестр этого не решает и решать не должен: он перечисляет прогоны, а какие
    киназы показывает работа — вопрос состава результатов. Фильтр
    стоит здесь потому, что панели рисунка 1 группируются по условию, а не по
    мишени: два набора поз разных киназ склеились бы в один ящик с подписью
    `local-poses`, и на 17.09 это измерено — 195 молекул вместо 100, медиана
    0.412 вместо 0.458. `pdb_id=None` снимает фильтр.

    Берётся тот же отбор, что и у таблицы «было/стало» (`registry.main_line_runs`):
    по одному прогону на источник — самый полный, среди равных самый
    свежий. Иначе рисунок и таблица говорили бы о разных числах: в реестре четыре
    прогона `source=diffsbdd`, и это не повторы одного условия, а разные способы
    задать карман — склеивать их в один ящик значит смешивать несравнимое.
    Того же требует сверка чисел: у числа одно место, откуда его берут.

    Реестр перечисляет прогоны со статусом `scored`; фильтр по наличию файла остаётся
    сверх этого, потому что рисункам нужен именно формат таблицы метрик.
    """
    if not (runs_dir / INDEX_CSV).is_file():
        return []
    return [
        папка
        for папка in main_line_runs(runs_dir, pdb_id=pdb_id)
        if (папка / METRICS_CSV).is_file()
    ]


def collect_figure_data(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """Склеивает метрики нескольких прогонов и подмешивает `rmsd_to_ref` из SDF.

    Возвращает таблицу формата таблицы метрик плюс колонки `source` (из паспорта прогона)
    и `rmsd_to_ref`. Прогоны объединяются, потому что сиды одного условия —
    независимые повторы: разброс по десяти сидам честнее, чем по одному.
    """
    if not run_dirs:
        raise FiguresError("Не задано ни одного прогона: рисовать нечего")

    куски: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        паспорт = read_passport(run_dir)
        путь = run_dir / METRICS_CSV
        if not путь.is_file():
            raise FiguresError(
                f"Нет файла {путь} (формат таблицы метрик): "
                f"сначала scripts/compute_metrics.py --run {run_dir}"
            )
        кусок = pd.read_csv(путь)
        кусок["source"] = str(паспорт["source"])
        кусок["rmsd_to_ref"] = кусок["mol_id"].map(read_rmsd(run_dir))
        # `selected` живёт в ranking.csv и нужен разбивке по отбору.
        # Прогон без переранжирования — не ошибка: колонка просто не появится,
        # и разбивка сама исчезнет из списка доступных.
        ranking = run_dir / RANKING_CSV
        if ranking.is_file():
            отбор = pd.read_csv(ranking)
            if "mol_id" in отбор.columns and "selected" in отбор.columns:
                кусок = кусок.merge(
                    отбор[["mol_id", "selected"]], on="mol_id", how="left", validate="one_to_one"
                )
        куски.append(кусок)
    return pd.concat(куски, ignore_index=True)


def rmsd_bins(rmsd: pd.Series) -> pd.Series:
    """Метки бинов RMSD по границам из конфигурации; где RMSD нет — пропуск."""
    низ, верх = RMSD_BIN_EDGES_A
    значения = pd.to_numeric(rmsd, errors="coerce")
    метки = pd.Series(pd.NA, index=значения.index, dtype="object")
    метки[значения < низ] = RMSD_BIN_LABELS[0]
    метки[(значения >= низ) & (значения <= верх)] = RMSD_BIN_LABELS[1]
    метки[значения > верх] = RMSD_BIN_LABELS[2]
    return метки


@dataclass(frozen=True)
class Grouping:
    """Одна разбивка молекул на группы: название, метки по молекулам, порядок групп.

    Обычно группы не пересекаются, и достаточно метки на каждую молекулу. Разбивка
    по отбору — исключение: «весь набор» и «топ по IFP-скору» **вложены** друг в друга
    ровно так же, как соседние колонки таблицы «было/стало», где топ считается
    по подмножеству тех же молекул. Поэтому для таких случаев группы задаются масками,
    а не метками: иначе «весь набор» молча означал бы «всё, что не попало в топ»,
    и медиана на рисунке разошлась бы с медианой в таблице.
    """

    name: str
    labels: pd.Series
    order: tuple[str, ...]
    masks: dict[str, pd.Series] | None = None

    def mask(self, метка: str) -> pd.Series:
        """Булева маска молекул одной группы."""
        if self.masks is not None:
            return self.masks[метка]
        return self.labels == метка


def groupings(df: pd.DataFrame) -> list[Grouping]:
    """Все разбивки, которые данные позволяют показать, в порядке вывода на рисунок.

    Их две, и они отвечают на разные вопросы, поэтому одна не заменяет другую
    Разбивка по условиям — главный результат работы:
    меняется ли сходство от guidance. Разбивка по бинам RMSD — доказательство,
    что метрика вообще различает позы, и она нужна в тексте даже тогда, когда
    условий станет много.

    Условия появляются только со свипом, поэтому на прогоне
    с единственным условием остаётся одна разбивка — по RMSD, а на прогоне
    `source=diffsbdd` наоборот: `rmsd_to_ref` там пуст по формату таблицы метрик.
    """
    найденные: list[Grouping] = []

    условия = df["condition"].dropna().astype(str).str.strip()
    условия = условия[условия != ""]
    if условия.nunique() > 1:
        найденные.append(
            Grouping(GROUPING_CONDITION, df["condition"].astype(str), tuple(dict.fromkeys(условия)))
        )

    if "selected" in df.columns:
        отбор = pd.to_numeric(df["selected"], errors="coerce")
        if отбор.notna().sum() > 0 and отбор.nunique() > 1:
            # Источники не смешиваются в один ящик: `source=local-poses`
            # и `source=diffsbdd` — разные утверждения, и подменять одно другим
            # нельзя. Поэтому групп столько же, сколько колонок
            # в таблице «было/стало»: на каждый источник весь набор и его топ.
            порядок: list[str] = []
            маски: dict[str, pd.Series] = {}
            for источник in dict.fromkeys(df["source"].astype(str)):
                свои = df["source"].astype(str) == источник
                порядок.append(источник)
                маски[источник] = свои
                топ = источник + SELECTION_TOP_SUFFIX
                порядок.append(топ)
                маски[топ] = свои & (отбор == 1)
            найденные.append(
                Grouping(
                    GROUPING_SELECTION,
                    df["source"].astype(str),
                    tuple(порядок),
                    masks=маски,
                )
            )

    метки = rmsd_bins(df["rmsd_to_ref"])
    if метки.notna().sum() > 0:
        найденные.append(Grouping(GROUPING_RMSD, метки, RMSD_BIN_LABELS))

    if not найденные:
        raise FiguresError(
            "Условие в прогоне одно, отбора нет и rmsd_to_ref не заполнен ни у одной "
            "молекулы: группировать нечем. Так выглядит прогон source=diffsbdd "
            "до переранжирования — сначала scripts/rerank.py."
        )
    return найденные


def _run_ids(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(dict.fromkeys(df["run_id"].astype(str)))


def _series(df: pd.DataFrame, маска: pd.Series, колонка: str) -> Any:
    """Числовые значения одной колонки в одной группе, без пропусков."""
    return pd.to_numeric(df.loc[маска, колонка], errors="coerce").dropna().to_numpy()


def _boxes(df: pd.DataFrame, разбивка: Grouping, колонка: str) -> tuple[list[Any], list[str]]:
    """Данные и подписи ящиков одной разбивки; пустые группы пропускаются."""
    данные: list[Any] = []
    подписи: list[str] = []
    for метка in разбивка.order:
        значения = _series(df, разбивка.mask(метка), колонка)
        if len(значения) == 0:
            continue
        данные.append(значения)
        подписи.append(f"{метка}\nn = {len(значения)}")
    return данные, подписи


def _молекул_в_панели(df: pd.DataFrame, разбивка: Grouping, колонка: str) -> int:
    """Сколько молекул показано на панели: объединение групп, а не сумма их размеров.

    У непересекающихся разбивок это одно и то же число, у разбивки по отбору — нет:
    «весь набор» и «топ» вложены, и сумма размеров ящиков считает отобранные дважды.
    На прогонах 6tgu подпись печатала 234 при 195 молекулах, и сверка чисел не нашла
    234 ни в одном поле данных, потому что поголовья такого размера не существует
. Размеры самих групп при этом не теряются: они
    подписаны под каждым ящиком.
    """
    молекулы: set[Any] = set()
    for метка in разбивка.order:
        значения = pd.to_numeric(df.loc[разбивка.mask(метка), колонка], errors="coerce").dropna()
        молекулы.update(значения.index)
    return len(молекулы)


def _заголовок_оси(вид: str) -> str:
    # Не `capitalize()`: он опускает остальные буквы и превращает RMSD в rmsd.
    return вид[0].upper() + вид[1:]


def figure_similarity(df: pd.DataFrame, out_path: Path) -> FigureEntry:
    """Рисунок 1: распределение IFP-Танимото к эталону, по панели на каждую разбивку.

    Панелей одна или две: разбивка по условиям и разбивка по бинам RMSD показывают
    разное и в тексте нужны обе , поэтому появление условий
    не отменяет картинку по RMSD, а добавляет к ней вторую панель.
    """
    разбивки = groupings(df)
    панели = [(разбивка, *_boxes(df, разбивка, "ifp_tanimoto")) for разбивка in разбивки]
    панели = [(разбивка, данные, подписи) for разбивка, данные, подписи in панели if данные]
    if not панели:
        raise FiguresError(
            "Колонка ifp_tanimoto пуста во всех группах: метрики по этим прогонам "
            "не посчитаны"
        )

    fig = Figure(figsize=(7.0 * len(панели), 4.5))
    оси = fig.subplots(1, len(панели), squeeze=False)[0]
    for ось, (разбивка, данные, подписи) in zip(оси, панели, strict=True):
        ящики = ось.boxplot(данные, tick_labels=подписи, showmeans=True, patch_artist=True)
        for ящик in ящики["boxes"]:
            ящик.set_facecolor("#cfe3f7")
            ящик.set_edgecolor("#33587a")
        ось.set_ylabel("IFP-Танимото к эталонной позе")
        ось.set_xlabel(_заголовок_оси(разбивка.name))
        ось.set_ylim(0.0, 1.0)
        ось.grid(axis="y", linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
    # Названия внутри картинки нет намеренно: по положению ФББ (п. 1.8) название
    # рисунка живёт в подписи под ним, и второй экземпляр заголовка разошёлся бы
    # с подписью при первой правке.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    # Число молекул называется по каждой панели отдельно. Общее среднее здесь врало бы:
    # панели строятся по разным подмножествам — RMSD заполнен только у набора поз, —
    # и одно число на всех не совпало бы ни с одной панелью (поймано сверкой чисел).
    виды = "; ".join(разбивка.name for разбивка, _, _ in панели)
    поголовье = tuple(
        (разбивка.name, _молекул_в_панели(df, разбивка, "ifp_tanimoto"))
        for разбивка, _, _ in панели
    )
    объёмы = "; ".join(f"{имя} — {сколько}" for имя, сколько in поголовье)
    return FigureEntry(
        path=out_path,
        title="Сходство отпечатка взаимодействий с эталонным",
        caption=(
            f"Распределение IFP-Танимото к эталонной позе. Панели — разбивки: {виды}. "
            "Ящик — межквартильный размах, линия — медиана, треугольник — среднее, "
            "усы — полтора межквартильных размаха, точки за ними — выбросы. "
            f"Молекул по панелям: {объёмы}. Число молекул каждой группы указано "
            "под её ящиком."
        ),
        run_ids=_run_ids(df),
        panels=поголовье,
    )


def _первая_с_данными(df: pd.DataFrame, разбивки: list[Grouping], колонка: str) -> Grouping:
    """Первая разбивка, в которой у колонки есть значения хотя бы в двух группах.

    Одной группы мало: рисунок сравнивает наборы между собой, и ящик в единственном
    экземпляре ничего не сравнивает. Отбор идёт по наличию **значений**, а не меток:
    метка есть у каждой молекулы, а `qed` заполнен только при `source=diffsbdd`.
    """
    for разбивка in разбивки:
        данные, _ = _boxes(df, разбивка, колонка)
        if len(данные) > 1:
            return разбивка
    raise FiguresError(
        f"Ни одна разбивка не даёт двух групп со значениями {колонка}: "
        f"проверьте, посчитано ли переранжирование (scripts/rerank.py) "
        f"и заполнены ли контрольные метрики у прогона source=diffsbdd"
    )


def figure_control(df: pd.DataFrame, out_path: Path) -> FigureEntry:
    """Рисунок 2: контрольные метрики (QED, SA score, число тяжёлых атомов)."""
    пустые = [
        имя
        for имя, _ in CONTROL_COLUMNS
        if имя not in df.columns or pd.to_numeric(df[имя], errors="coerce").notna().sum() == 0
    ]
    if пустые:
        raise FiguresError(
            f"Контрольные метрики не посчитаны ни у одной молекулы: {', '.join(пустые)}. "
            "На наборе поз (source=local-poses) они вырождены по построению и по "
            "формату таблицы метрик остаются пустыми; рисунок строится по прогону "
            "source=diffsbdd."
        )

    # Одна разбивка, а не все: контрольные метрики сравнивают наборы между собой,
    # и разбивать их ещё и по RMSD незачем — QED позы от её отклонения не зависит.
    # Берётся не первая попавшаяся, а первая, где у контрольных колонок есть значения:
    # разбивка по бинам RMSD заполнена только у набора поз, где эти колонки пусты
    # по формату таблицы метрик, и рисунок из трёх пустых ящиков строился бы молча.
    разбивка = _первая_с_данными(df, groupings(df), CONTROL_COLUMNS[0][0])
    вид = разбивка.name

    fig = Figure(figsize=(10.0, 4.0))
    оси = fig.subplots(1, len(CONTROL_COLUMNS))
    for ось, (имя, заголовок) in zip(оси, CONTROL_COLUMNS, strict=True):
        данные, подписи = _boxes(df, разбивка, имя)
        # `showmeans` как у рисунка 1: два ящичных графика в одной работе, построенные
        # по разным правилам, читатель сравнивает как одинаковые.
        ось.boxplot(данные, tick_labels=подписи, showmeans=True)
        # Подпись именно оси Y, а не заголовок панели: п. 1.9 требует подписанных
        # осей у всех графиков, и «QED» над панелью этому требованию не отвечает.
        ось.set_ylabel(заголовок)
        ось.set_xlabel(_заголовок_оси(вид))
        ось.grid(axis="y", linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    # Панель здесь — метрика, а не разбивка: разбивка одна на все три. Поголовье
    # считается по каждой колонке отдельно, потому что заполнены они независимо:
    # `qed` может быть посчитан там, где `sa_score` пуст, и одно число на три панели
    # врало бы ровно так же, как общее среднее в подписи рисунка 1.
    поголовье = tuple(
        (заголовок, _молекул_в_панели(df, разбивка, имя)) for имя, заголовок in CONTROL_COLUMNS
    )
    объёмы = "; ".join(f"{имя} — {сколько}" for имя, сколько in поголовье)
    return FigureEntry(
        path=out_path,
        title="Контрольные метрики сгенерированных молекул",
        caption=(
            f"Контрольные метрики по группам ({вид}): QED, SA score и число тяжёлых "
            "атомов. Ящик — межквартильный размах, линия — медиана, треугольник — "
            "среднее, усы — полтора межквартильных размаха, точки за ними — выбросы. "
            f"Молекул по панелям: {объёмы}. Число молекул каждой группы указано "
            "под её ящиком. "
            "Ни одна из трёх величин при отборе значимо не меняется; разнообразие "
            "набора, которое отбор снижает, показано не здесь, а в таблице кривой."
        ),
        run_ids=_run_ids(df),
        panels=поголовье,
    )


# Схема пайплайна: подпись блока и координаты его левого нижнего угла. Координаты
# заданы явно, а не вычисляются раскладкой: блоков шесть, и читаемость важнее общности.
_PIPELINE_BLOCKS: Final[tuple[tuple[str, float, float], ...]] = (
    ("Структура KLIFS\n(белок + лиганд)", 0.3, 2.9),
    ("Карман:\n85 позиций", 3.2, 2.9),
    ("Наш IFP\n85 × 7 типов", 6.1, 2.9),
    ("Эталонный IFP\nKLIFS", 6.1, 0.5),
    ("Скор и метрики:\nТанимото, Tversky,\nH-связь с шарниром", 9.0, 1.7),
    ("metrics_per_molecule.csv\nranking.csv\nтаблица «было/стало»", 11.9, 1.7),
)

# Стрелки: откуда, с какой стороны блока, куда и в какую сторону. Стороны заданы
# явно, потому что стрелка от центра к центру ныряет под блок: у диагональных
# переходов подрезка по длине (`shrink`) даёт разный отступ на разных стрелках.
_PIPELINE_ARROWS: Final[tuple[tuple[int, str, int, str], ...]] = (
    (0, "right", 1, "left"),
    (1, "right", 2, "left"),
    (2, "bottom", 4, "left"),
    (3, "top", 4, "left"),
    (4, "right", 5, "left"),
)

_BLOCK_WIDTH: Final[float] = 2.5
_BLOCK_HEIGHT: Final[float] = 1.3
# `boxstyle="round,pad=..."` рисует рамку шире заданного прямоугольника, и якорь
# стрелки без этой поправки оказывается внутри блока, а наконечник — под рамкой.
_BLOCK_PAD: Final[float] = 0.08


def _block_anchor(блок: tuple[str, float, float], сторона: str) -> tuple[float, float]:
    """Точка на границе блока схемы: середина названной стороны."""
    _, x, y = блок
    середина = (x + _BLOCK_WIDTH / 2, y + _BLOCK_HEIGHT / 2)
    стороны = {
        "left": (x - _BLOCK_PAD, середина[1]),
        "right": (x + _BLOCK_WIDTH + _BLOCK_PAD, середина[1]),
        "top": (середина[0], y + _BLOCK_HEIGHT + _BLOCK_PAD),
        "bottom": (середина[0], y - _BLOCK_PAD),
    }
    if сторона not in стороны:
        raise FiguresError(f"Неизвестная сторона блока: {сторона}")
    return стороны[сторона]


def figure_pipeline(out_path: Path) -> FigureEntry:
    """Рисунок 3: схема пути от структуры кармана до таблицы метрик.

    Данных не требует: это схема метода, а не результат. Рисуется кодом по той же
    причине, что и остальные, — чтобы правка пайплайна означала правку одного файла,
    а не поиск исходника картинки в переписке.
    """
    fig = Figure(figsize=(12.0, 4.0))
    ax = fig.subplots()
    ax.set_xlim(0.0, 14.7)
    ax.set_ylim(0.0, 4.6)
    ax.axis("off")

    for подпись, x, y in _PIPELINE_BLOCKS:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                _BLOCK_WIDTH,
                _BLOCK_HEIGHT,
                boxstyle="round,pad=0.08",
                facecolor="#eef4fb",
                edgecolor="#33587a",
            )
        )
        ax.text(
            x + _BLOCK_WIDTH / 2,
            y + _BLOCK_HEIGHT / 2,
            подпись,
            ha="center",
            va="center",
            fontsize=BLOCK_TEXT_PT,
        )

    for откуда, сторона_из, куда, сторона_в in _PIPELINE_ARROWS:
        ax.annotate(
            "",
            xy=_block_anchor(_PIPELINE_BLOCKS[куда], сторона_в),
            xytext=_block_anchor(_PIPELINE_BLOCKS[откуда], сторона_из),
            arrowprops={"arrowstyle": "->", "color": "#33587a", "shrinkA": 2, "shrinkB": 2},
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    return FigureEntry(
        path=out_path,
        title="Схема расчёта отпечатка взаимодействий",
        caption=(
            "Карман KLIFS и поза лиганда дают наш отпечаток, он сравнивается "
            "с эталонным отпечатком KLIFS, и из сравнения получаются метрики "
            "и переранжирование."
        ),
        run_ids=(),
    )


#: Правая граница панели разрешения. У распределения длинный хвост (максимум 35 Å),
#: и без обрезки девять десятых базы сжимаются в один столбик. Хвост за границей
#: в панель не попадает, и подпись это называет, а не умалчивает.
RESOLUTION_AXIS_LIMIT_A: Final[float] = 6.0

#: Столбиков в гистограмме. Тридцать — чтобы шаг по разрешению был около 0.2 Å:
#: мельче становится видна дискретность округления, крупнее пропадает порог.
BASE_HIST_BINS: Final[int] = 30


def figure_base_composition(out_path: Path, klifs_dir: Path = KLIFS_DIR) -> FigureEntry:
    """Рисунок состава базы: распределения разрешения и оценки качества с порогами.

    Строится из выгрузки KLIFS и пакета мишени, а не из прогонов, поэтому `run_ids`
    у него пуст — как у схемы конвейера. Отвечает на вопрос «почему 6tgu» показом,
    а не пересказом процедуры: видно и где стоят пороги, и где в распределении
    оказалась отобранная структура.
    """
    try:
        разрешение, качество = selection_values(klifs_dir)
    except BaseStatsError as ошибка:
        raise FiguresError(f"состав базы не измерить: {ошибка}") from ошибка

    пакет = TARGETS_DIR / MAIN_TARGET_PDB_ID / "target.json"
    if not пакет.is_file():
        raise FiguresError(
            f"нет пакета мишени {пакет}: рисунок показывает место мишени в базе, "
            "и без её разрешения и оценки качества он говорил бы только о базе"
        )
    мишень = json.loads(пакет.read_text(encoding="utf-8"))

    fig = Figure(figsize=(10.0, 4.0))
    панели = fig.subplots(1, 2)
    видимые = [значение for значение in разрешение if значение <= RESOLUTION_AXIS_LIMIT_A]
    for ось, значения, порог, у_мишени, заголовок in (
        (панели[0], видимые, MAX_RESOLUTION, float(мишень["resolution"]), "Разрешение, Å"),
        (панели[1], качество, MIN_QUALITY_SCORE, float(мишень["quality_score"]), "Оценка качества"),
    ):
        ось.hist(значения, bins=BASE_HIST_BINS, color="0.7", edgecolor="0.3")
        # Порог и мишень различаются начертанием, а не цветом: работу печатают
        # на чёрно-белом принтере, и цветовая пара там неразличима.
        ось.axvline(порог, linestyle="--", color="black", label="порог отбора")
        ось.axvline(у_мишени, linestyle=":", color="black", label=f"мишень {MAIN_TARGET_PDB_ID}")
        ось.set_xlabel(заголовок)
        ось.set_ylabel("Структур")
        ось.legend()
        ось.grid(axis="y", linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    return FigureEntry(
        path=out_path,
        title="Состав базы KLIFS и место выбранной мишени",
        caption=(
            f"Распределение разрешения ({len(разрешение)} структур, для которых база "
            f"его указывает) и интегральной оценки качества ({len(качество)} структур) "
            "по всей выгрузке KLIFS. Штриховая линия — порог отбора, точечная — "
            f"выбранная мишень {MAIN_TARGET_PDB_ID}: разрешение "
            f"{мишень['resolution']} Å, оценка качества {мишень['quality_score']}. "
            "Правый хвост распределения разрешения за 6 Å не показан: он сжимал бы "
            "остальную часть распределения в один столбик."
        ),
        run_ids=(),
    )


#: Хвосты имён таблиц уточнения позы: опыт и отрицательный контроль потенциала
#: работы. `all7` — все семь типов взаимодействий, `clash` — с физическим членом
#: в формуле, то есть вариант `М1`. Вариант без `clash` — нулевой вес этого члена,
#: он идёт на рисунок трёх состояний, а не сюда.
REFINEMENT_SUFFIX: Final[str] = "-all7-clash.csv"
CONTROL_SUFFIX: Final[str] = "-all7-control-clash.csv"

#: Прозрачность точек. Скор принимает немного значений (мера дискретна), поэтому
#: молекулы садятся друг на друга, и без прозрачности облако выглядит как десяток
#: точек вместо четырёх сотен.
REFINEMENT_POINT_ALPHA: Final[float] = 0.45

#: Поле вокруг облака точек по обеим осям, в единицах скора. Нужно, чтобы крайние
#: точки не садились на рамку и на диагональ у самого угла.
REFINEMENT_AXIS_MARGIN: Final[float] = 0.03


def _пары_уточнения(таблица: Path) -> tuple[list[float], list[float]]:
    """Скор до и после уточнения из таблицы оптимизации поз."""
    строки = list(csv.DictReader(таблица.open(encoding="utf-8", newline="")))
    return (
        [float(строка["score_before"]) for строка in строки],
        [float(строка["score_after"]) for строка in строки],
    )


def figure_refinement(out_path: Path, results_dir: Path = RESULTS_DIR) -> FigureEntry:
    """Рисунок уточнения позы: скор до против скора после, рядом — контроль.

    Строится из таблиц выхода этапа, а не из папок прогонов: уточнение позы считается
    после прогона и его результат живёт в `results/`. Диагональ показывает
    «без изменения», и содержание подраздела читается прямо с картинки — облако опыта
    лежит над ней, облако контроля на ней и ниже.
    """
    опыты = sorted(
        путь
        for путь in results_dir.glob(f"{POSES_PREFIX}*{REFINEMENT_SUFFIX}")
        if "-control-" not in путь.name
    )
    if not опыты:
        raise FiguresError(
            f"в {results_dir} нет таблиц уточнения позы {POSES_PREFIX}*{REFINEMENT_SUFFIX}: "
            "рисунок показывает, что даёт уточнение, и без них он пуст"
        )

    прогоны: list[str] = []
    до_опыт: list[float] = []
    после_опыт: list[float] = []
    до_контроль: list[float] = []
    после_контроль: list[float] = []
    for таблица in опыты:
        прогон = таблица.name[len(POSES_PREFIX) : -len(REFINEMENT_SUFFIX)]
        контроль = results_dir / f"{POSES_PREFIX}{прогон}{CONTROL_SUFFIX}"
        if not контроль.is_file():
            raise FiguresError(
                f"у прогона {прогон} есть опыт, но нет контроля {контроль.name}: "
                "правая панель показывает именно контроль, и без него рисунок "
                "утверждал бы прирост, не предъявляя того, с чем он сравнивается"
            )
        прогоны.append(прогон)
        пара_до, пара_после = _пары_уточнения(таблица)
        до_опыт += пара_до
        после_опыт += пара_после
        пара_до, пара_после = _пары_уточнения(контроль)
        до_контроль += пара_до
        после_контроль += пара_после

    # `strict=True` намеренно: разная длина колонок означает битую таблицу, и молчаливое
    # обрезание по короткой превратило бы её в правдоподобное число.
    выросло = sum(1 for до, после in zip(до_опыт, после_опыт, strict=True) if после > до)
    выросло_контроль = sum(
        1 for до, после in zip(до_контроль, после_контроль, strict=True) if после > до
    )

    # Обе панели живут в одних пределах, и пределы берутся по данным, а не от нуля:
    # скор нигде не опускается ниже примерно 0.28, и поле от нуля отдало бы половину
    # площади пустому месту, сжав облако в угол.
    все_значения = [*до_опыт, *после_опыт, *до_контроль, *после_контроль]
    низ = min(все_значения) - REFINEMENT_AXIS_MARGIN
    верх = max(все_значения) + REFINEMENT_AXIS_MARGIN

    fig = Figure(figsize=(10.0, 4.8))
    панели = fig.subplots(1, 2, sharex=True, sharey=True)
    for ось, до, после, заголовок in (
        (панели[0], до_опыт, после_опыт, "Потенциал работы"),
        (панели[1], до_контроль, после_контроль, "Отрицательный контроль"),
    ):
        ось.scatter(
            до, после, s=18, color="0.25", alpha=REFINEMENT_POINT_ALPHA, edgecolors="none"
        )
        # Диагональ — линия «скор не изменился». Она же единственная опорная линия
        # рисунка, поэтому чёрная штриховая: цветом её отличать нечем при ч/б печати.
        ось.plot([низ, верх], [низ, верх], linestyle="--", color="black", linewidth=1.0)
        ось.set_xlim(низ, верх)
        ось.set_ylim(низ, верх)
        ось.set_aspect("equal")
        ось.set_title(заголовок)
        ось.set_xlabel("Скор до уточнения")
        ось.grid(linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
    панели[0].set_ylabel("Скор после уточнения")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    return FigureEntry(
        path=out_path,
        title="Уточнение позы по потенциалу: скор до и после",
        caption=(
            f"Каждая точка — молекула, всего {len(до_опыт)} из {len(прогоны)} прогонов "
            "генерации. Штриховая линия — равенство скора до и после уточнения. "
            f"Слева потенциал работы: скор вырос у {выросло} молекул из {len(до_опыт)}. "
            f"Справа отрицательный контроль: {выросло_контроль} из {len(до_контроль)}. "
            "Точки полупрозрачны, потому что мера дискретна и молекулы с одинаковым "
            "скором садятся друг на друга."
        ),
        run_ids=tuple(прогоны),
        panels=(
            ("Потенциал работы", len(до_опыт)),
            ("Отрицательный контроль", len(до_контроль)),
        ),
        facts=(
            ("молекул в опыте", len(до_опыт)),
            ("прогонов", len(прогоны)),
            ("скор вырос в опыте", выросло),
            ("скор вырос в контроле", выросло_контроль),
        ),
    )


#: Помолекулярная выгрузка пересчёта по правилам KLIFS: на ней стоит подраздел
#: об отборе. Имя литералом, а не из `layout`: файл пишет чужой скрипт
#: (`scripts/rescore_klifs_rules.py`), и там оно тоже записано литералом.
RESCORE_MOLECULES_CSV: Final[str] = "klifs_rules_rescore_molecules.csv"


def figure_selection_size(out_path: Path, results_dir: Path = RESULTS_DIR) -> FigureEntry:
    """Рисунок отбора: что он улучшает и чем за это платит.

    Слева Танимото к эталону у отобранных и отбракованных молекул — выигрыш; справа
    число тяжёлых атомов в тех же двух группах — его цена. Обе панели одним взглядом
    отвечают на возражение «отобрали молекулы покрупнее»: оно верно, и рисунок этого
    не прячет, а показывает рядом с выигрышем.

    Топ выбирается тем же `rank_molecules`, что и штатное ранжирование, и отдельно
    внутри каждого прогона: доля берётся от прогона, а не от объединённой выборки.
    """
    таблица = results_dir / RESCORE_MOLECULES_CSV
    if not таблица.is_file():
        raise FiguresError(
            f"нет помолекулярной выгрузки {таблица}: рисунок сравнивает отобранные "
            "молекулы с отбракованными, и без неё сравнивать нечего"
        )
    строки = list(csv.DictReader(таблица.open(encoding="utf-8", newline="")))
    if not строки:
        raise FiguresError(f"выгрузка {таблица} пуста")

    по_прогонам: dict[str, list[dict[str, str]]] = {}
    for строка in строки:
        по_прогонам.setdefault(строка["run_id"], []).append(строка)

    группы: dict[str, dict[str, list[float]]] = {
        "отобранные": {"tanimoto": [], "n_heavy": []},
        "отбракованные": {"tanimoto": [], "n_heavy": []},
    }
    for _, молекулы in sorted(по_прогонам.items()):
        ранжированные = rank_molecules(
            [
                ScoredMolecule(
                    м["mol_id"],
                    float(м["score_all7"]),
                    float(м["score_all7"]) / float(м["n_heavy"]),
                )
                for м in молекулы
            ],
            top_fraction=DEFAULT_TOP_FRACTION,
        )
        выбрано = {с["mol_id"] for с in ранжированные if с["selected"]}
        for м in молекулы:
            куда = "отобранные" if м["mol_id"] in выбрано else "отбракованные"
            группы[куда]["tanimoto"].append(float(м["tanimoto"]))
            группы[куда]["n_heavy"].append(float(м["n_heavy"]))

    fig = Figure(figsize=(10.0, 4.4))
    панели = fig.subplots(1, 2)
    медианы: dict[str, dict[str, float]] = {}
    for ось, колонка, заголовок in (
        (панели[0], "tanimoto", "Танимото к эталону"),
        (панели[1], "n_heavy", "Число тяжёлых атомов"),
    ):
        данные = [группы[имя][колонка] for имя in ("отобранные", "отбракованные")]
        подписи = [
            имя + chr(10) + f"n = {len(группы[имя][колонка])}"
            for имя in ("отобранные", "отбракованные")
        ]
        ось.boxplot(данные, tick_labels=подписи, medianprops={"color": "black"})
        ось.set_ylabel(заголовок)
        ось.grid(axis="y", linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
        медианы[колонка] = {
            имя: float(median(группы[имя][колонка])) for имя in ("отобранные", "отбракованные")
        }
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    всего = len(строки)
    отобрано = len(группы["отобранные"]["tanimoto"])
    return FigureEntry(
        path=out_path,
        title="Отбор по скору: выигрыш в сходстве и его цена в размере",
        caption=(
            f"Молекул {всего} из {len(по_прогонам)} прогонов генерации, отобрано "
            f"{отобрано}. Слева Танимото к эталонному отпечатку: медиана "
            f"{медианы['tanimoto']['отобранные']:.3f} у отобранных против "
            f"{медианы['tanimoto']['отбракованные']:.3f} у отбракованных. Справа число "
            f"тяжёлых атомов: {медианы['n_heavy']['отобранные']:.0f} против "
            f"{медианы['n_heavy']['отбракованные']:.0f} — отбор смещён к более крупным "
            "молекулам, и это его свойство, а не погрешность. Топ выбран внутри каждого "
            "прогона отдельно тем же правилом, что и штатное ранжирование."
        ),
        run_ids=tuple(sorted(по_прогонам)),
        panels=(
            ("Танимото к эталону", всего),
            ("Число тяжёлых атомов", всего),
        ),
        facts=(
            ("молекул", всего),
            ("прогонов", len(по_прогонам)),
            ("отобрано", отобрано),
            ("Танимото, медиана у отобранных", медианы["tanimoto"]["отобранные"]),
            ("Танимото, медиана у отбракованных", медианы["tanimoto"]["отбракованные"]),
            ("тяжёлых атомов, медиана у отобранных", медианы["n_heavy"]["отобранные"]),
            ("тяжёлых атомов, медиана у отбракованных", медианы["n_heavy"]["отбракованные"]),
        ),
    )


#: Помолекулярные точки наборов поз и сводка того же пересчёта. Оба файла пишет
#: `scripts/rescore_klifs_rules.py`; имена литералами по той же причине, что у выгрузки
#: молекул — там они тоже записаны литералом.
RESCORE_POINTS_CSV: Final[str] = "klifs_rules_rescore_points.csv"

#: Вид позы, снятой с кристалла. Ею проверяется верхняя планка меры: если скор
#: не ставит её первой, различать положение нечем.
NATIVE_POSE_KIND: Final[str] = "native"


def _сводка_пересчёта(results_dir: Path) -> dict[str, str]:
    """Показатели пересчёта по правилам KLIFS: «имя показателя» -> «значение»."""
    файл = results_dir / RESCORE_SUMMARY_CSV
    if not файл.is_file():
        return {}
    return {
        строка["показатель"]: строка["значение"]
        for строка in csv.DictReader(файл.open(encoding="utf-8", newline=""))
    }


def figure_pose_scores(out_path: Path, results_dir: Path = RESULTS_DIR) -> FigureEntry:
    """Рисунок различения поз: скор против отклонения позы от кристаллической.

    По панели на мишень, точка — поза набора. Наклон облака и есть ранговая связь,
    а отделённая точка слева вверху — нативная поза. Числа ROC-AUC и Спирмена берутся
    из сводки того же пересчёта по правилам KLIFS, а не считаются здесь заново
    и не берутся из сводки пути ProLIF: это разные величины, и подпись обязана
    называть те же, что таблица.
    """
    таблица = results_dir / RESCORE_POINTS_CSV
    if not таблица.is_file():
        raise FiguresError(
            f"нет точек наборов поз {таблица}: рисунок показывает облако поз, "
            "а в сводках лежат только медианы по наборам. Файл пишет "
            "scripts/rescore_klifs_rules.py"
        )
    строки = list(csv.DictReader(таблица.open(encoding="utf-8", newline="")))
    if not строки:
        raise FiguresError(f"выгрузка точек {таблица} пуста")

    по_мишеням: dict[str, list[dict[str, str]]] = {}
    for строка in строки:
        по_мишеням.setdefault(строка["pdb_id"], []).append(строка)

    сводка = _сводка_пересчёта(results_dir)
    fig = Figure(figsize=(4.4 * len(по_мишеням), 4.2))
    панели = fig.subplots(1, len(по_мишеням), sharey=True, squeeze=False)[0]
    описания: list[str] = []
    for ось, (мишень, точки) in zip(панели, sorted(по_мишеням.items()), strict=True):
        обычные = [т for т in точки if т["pose_kind"] != NATIVE_POSE_KIND]
        нативные = [т for т in точки if т["pose_kind"] == NATIVE_POSE_KIND]
        ось.scatter(
            [float(т["rmsd_to_ref"]) for т in обычные],
            [float(т["score_all7"]) for т in обычные],
            s=14,
            color="0.35",
            alpha=REFINEMENT_POINT_ALPHA,
            edgecolors="none",
        )
        # Нативная поза — полый кружок с чёрным контуром: она одна против сотен точек,
        # и заливкой её на облаке не найти, а цветом при ч/б печати не отличить.
        ось.scatter(
            [float(т["rmsd_to_ref"]) for т in нативные],
            [float(т["score_all7"]) for т in нативные],
            s=70,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            label="нативная поза",
        )
        ось.set_title(мишень)
        ось.set_xlabel("Отклонение от кристаллической позы, Å")
        ось.grid(linestyle=":", alpha=0.5)
        if нативные:
            ось.legend(loc="lower left")
        _apply_font_sizes(ось)
        auc = сводка.get(f"{мишень}, все семь типов: ROC-AUC, медиана", "")
        ро = сводка.get(f"{мишень}, все семь типов: Спирмен с отклонением, медиана", "")
        описания.append(
            f"{мишень} — поз {len(точки)}"
            + (f", ROC-AUC {auc}" if auc else "")
            + (f", Спирмен {ро}" if ро else "")
        )
    панели[0].set_ylabel("Скор по правилам KLIFS")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    прогоны = tuple(sorted({строка["run_id"] for строка in строки}))
    return FigureEntry(
        path=out_path,
        title="Скор различает положение лиганда в кармане",
        caption=(
            f"Каждая точка — поза из {len(прогоны)} наборов по {len(по_мишеням)} мишеням, "
            f"всего {len(строки)}. Полый кружок — нативная поза, снятая с кристалла. "
            "По медианам наборов: " + "; ".join(описания) + ". Медианы взяты из сводки "
            "того же пересчёта по правилам KLIFS, а не посчитаны по объединённому облаку."
        ),
        run_ids=прогоны,
        panels=tuple((мишень, len(точки)) for мишень, точки in sorted(по_мишеням.items())),
        facts=(
            ("точек всего", len(строки)),
            ("наборов поз", len(прогоны)),
            ("мишеней", len(по_мишеням)),
        ),
    )


#: Сколько позиций кармана размечает KLIFS. Раскладка 595-битной строки — блоками
#: по позициям, поэтому позиция это `индекс // 7 + 1`, тип — `индекс % 7`.
KLIFS_POSITIONS: Final[int] = 85


def figure_ifp_map(out_path: Path, targets_dir: Path = TARGETS_DIR) -> FigureEntry:
    """Карта отпечатка: 85 позиций кармана против 7 типов взаимодействия.

    Одна картинка показывает три вещи сразу: как устроен позиционный отпечаток,
    насколько он разрежен и в чём именно расходится наш расчёт с эталоном KLIFS.
    Биты считаются на лету из пакета мишени — на диске их нет ни в каком виде,
    в файлах выхода этапа лежат только счётчики расхождений.
    """
    пакет = targets_dir / MAIN_TARGET_PDB_ID / "target.json"
    if not пакет.is_file():
        raise FiguresError(f"нет пакета мишени {пакет}: отпечаток не из чего считать")
    эталон = json.loads(пакет.read_text(encoding="utf-8")).get("klifs_ifp_bits")
    if not эталон:
        raise FiguresError(
            f"у пакета {пакет} нет эталонного отпечатка KLIFS: карта сравнивает "
            "наш расчёт с эталоном, и без второй половины сравнивать не с чем"
        )
    try:
        наши = fingerprint_by_klifs_rules(пакет)
        сравнение = compare_with_reference(MAIN_TARGET_PDB_ID, наши, эталон)
    except KlifsRulesError as ошибка:
        raise FiguresError(f"отпечаток не посчитан: {ошибка}") from ошибка

    # Три состояния бита. Различаются формой и заливкой, а не цветом: рисунок
    # печатают чёрно-белым, и заливка против контура там читается, а оттенок нет.
    состояния: dict[str, list[tuple[int, int]]] = {"совпало": [], "пропущено": [], "лишнее": []}
    for индекс, (наш, эталонный) in enumerate(zip(наши, эталон, strict=True)):
        позиция = индекс // len(KLIFS_INTERACTION_TYPES) + 1
        тип = индекс % len(KLIFS_INTERACTION_TYPES)
        if наш == "1" and эталонный == "1":
            состояния["совпало"].append((позиция, тип))
        elif наш == "0" and эталонный == "1":
            состояния["пропущено"].append((позиция, тип))
        elif наш == "1" and эталонный == "0":
            состояния["лишнее"].append((позиция, тип))

    fig = Figure(figsize=(12.0, 3.6))
    ось = fig.subplots()
    for имя, маркер, заливка in (
        ("совпало", "s", "black"),
        ("пропущено", "x", "black"),
        ("лишнее", "s", "none"),
    ):
        точки = состояния[имя]
        if not точки:
            continue
        ось.scatter(
            [позиция for позиция, _ in точки],
            [тип for _, тип in точки],
            marker=маркер,
            s=28,
            facecolors=заливка,
            edgecolors="black",
            linewidths=1.0,
            label=имя,
        )
    ось.set_yticks(range(len(KLIFS_INTERACTION_TYPES)))
    ось.set_yticklabels(KLIFS_INTERACTION_TYPES)
    ось.set_xlim(0, KLIFS_POSITIONS + 1)
    ось.set_ylim(-0.6, len(KLIFS_INTERACTION_TYPES) - 0.4)
    ось.set_xlabel("Позиция кармана в разметке KLIFS")
    ось.set_ylabel("Тип взаимодействия")
    ось.grid(linestyle=":", alpha=0.4)
    ось.legend(loc="upper right", ncol=3)
    _apply_font_sizes(ось)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    совпало = len(состояния["совпало"])
    расхождений = len(сравнение.differences)
    return FigureEntry(
        path=out_path,
        title=f"Карта отпечатка мишени {MAIN_TARGET_PDB_ID}: наш расчёт против эталона KLIFS",
        caption=(
            f"Сетка 85 позиций кармана на {len(KLIFS_INTERACTION_TYPES)} типов "
            f"взаимодействия — {KLIFS_POSITIONS * len(KLIFS_INTERACTION_TYPES)} бит, "
            f"из которых у эталона заполнено {сравнение.reference_total}. Отпечаток "
            f"разрежен: занято меньше трёх процентов сетки. Совпало с эталоном "
            f"{совпало} бит, расхождений {расхождений}."
        ),
        run_ids=(),
        facts=(
            ("бит в сетке", KLIFS_POSITIONS * len(KLIFS_INTERACTION_TYPES)),
            ("бит у эталона", сравнение.reference_total),
            ("совпало", совпало),
            ("расхождений", расхождений),
        ),
    )


#: Три состояния позы, которые сравнивает подраздел о независимых мерах: как вышло
#: из модели, после уточнения без физического члена и после уточнения потенциалом
#: работы. Порядок значим — он же порядок столбцов в группе.
POSE_STATES: Final[tuple[tuple[str, str], ...]] = (
    ("source", "исходные"),
    ("all7", "нулевой вес"),
    ("all7-clash", "потенциал работы"),
)

#: Ширина одного столбца в группе, в долях шага между группами.
STATE_BAR_WIDTH: Final[float] = 0.26


def _измерение(строки: tuple[Any, ...], вариант: str, метрика: str) -> float | None:
    """Одно число реестра измерений или `None`, если его там нет."""
    найдено = [с for с in select(строки, variant=вариант) if с.metric == метрика]
    return float(найдено[0].value) if найдено else None


def figure_states(out_path: Path, measurements_csv: Path = MEASUREMENTS_CSV) -> FigureEntry:
    """Рисунок трёх состояний: физичность позы и энергия связывания независимыми мерами.

    Слева доля поз, прошедших все применимые проверки PoseBusters, справа медиана
    энергии Vina. Провал среднего столбца и совпадение крайних — всё содержание
    подраздела: потенциал не портит того, чего не оптимизирует, а физический член
    в его формуле нужен не для красоты.
    """
    try:
        строки = read_measurements(measurements_csv)
    except MeasurementError as ошибка:
        raise FiguresError(f"реестр измерений не прочитан: {ошибка}") from ошибка

    прогоны: dict[str, str] = {}
    for строка in строки:
        if строка.measurement in {"physics", "docking"} and ":" in строка.variant:
            прогон = строка.variant.split(":", 1)[0]
            прогоны.setdefault(прогон, строка.target)

    собрано: list[tuple[str, str, dict[str, float | None], dict[str, float | None]]] = []
    пропуски: list[str] = []
    for прогон, мишень in sorted(прогоны.items(), key=lambda пара: (пара[1], пара[0])):
        физичность = {
            метка: _измерение(строки, f"{прогон}:{состояние}", "valid_all")
            for состояние, метка in POSE_STATES
        }
        энергия = {
            метка: _измерение(строки, f"{прогон}:{состояние}", "vina_score_median")
            for состояние, метка in POSE_STATES
        }
        нет_чисел = [
            f"{прогон}:{состояние} ({мера})"
            for состояние, метка in POSE_STATES
            for мера, значения in (("физичность", физичность), ("энергия", энергия))
            if значения[метка] is None
        ]
        if нет_чисел:
            пропуски += нет_чисел
            continue
        собрано.append((прогон, мишень, физичность, энергия))

    # Столбец нулевой высоты читается как «мера равна нулю», а не «не измерено»,
    # и рисунок на неполных данных врал бы молча: у прогона без физичности левая
    # панель показывала бы провал, которого не измеряли. Пустая гистограмма хуже
    # отказа — то же правило, что у состава базы.
    if not собрано:
        raise FiguresError(
            f"в реестре {measurements_csv} нет полного набора трёх состояний ни у одного "
            f"прогона, не хватает: {', '.join(sorted(set(пропуски))[:6])}"
            + (" и других" if len(set(пропуски)) > 6 else "")
            + ". Физичность кладёт scripts/check_physics.py --target, энергию — "
            "scripts/score_docking.py"
        )

    подписи = [
        мишень + chr(10) + прогон.rsplit("-", 1)[-1] for прогон, мишень, _, _ in собрано
    ]
    места = list(range(len(собрано)))
    fig = Figure(figsize=(11.0, 4.4))
    панели = fig.subplots(1, 2)
    физичности = [запись[2] for запись in собрано]
    энергии = [запись[3] for запись in собрано]
    for ось, ряды, заголовок in (
        (панели[0], физичности, "Доля валидных поз (PoseBusters)"),
        (панели[1], энергии, "Медиана энергии Vina, ккал/моль"),
    ):
        for номер, (_, метка) in enumerate(POSE_STATES):
            высоты = [ряд[метка] for ряд in ряды]
            ось.bar(
                [место + (номер - 1) * STATE_BAR_WIDTH for место in места],
                [0.0 if высота is None else высота for высота in высоты],
                width=STATE_BAR_WIDTH,
                # Три столбца различаются заливкой от светлой к тёмной: при ч/б печати
                # это единственный различимый порядок, и он совпадает с порядком стадий.
                color=("0.85", "0.55", "0.2")[номер],
                edgecolor="black",
                linewidth=0.6,
                label=метка,
            )
        ось.set_xticks(места)
        ось.set_xticklabels(подписи)
        ось.set_ylabel(заголовок)
        ось.grid(axis="y", linestyle=":", alpha=0.5)
        _apply_font_sizes(ось)
    # Легенда одна на обе панели и вынесена над ними: внутри осей она закрывала
    # столбцы — слева верхние, справа нижние, потому что энергия отрицательна.
    ручки, имена = панели[0].get_legend_handles_labels()
    fig.legend(ручки, имена, loc="upper center", ncol=len(POSE_STATES), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI)

    неполных = len({пропуск.split(":")[0] for пропуск in пропуски})
    хвост = (
        ""
        if not пропуски
        else f" Прогонов без полного набора трёх состояний: {неполных}, "
        "они на рисунок не попали."
    )
    return FigureEntry(
        path=out_path,
        title="Три состояния позы: физичность и энергия связывания",
        caption=(
            f"Прогонов {len(собрано)}, в каждой группе три столбца — как вышло из модели, "
            "после уточнения без физического члена и после уточнения потенциалом работы. "
            "Слева доля поз, прошедших все применимые проверки PoseBusters, справа "
            "медиана энергии Vina (чем меньше, тем прочнее связывание)." + хвост
        ),
        run_ids=tuple(прогон for прогон, _, _, _ in собрано),
        panels=tuple(
            (
                мишень + " " + прогон.rsplit("-", 1)[-1],
                int(_измерение(строки, f"{прогон}:source", "molecules") or 0),
            )
            for прогон, мишень, _, _ in собрано
        ),
        facts=(("прогонов", len(собрано)),),
    )


def write_figure_facts(entries: Sequence[FigureEntry], out_dir: Path) -> Path:
    """Пишет `facts.csv`: числа, которые называет подпись каждого рисунка.

    Опись нужна затем же, зачем `panels.csv`: подпись уходит в текст дословно, а число
    в тексте обязано иметь машинный источник. 19.09 сверка нашла в подписях
    два числа, которых не было нигде, — поголовье точек облака и число молекул
    с выросшим скором.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    файл = out_dir / FIGURE_FACTS_CSV
    with файл.open("w", encoding="utf-8", newline="") as поток:
        писатель = csv.writer(поток, lineterminator=CSV_EOL)
        писатель.writerow(("figure", "fact", "value"))
        for номер, рисунок in enumerate(entries, start=1):
            for имя, значение in рисунок.facts:
                писатель.writerow((номер, имя, значение))
    return файл


def write_panel_sizes(entries: Sequence[FigureEntry], out_dir: Path) -> Path:
    """`panels.csv` рядом с рисунками: сколько молекул на каждой панели.

    Файл производный, как `before_after.csv`: собирается из тех же прогонов той же
    командой. Нужен сверке чисел — иначе поголовье панели из подписи
    подтверждать нечем, и оно ловит первое попавшееся совпадение в прозе.
    """
    путь = out_dir / PANELS_CSV
    путь.parent.mkdir(parents=True, exist_ok=True)
    with путь.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл, lineterminator=CSV_EOL)
        писатель.writerow(("figure", "panel", "molecules"))
        for номер, entry in enumerate(entries, start=1):
            for имя, сколько in entry.panels:
                писатель.writerow((номер, имя, сколько))
    return путь


#: Библиотеки, от версии которых зависит **вид** рисунка. Перечень короткий
#: и не пересекается с `runs.NUMERIC_LIBRARIES` намеренно: тот перечисляет то,
#: от чего зависят числа, и смешивать два вопроса в одном списке значило бы
#: получить список, который не отвечает ни на один.
FIGURE_LIBRARIES: Final[tuple[str, ...]] = ("matplotlib",)


def write_environment(out_dir: Path) -> Path:
    """`environment.json` рядом с рисунками: чем они нарисованы.

    Отвечает на вопрос, который паспорт прогона не покрывает и покрывать не должен:
    `matplotlib` не участвует в производстве ни одного числа, но вид рисунка от его
    версии зависит. Для чисел это закрыто полем `environment`
    в паспорте; для рисунков его закрывает этот файл.

    Версии читаются у установленных пакетов, а не из `pyproject.toml`: в файле стоит
    ограничение, а нарисовано тем, что действительно стоит в образе.

    Пишется при полной сборке, тем же условием, что `README.md` и `panels.csv`:
    файл описывает, чем нарисован **набор**. После `--only` он говорил бы о версии,
    которой нарисован один рисунок из трёх, — это хуже отсутствия записи.
    """
    from importlib import metadata

    окружение: dict[str, str] = {"python": platform.python_version()}
    for имя in FIGURE_LIBRARIES:
        try:
            окружение[имя] = metadata.version(имя)
        except metadata.PackageNotFoundError:
            # Тот же выбор, что в `runs.library_versions`: отсутствующий пакет
            # пропускается, а не пишется пустым. Рисунок без matplotlib не собрался бы
            # вовсе, поэтому ветка означает «файл пишут не тем кодом», а не пропуск.
            continue

    out_dir.mkdir(parents=True, exist_ok=True)
    путь = out_dir / FIGURES_ENVIRONMENT_JSON
    путь.write_text(json.dumps(окружение, ensure_ascii=False, indent=2), encoding="utf-8")
    return путь


def write_readme(entries: Sequence[FigureEntry], out_dir: Path) -> Path:
    """`README.md` рядом с рисунками: подпись и `run_id` данных для каждого файла.

    По самому файлу картинки нельзя сказать, из каких прогонов она собрана,
    а каждое число и каждый рисунок должны сводиться к прогону.
    """
    строки = [
        "# Рисунки раздела «Результаты и обсуждение»",
        "",
        "Собираются командой `python scripts/make_figures.py`.",
        "Руками не редактируются: файл переписывается при каждой сборке.",
        "",
        f"Чем нарисовано — `{FIGURES_ENVIRONMENT_JSON}` рядом с этим файлом.",
        "Версия `matplotlib` в паспорт прогона не пишется намеренно:",
        "она не участвует в счёте чисел, но задаёт вид рисунка.",
        "",
        "Подписи оформлены по положению ФББ МГУ (Приложение 3, п. 1.8),",
        "разбор требований — `docs/thesis-requirements.md`:",
        "",
        "- подпись ставится **под** рисунком целиком, как она приведена ниже;",
        "- в тексте на каждый рисунок должна быть ссылка вида «(Рисунок N)»;",
        "- нумерация сквозная и присвоена **по порядку собранных рисунков**, а не по",
        "  имени файла: если рисунок контрольных метрик не построен (нет прогона",
        "  `source=diffsbdd`), схема получает номер 2, а не 3 — пропуск номера в тексте",
        "  был бы дефектом оформления.",
        "",
    ]
    for номер, entry in enumerate(entries, start=1):
        источник = ", ".join(entry.run_ids) if entry.run_ids else "данные не используются"
        строки += [
            f"## Рисунок {номер} — файл `{entry.path.name}`",
            "",
            f"**Подпись под рисунком.** {full_caption(entry, номер)}",
            "",
            f"**Ссылка в тексте:** (Рисунок {номер})",
            "",
            f"**Данные:** {источник}",
            "",
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    путь = out_dir / FIGURES_README
    путь.write_text("\n".join(строки), encoding="utf-8")
    return путь
