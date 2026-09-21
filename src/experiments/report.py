"""Сборка таблицы «было/стало» из папок прогонов.

Вход — папки прогонов: паспорт (`run.json`), таблица метрик (`metrics_per_molecule.csv`)
и таблица ранжирования (`ranking.csv`); выход — `before_after.md` и `before_after.csv`.

Колонка таблицы — это условие эксперимента, а не прогон: `condition` живёт в
`metrics_per_molecule.csv`, и один прогон может дать несколько условий. В шапке колонки
всегда стоят и `run_id`, и `source`: без них число в тексте курсовой невозможно проверить.

Доверительные интервалы здесь не считаются. Бутстрэп и Манна–Уитни считаются отдельно,
она производит `metrics_summary.csv`; отчёт берёт границы оттуда, если файл есть,
и печатает значение без ДИ, если его нет. Своя реализация бутстрэпа была бы
дублированием чужого кода.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from rdkit import Chem

from evaluation.control import ControlError, diversity, uniqueness
from experiments.layout import (
    BEFORE_AFTER_CSV,
    BEFORE_AFTER_MD,
    CSV_EOL,
    METRICS_CSV,
    RANKING_CSV,
    RUN_JSON,
    SOURCE_LOCAL_POSES,
    SUMMARY_CSV,
)

# Как метрика сворачивается в одно число (`docs/metrics.md`, раздел 5): доли — среднее
# нулей и единиц, распределения — медиана с межквартильным размахом. Третий вид —
# метрика набора (`set`): uniqueness и diversity не имеют значения у отдельной молекулы,
# поэтому колонок в таблице метрик у них нет, и считаются они по столбцу `smiles`.
Aggregation = Literal["share", "distribution", "set"]

# Ячейка, которую нечем заполнить. Пустая строка выглядела бы как ноль.
EMPTY_CELL: Final[str] = "—"


@dataclass(frozen=True)
class MetricSpec:
    """Одна строка отчёта: откуда берётся колонка и как её агрегировать."""

    column: str
    title: str
    aggregation: Aggregation
    # Метрика вырождена на наборе поз: одна молекула в ста конформациях даёт одно
    # значение QED, размноженное сто раз. Такие строки при
    # `source=local-poses` не выводятся вовсе, а не выводятся пустыми.
    diffsbdd_only: bool = False
    # `ifp_score` и `ifp_score_norm` живут в `ranking.csv`, остальное — в таблице метрик.
    from_ranking: bool = False


# Порядок строк отчёта: целевые метрики выше контрольных — таблицу читают сверху вниз,
# и первым должен стоять ответ на вопрос работы.
REPORT_METRICS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec("ifp_tanimoto", "IFP-Танимото к эталону", "distribution"),
    MetricSpec("ifp_tversky", "IFP-Tversky (M -> R)", "distribution"),
    MetricSpec("hinge_hbond", "Доля молекул с H-связью с шарниром", "share"),
    MetricSpec("ifp_score", "IFP-скор (без HYD)", "distribution", from_ranking=True),
    MetricSpec("ifp_score_norm", "IFP-скор на тяжёлый атом", "distribution", from_ranking=True),
    MetricSpec("valid", "Validity", "share"),
    MetricSpec("connected", "Connectivity", "share"),
    MetricSpec("uniqueness", "Uniqueness", "set", diffsbdd_only=True),
    MetricSpec("posebusters_valid", "PoseBusters valid", "share"),
    MetricSpec("qed", "QED", "distribution", diffsbdd_only=True),
    MetricSpec("sa_score", "SA score", "distribution", diffsbdd_only=True),
    MetricSpec("n_heavy_atoms", "Число тяжёлых атомов", "distribution", diffsbdd_only=True),
    MetricSpec("diversity", "Diversity", "set", diffsbdd_only=True),
    MetricSpec("vina_score", "Vina score", "distribution"),
    MetricSpec("ligand_efficiency", "Ligand efficiency", "distribution"),
)


@dataclass(frozen=True)
class Column:
    """Колонка отчёта: одно условие одного прогона вместе с его молекулами."""

    condition: str
    run_id: str
    source: str
    molecules: pd.DataFrame
    summary: pd.DataFrame | None

    @property
    def header(self) -> str:
        return f"{self.condition} ({self.run_id}, {self.source})"


def load_run(run_dir: Path) -> list[Column]:
    """Читает папку прогона и возвращает по колонке на каждое условие внутри неё.

    Прогон без `run.json` считается несуществующим: это ошибка,
    а не повод пропустить папку молча.
    """
    passport_path = run_dir / RUN_JSON
    if not passport_path.is_file():
        raise FileNotFoundError(
            f"В {run_dir} нет {RUN_JSON}: прогон без паспорта считается несуществующим "
            "и в таблицы не идёт (формат паспорта прогона)"
        )
    passport = json.loads(passport_path.read_text(encoding="utf-8"))
    for field in ("run_id", "source"):
        if not passport.get(field):
            raise ValueError(f"В {passport_path} не заполнено обязательное поле '{field}'")

    molecules = _read_csv(run_dir / METRICS_CSV)
    missing = {"mol_id", "condition"} - set(molecules.columns)
    if missing:
        raise ValueError(
            f"{run_dir / METRICS_CSV} не соответствует формату таблицы метрик: "
            f"нет колонок {sorted(missing)}"
        )

    ranking_path = run_dir / RANKING_CSV
    if ranking_path.is_file():
        ranking = _read_csv(ranking_path)
        if "mol_id" not in ranking.columns:
            raise ValueError(
                f"{ranking_path}: таблица ранжирования без колонки 'mol_id'"
            )
        molecules = molecules.merge(ranking, on="mol_id", how="left", validate="one_to_one")

    summary_path = run_dir / SUMMARY_CSV
    summary = _read_csv(summary_path) if summary_path.is_file() else None

    run_id, source = str(passport["run_id"]), str(passport["source"])
    named = molecules["condition"].notna() & (molecules["condition"].astype(str).str.strip() != "")
    if named.any():
        группы = list(molecules[named].groupby("condition", sort=False))
        # Одно условие на весь прогон не отменяет сравнения «до отбора и после»:
        # ради него сделан уровень A, и до появления `condition` таблица строилась
        # именно так. Несколько условий делятся сами — разбивать их ещё и по `selected`
        # значит удвоить колонки, ничего не добавив.
        if len(группы) == 1:
            условие, frame = группы[0]
            return _split_by_selection(frame, passport, summary, condition=str(условие))
        return [
            Column(
                condition=str(condition),
                run_id=run_id,
                source=source,
                molecules=frame,
                summary=summary,
            )
            for condition, frame in группы
        ]
    return _split_by_selection(molecules, passport, summary)


def _split_by_selection(
    molecules: pd.DataFrame,
    passport: dict[str, Any],
    summary: pd.DataFrame | None,
    condition: str | None = None,
) -> list[Column]:
    """Строит «было» и «стало» из колонки `selected`.

    Прогон, сделанный одной командой, приходит без `condition` в паспорте: условие
    там задавать нечем, оно возникает только после переранжирования. Сравнивать
    в таблице надо одно и то же множество молекул до отбора и после — это и есть
    смысл переранжирования, ради которого сделан уровень A.

    `condition` задаётся, когда прогон всё-таки помечен одним условием: тогда
    колонки называются им, а не источником. Без метки подписью служит `source` —
    так таблица выглядела до того, как условия начали проставлять.

    Без `ranking.csv` прогон даёт одну колонку: отбирать нечем, и притворяться,
    что «стало» посчитано, нельзя.
    """
    run_id, source = str(passport["run_id"]), str(passport["source"])
    подпись_условия = condition if condition is not None else source
    колонки = [
        Column(
            condition=подпись_условия,
            run_id=run_id,
            source=source,
            molecules=molecules,
            summary=summary,
        )
    ]
    if "selected" not in molecules.columns:
        return колонки

    отобранные = molecules[molecules["selected"].fillna(0).astype(int) == 1]
    if отобранные.empty or len(отобранные) == len(molecules):
        return колонки

    доля = passport.get("ranking", {}).get("top_fraction")
    подпись = (
        f"{подпись_условия}, топ {доля:.0%}"
        if isinstance(доля, (int, float))
        else f"{подпись_условия}, топ"
    )
    колонки.append(
        Column(
            condition=подпись,
            run_id=run_id,
            source=source,
            molecules=отобранные,
            summary=summary,
        )
    )
    return колонки


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла {path}")
    return pd.read_csv(path)


def _interval(column: Column, metric: MetricSpec) -> tuple[float, float] | None:
    """Границы доверительных интервалов из `metrics_summary.csv`.

    **Условие здесь — подпись колонки, а не поле формата таблицы метрик.** `column.condition`
    строит `_split_by_selection`: «diffsbdd», «diffsbdd, топ 20%». Колонка `condition`
    из паспорта прогона пуста во всех шести прогонах, и `metrics_summary.csv`,
    заполненный по ней, не совпадёт с подписями никогда — интервалы просто не
    подтянутся, а выглядеть это будет как «файла нет».

    Отсутствие колонки `condition` и несколько строк на одно условие — отказ,
    а не «берём первую строку». До 30.08 здесь стоял молчаливый пропуск фильтра:
    файл без `condition` давал **один и тот же интервал всем колонкам таблицы**,
    и заметить это в готовом документе было нечем.
    """
    if column.summary is None:
        return None
    low_column, high_column = f"{metric.column}_lo", f"{metric.column}_hi"
    if low_column not in column.summary.columns or high_column not in column.summary.columns:
        return None
    if "condition" not in column.summary.columns:
        raise ValueError(
            f"в {SUMMARY_CSV} нет колонки condition, а границы ДИ в нём есть. "
            "Без неё интервал нельзя отнести к условию, и все колонки таблицы получили бы "
            f"один и тот же: добавьте condition со значениями вида {column.condition!r}"
        )

    rows = column.summary[column.summary["condition"] == column.condition]
    if rows.empty:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"в {SUMMARY_CSV} условию {column.condition!r} отвечает не одна строка, а {len(rows)}; "
            "какую из них брать, файл не говорит"
        )

    low, high = rows.iloc[0][low_column], rows.iloc[0][high_column]
    if pd.isna(low) or pd.isna(high):
        return None
    return float(low), float(high)


@dataclass(frozen=True)
class Cell:
    """Значение метрики в одном условии; `value is None` означает «считать нечего»."""

    value: float | None
    n: int
    iqr_low: float | None = None
    iqr_high: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None

    def render(self) -> str:
        if self.value is None:
            return EMPTY_CELL
        text = f"{self.value:.3g}"
        if self.ci_low is not None and self.ci_high is not None:
            text += f" [ДИ {self.ci_low:.3g}–{self.ci_high:.3g}]"
        if self.iqr_low is not None and self.iqr_high is not None:
            text += f" (IQR {self.iqr_low:.3g}–{self.iqr_high:.3g})"
        return text


def _compute_set_cell(column: Column, metric: MetricSpec) -> Cell:
    """Считает метрику набора по столбцу `smiles` условия.

    Своей реализации здесь нет намеренно: те же `uniqueness` и `diversity`
    из `evaluation.control` считают эти числа, и две реализации одной
    формулы разошлись бы в первом же спорном месте — например, в том, снимаются ли
    явные водороды перед канонизацией.
    """
    if "smiles" not in column.molecules.columns:
        return Cell(value=None, n=0)
    строки = [str(значение) for значение in column.molecules["smiles"].dropna()]
    молекулы = [Chem.MolFromSmiles(строка) for строка in строки]
    if not молекулы:
        return Cell(value=None, n=0)
    try:
        if metric.column == "uniqueness":
            доля, _, знаменатель = uniqueness(молекулы)
            return Cell(value=доля, n=знаменатель)
        return Cell(value=diversity(молекулы), n=len(молекулы))
    except ControlError:
        # Набор, на котором метрика не определена (одна валидная молекула), — это
        # пустая ячейка, а не ноль: ноль читался бы как «разнообразия нет».
        return Cell(value=None, n=len(молекулы))


def metric_values(column: Column, metric: MetricSpec) -> pd.Series:
    """Числовые значения метрики в условии; непосчитанные молекулы отброшены.

    Публичная потому, что по этому же множеству считаются бутстрэп-интервалы
    (`evaluation.stats`). Считай их разными отборами — и в таблице встретятся
    медиана по одному набору молекул с интервалом по другому, причём совпадать
    они будут почти всегда, а расходиться на тех прогонах, где часть метрик
    не посчитана.
    """
    if metric.column not in column.molecules.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(column.molecules[metric.column], errors="coerce").dropna()


def compute_cell(column: Column, metric: MetricSpec) -> Cell:
    """Сворачивает колонку метрики в одно число: долю, медиану с IQR или метрику набора."""
    if metric.aggregation == "set":
        return _compute_set_cell(column, metric)
    values = metric_values(column, metric)
    if values.empty:
        return Cell(value=None, n=0)

    interval = _interval(column, metric)
    ci_low, ci_high = interval if interval else (None, None)

    if metric.aggregation == "share":
        return Cell(value=float(values.mean()), n=len(values), ci_low=ci_low, ci_high=ci_high)
    return Cell(
        value=float(values.median()),
        n=len(values),
        iqr_low=float(values.quantile(0.25)),
        iqr_high=float(values.quantile(0.75)),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def _is_applicable(metric: MetricSpec, column: Column) -> bool:
    return not (metric.diffsbdd_only and column.source == SOURCE_LOCAL_POSES)


def build_table(columns: list[Column]) -> tuple[pd.DataFrame, list[str]]:
    """Строит таблицу «метрики × условия» и список строк, не вошедших в неё.

    Строка не выводится, если метрика неприменима во всех условиях сразу (вырождена
    на наборе поз) или ни в одном условии не посчитана. Выброшенное не исчезает молча:
    оно возвращается вторым значением и печатается под таблицей.
    """
    if not columns:
        raise ValueError("Нечего собирать: не передано ни одного прогона")

    rows: dict[str, list[str]] = {}
    dropped: list[str] = []
    for metric in REPORT_METRICS:
        cells = [
            compute_cell(column, metric) if _is_applicable(metric, column) else Cell(None, 0)
            for column in columns
        ]
        if not any(_is_applicable(metric, column) for column in columns):
            dropped.append(f"{metric.title}: вырождена при source={SOURCE_LOCAL_POSES}")
            continue
        if all(cell.value is None for cell in cells):
            dropped.append(f"{metric.title}: не посчитана ни в одном условии")
            continue
        rows[metric.title] = [cell.render() for cell in cells]

    table = pd.DataFrame.from_dict(
        rows, orient="index", columns=[column.header for column in columns]
    )
    table.index.name = "Метрика"
    return table, dropped


def to_long_frame(columns: list[Column]) -> pd.DataFrame:
    """Те же числа машинно: одна строка на пару «метрика × условие»."""
    records = []
    for metric in REPORT_METRICS:
        for column in columns:
            if not _is_applicable(metric, column):
                continue
            cell = compute_cell(column, metric)
            if cell.value is None:
                continue
            records.append(
                {
                    "metric": metric.column,
                    "aggregation": metric.aggregation,
                    "condition": column.condition,
                    "run_id": column.run_id,
                    "source": column.source,
                    "n": cell.n,
                    "value": cell.value,
                    "iqr_low": cell.iqr_low,
                    "iqr_high": cell.iqr_high,
                    "ci_low": cell.ci_low,
                    "ci_high": cell.ci_high,
                }
            )
    return pd.DataFrame.from_records(records)


def _markdown_table(table: pd.DataFrame) -> str:
    """Рисует markdown-таблицу вручную.

    `DataFrame.to_markdown` требует пакет `tabulate`, которого нет в зависимостях:
    заводить его ради трёх строк форматирования дороже, чем написать их.
    """
    header = [str(table.index.name or "")] + [str(name) for name in table.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for title, row in table.iterrows():
        lines.append("| " + " | ".join([str(title)] + [str(value) for value in row]) + " |")
    return "\n".join(lines)


def render_markdown(table: pd.DataFrame, columns: list[Column], dropped: list[str]) -> str:
    """Собирает `before_after.md`: таблицу и происхождение каждого числа в ней."""
    lines = [
        "# Таблица «было/стало»",
        "",
        "Собрано `scripts/build_report.py`. Те же числа машинно — "
        "в `before_after.csv`.",
        "",
        "Источники колонок:",
        "",
    ]
    lines += [
        f"- **{column.condition}** — `run_id={column.run_id}`, `source={column.source}`, "
        f"молекул: {len(column.molecules)}"
        for column in columns
    ]
    lines += ["", _markdown_table(table), ""]

    if any(column.summary is None for column in columns):
        lines += [
            f"Доверительные интервалы не показаны: `{SUMMARY_CSV}` ещё "
            "не посчитан. Распределения приведены медианой с межквартильным размахом.",
            "",
        ]
    if dropped:
        lines += ["Не вошло в таблицу:", ""]
        lines += [f"- {reason}" for reason in dropped]
        lines += [""]
    return "\n".join(lines)


def build_report(run_dirs: list[Path], out_dir: Path) -> tuple[Path, Path]:
    """Читает прогоны и пишет `before_after.md` и `before_after.csv`."""
    columns: list[Column] = []
    for run_dir in run_dirs:
        columns.extend(load_run(run_dir))

    table, dropped = build_table(columns)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / BEFORE_AFTER_MD
    csv_path = out_dir / BEFORE_AFTER_CSV
    markdown_path.write_text(render_markdown(table, columns, dropped), encoding="utf-8")
    # `lineterminator` задан явно: умолчание pandas совпадает с `CSV_EOL` сегодня,
    # но это совпадение, а не наше решение. Смена умолчания
    # переписала бы файл целиком — 39 строк в diff вместо ни одной.
    to_long_frame(columns).to_csv(
        csv_path, index=False, encoding="utf-8", lineterminator=CSV_EOL
    )
    return markdown_path, csv_path
