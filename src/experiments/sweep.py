"""Свип по размеру отбираемого топа: кривая компромисса.

Отвечает на вопрос «чем платим за ужесточение отбора»: как меняются целевые метрики
и не портятся ли контрольные, когда доля отбираемых молекул падает с 50 % до 5 %.

**Прогон при этом не трогается.** Переранжирование с другой долей топа разрешено
, но перезапись `ranking.csv` меняла бы колонку `selected` у прогона,
числа которого уже стоят в тексте курсовой. Свип поэтому читает готовые `ranking.csv`
и `metrics_per_molecule.csv` и режет их по рангу сам: скор и ранг от доли не зависят,
меняется только граница отбора.

**Честная оговорка, ради которой считаются `distinct_scores` и `ties_at_cut`.**
Скор принимает мало различных значений: у эталона 6tgu пять бит по типам скора,
поэтому у молекул прогона не больше шести уровней. Доли, попадающие внутрь одного
уровня, режут группу молекул с одинаковым скором — и порядок внутри неё определяется
не качеством, а появлением молекулы в прогоне (`kinase_ifp.scoring.rank_molecules`).
Кривая, посчитанная без этих двух колонок, выглядела бы измерением там, где часть
точек различается произволом разреза. Ответ на это — не выбор «правильных» долей,
а публикация того, где разрез попал
в середину группы.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from evaluation.stats import bootstrap_ci
from experiments.layout import CSV_EOL, SOURCE_LOCAL_POSES, SWEEP_CSV
from experiments.report import (
    REPORT_METRICS,
    Column,
    MetricSpec,
    compute_cell,
    load_run,
    metric_values,
)

# Доли топа. 100 % — это сам прогон без отбора: кривая обязана начинаться с точки
# «ничего не отбирали», иначе не с чем сравнивать остальные.
DEFAULT_SWEEP_FRACTIONS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)

SWEEP_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "source",
    "top_fraction",
    "n_selected",
    "distinct_scores",
    "score_at_cut",
    "ties_at_cut",
    "metric",
    "aggregation",
    "value",
    "iqr_low",
    "iqr_high",
    "ci_low",
    "ci_high",
)


class SweepError(RuntimeError):
    """Свип посчитать нечем: у прогона нет ранжирования или доли заданы неверно."""


@dataclass(frozen=True)
class Cut:
    """Разрез набора по доле: что попало в топ и насколько граница произвольна."""

    fraction: float
    molecules: pd.DataFrame
    distinct_scores: int
    score_at_cut: float | None
    ties_at_cut: int


def _all_molecules(run_dir: Path) -> tuple[Column, str, str]:
    """Все молекулы прогона одной колонкой, вместе с `run_id` и `source`.

    `load_run` делит прогон на условия — для свипа это лишнее: резать надо
    исходный набор, а не уже отобранный кем-то топ.
    """
    колонки = load_run(run_dir)
    if not колонки:
        raise SweepError(f"{run_dir}: прогон не дал ни одного условия")
    молекулы = pd.concat([колонка.molecules for колонка in колонки], ignore_index=True)
    молекулы = молекулы.drop_duplicates(subset="mol_id")
    if "rank" not in молекулы.columns or молекулы["rank"].isna().all():
        raise SweepError(
            f"{run_dir}: нет колонки 'rank' — прогон не переранжирован, "
            "свип по размеру топа считать не по чему (формат таблицы ранжирования)"
        )
    первая = колонки[0]
    return (
        Column(
            condition="весь набор",
            run_id=первая.run_id,
            source=первая.source,
            molecules=молекулы,
            summary=None,
        ),
        первая.run_id,
        первая.source,
    )


def _скоры(frame: pd.DataFrame) -> pd.Series:
    """Числовые значения скора; отсутствующая колонка — пустой ряд, а не отказ.

    Прогон без скора сюда не доходит (`_all_molecules` требует ранга), но колонка
    и ранг приходят из разных файлов, и отсутствие одной при наличии другой должно
    давать пустую статистику, а не исключение из глубины pandas.
    """
    if "ifp_score" not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame["ifp_score"], errors="coerce").dropna()


def cut_by_fraction(молекулы: pd.DataFrame, fraction: float) -> Cut:
    """Отбирает верхнюю долю по рангу и описывает, куда пришлась граница разреза."""
    if not 0.0 < fraction <= 1.0:
        raise SweepError(f"Доля топа {fraction} вне диапазона (0, 1]")

    ранги = pd.to_numeric(молекулы["rank"], errors="coerce")
    # Округление вверх — как в `rank_molecules`: топ в 5 % от 95 молекул это 5, не 4.
    размер = math.ceil(len(молекулы) * fraction)
    отобранные = молекулы[ранги <= размер]

    скоры = _скоры(отобранные)
    все_скоры = _скоры(молекулы)
    граница = float(скоры.min()) if not скоры.empty else None
    # Сколько молекул всего набора имеют пограничный скор: если их больше, чем влезло
    # в топ, разрез прошёл внутри группы одинаковых, и часть отбора — произвол порядка.
    # На полном наборе разреза нет вовсе, и число ничьих там не «ноль совпадений»,
    # а отсутствие вопроса: отбор не делался.
    режем = размер < len(молекулы)
    ничьи = int((все_скоры == граница).sum()) if режем and граница is not None else 0
    return Cut(
        fraction=fraction,
        molecules=отобранные,
        distinct_scores=int(скоры.nunique()),
        score_at_cut=граница,
        ties_at_cut=ничьи,
    )


def _строки_точки(колонка: Column, cut: Cut, run_id: str, source: str) -> list[dict[str, object]]:
    """Одна точка кривой: все применимые метрики на отобранном подмножестве."""
    точка = Column(
        condition=f"топ {cut.fraction:.0%}",
        run_id=run_id,
        source=source,
        molecules=cut.molecules,
        summary=None,
    )
    строки: list[dict[str, object]] = []
    for метрика in REPORT_METRICS:
        if метрика.diffsbdd_only and source == SOURCE_LOCAL_POSES:
            continue
        ячейка = compute_cell(точка, метрика)
        if ячейка.value is None:
            continue
        ci_low, ci_high = _интервал(точка, метрика)
        строки.append(
            {
                "run_id": run_id,
                "source": source,
                "top_fraction": round(cut.fraction, 4),
                "n_selected": len(cut.molecules),
                "distinct_scores": cut.distinct_scores,
                "score_at_cut": cut.score_at_cut,
                "ties_at_cut": cut.ties_at_cut,
                "metric": метрика.column,
                "aggregation": метрика.aggregation,
                "value": ячейка.value,
                "iqr_low": ячейка.iqr_low,
                "iqr_high": ячейка.iqr_high,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return строки


def _интервал(колонка: Column, метрика: MetricSpec) -> tuple[float | None, float | None]:
    """Бутстрэп-интервал точки свипа.

    Считается здесь, а не берётся из `metrics_summary.csv`: тот файл описывает два
    условия прогона, а точек у кривой шесть, и своего интервала у них там нет.
    Статистика та же самая — `evaluation.stats.bootstrap_ci` со своим зерном,
    поэтому повторный запуск даёт те же границы.
    """
    if метрика.aggregation == "set":
        return None, None
    значения = metric_values(колонка, метрика)
    if значения.empty:
        return None, None
    низ, верх = bootstrap_ci(
        значения, aggregate="mean" if метрика.aggregation == "share" else "median"
    )
    return низ, верх


def sweep_run(
    run_dir: Path, fractions: tuple[float, ...] = DEFAULT_SWEEP_FRACTIONS
) -> pd.DataFrame:
    """Считает кривую компромисса для одного прогона. Возвращает длинную таблицу."""
    колонка, run_id, source = _all_molecules(run_dir)
    строки: list[dict[str, object]] = []
    for доля in sorted(fractions, reverse=True):
        cut = cut_by_fraction(колонка.molecules, доля)
        if cut.molecules.empty:
            raise SweepError(f"{run_dir}: доля {доля} не отобрала ни одной молекулы")
        строки.extend(_строки_точки(колонка, cut, run_id, source))
    return pd.DataFrame(строки, columns=list(SWEEP_COLUMNS))


def build_sweep(
    run_dirs: list[Path],
    out_dir: Path,
    fractions: tuple[float, ...] = DEFAULT_SWEEP_FRACTIONS,
) -> tuple[Path, Path]:
    """Собирает свип по прогонам и пишет `sweep_top_fraction.csv` и `.md`."""
    if not run_dirs:
        raise SweepError("не задано ни одного прогона")
    таблица = pd.concat(
        [sweep_run(каталог, fractions) for каталог in run_dirs], ignore_index=True
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / SWEEP_CSV
    таблица.to_csv(csv_path, index=False, lineterminator=CSV_EOL)
    md_path = out_dir / SWEEP_CSV.replace(".csv", ".md")
    md_path.write_text(render_markdown(таблица), encoding="utf-8", newline="\n")
    return md_path, csv_path


# Метрики, попадающие в читаемую таблицу: по одной на каждый вопрос, который задают
# кривой. Остальное лежит в CSV — таблица на пятнадцать строк не читается глазами.
СТРОКИ_ОТЧЁТА: Final[tuple[tuple[str, str], ...]] = (
    ("ifp_score", "IFP-скор"),
    ("ifp_score_norm", "IFP-скор на тяжёлый атом"),
    ("ifp_tanimoto", "IFP-Танимото к эталону"),
    ("hinge_hbond", "Доля молекул с ключевой H-связью"),
    ("qed", "QED"),
    ("sa_score", "SA score"),
    ("n_heavy_atoms", "Число тяжёлых атомов"),
    ("diversity", "Diversity"),
)


def render_markdown(таблица: pd.DataFrame) -> str:
    """Читаемая таблица: строки — метрики, колонки — размеры топа."""
    куски: list[str] = [
        "# Кривая компромисса: размер отбираемого топа\n",
        "Собрано `scripts/run_sweep.py`. Те же числа машинно — "
        "в `sweep_top_fraction.csv`.\n",
    ]
    for (run_id, source), часть in таблица.groupby(["run_id", "source"], sort=False):
        доли = sorted(часть["top_fraction"].unique(), reverse=True)
        шапка = [f"топ {доля:.0%}" for доля in доли]
        куски.append(f"\n## Прогон `{run_id}` (`{source}`)\n")

        описание = часть.drop_duplicates("top_fraction").set_index("top_fraction")
        куски.append("| Что отобрано | " + " | ".join(шапка) + " |")
        куски.append("|---" * (len(доли) + 1) + "|")
        куски.append(
            "| Молекул | "
            + " | ".join(str(int(описание.loc[доля, "n_selected"])) for доля in доли)
            + " |"
        )
        куски.append(
            "| Различных значений скора | "
            + " | ".join(str(int(описание.loc[доля, "distinct_scores"])) for доля in доли)
            + " |"
        )
        куски.append(
            "| Молекул с пограничным скором | "
            + " | ".join(str(int(описание.loc[доля, "ties_at_cut"])) for доля in доли)
            + " |"
        )
        куски.append("")
        куски.append("| Метрика | " + " | ".join(шапка) + " |")
        куски.append("|---" * (len(доли) + 1) + "|")
        for имя, заголовок in СТРОКИ_ОТЧЁТА:
            строки_метрики = часть[часть["metric"] == имя].set_index("top_fraction")
            if строки_метрики.empty:
                continue
            ячейки = []
            for доля in доли:
                if доля not in строки_метрики.index:
                    ячейки.append("—")
                    continue
                строка = строки_метрики.loc[доля]
                текст = f"{строка['value']:.3g}"
                if pd.notna(строка["ci_low"]) and pd.notna(строка["ci_high"]):
                    текст += f" [{строка['ci_low']:.3g}–{строка['ci_high']:.3g}]"
                ячейки.append(текст)
            куски.append(f"| {заголовок} | " + " | ".join(ячейки) + " |")
    куски.append(
        "\nВ квадратных скобках — 95-процентный доверительный интервал бутстрэпа.\n"
        "Строка «молекул с пограничным скором» показывает, сколько молекул всего набора\n"
        "имеют то же значение скора, что и последняя отобранная: если их больше, чем\n"
        "поместилось в топ, граница прошла внутри группы неразличимых молекул.\n"
    )
    return "\n".join(куски)
