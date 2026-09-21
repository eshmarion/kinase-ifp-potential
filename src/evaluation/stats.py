"""Доверительные интервалы, критерий Манна–Уитни и `metrics_summary.csv`.

Правило, ради которого модуль существует: формулировка «метрика выросла» без
доверительного интервала в текст курсовой не идёт (`docs/metrics.md`, раздел 5).
До появления этого файла `ci_low` и `ci_high` были пусты во всех строках
`results/before_after.csv`, то есть раздел «Результаты» не мог содержать
ни одного утверждения о росте.

**Считается на numpy, а не на scipy.** `scipy` стоит в окружении как зависимость
MDAnalysis, prolif и scikit-learn, но в `pyproject.toml` не объявлен, а `mypy` без
стабов его не пропускает. Опираться на пакет, который никто не объявлял, — это ровно
случай `opencadd`, чей `setup.py` не перечислял зависимостей. Поэтому ранговый критерий
реализован здесь: тридцать строк под тестом на известном ответе дешевле правки
чужой зоны за сутки до сдачи.

**Условие здесь — подпись колонки отчёта, а не поле формата таблицы метрик.** `metrics_summary.csv`
читает `report._interval`, сопоставляя строки по `column.condition`: «diffsbdd»,
«diffsbdd, топ 20%». Файл, заполненный по колонке `condition` таблицы метрик, не совпал бы
с подписями никогда, а выглядело бы это как «файла нет». Отсюда способ
сборки: подписи не строятся заново, а берутся у `report.load_run` — одной функцией
на обоих концах.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from experiments.layout import CSV_EOL, SUMMARY_CSV
from experiments.report import REPORT_METRICS, Column, MetricSpec, load_run, metric_values
from kinase_ifp.config import (
    BOOTSTRAP_PERCENTILES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PERMUTATION_RESAMPLES,
    PERMUTATION_SEED,
)

# Как сворачивается выборка внутри бутстрэпа. Доли — среднее нулей и единиц,
# распределения — медиана: ровно то, чем `report.compute_cell` считает саму ячейку.
# Интервал вокруг одной статистики при другом числе в таблице был бы подлогом.
Aggregate = Literal["mean", "median"]

# Метрики набора (`uniqueness`, `diversity`) определены только на наборе целиком
# и колонок в таблице метрик не имеют: пересэмплировать нечего. Их ячейки остаются без ДИ,
# и это неприменимость, а не пропуск.
SUMMARY_METRICS: Final[tuple[MetricSpec, ...]] = tuple(
    metric for metric in REPORT_METRICS if metric.aggregation != "set"
)

SUMMARY_COLUMNS: Final[tuple[str, ...]] = ("condition",) + tuple(
    имя
    for metric in SUMMARY_METRICS
    for имя in (f"{metric.column}_lo", f"{metric.column}_hi", f"{metric.column}_p")
)

# Подпись отобранного топа начинается с подписи её прогона и этого разделителя:
# «diffsbdd» → «diffsbdd, топ 20%» (`report._split_by_selection`).
TOP_SUFFIX: Final[str] = ", топ"


class StatsError(RuntimeError):
    """Статистику посчитать нечем: пустая выборка или несопоставимые условия."""


def bootstrap_ci(
    values: Sequence[float] | pd.Series | np.ndarray,
    *,
    aggregate: Aggregate,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Перцентильный бутстрэп-интервал статистики: границы 2.5% и 97.5%.

    Принимает значения метрики по молекулам одного условия, возвращает нижнюю
    и верхнюю границы. `aggregate` выбирает статистику: `mean` для долей,
    `median` для распределений.

    Зерно генератора фиксировано (`BOOTSTRAP_SEED`): числа уходят в текст курсовой,
    и повторный запуск обязан дать те же границы.
    """
    выборка = np.asarray(values, dtype=float)
    if выборка.size == 0:
        raise StatsError("бутстрэп по пустой выборке: считать нечего")
    if bool(np.isnan(выборка).any()):
        raise StatsError(
            "в выборке есть NaN: непосчитанные молекулы отбрасываются до бутстрэпа, "
            "иначе интервал считается по другому множеству, чем само число"
        )

    rng = np.random.default_rng(seed)
    индексы = rng.integers(0, выборка.size, size=(resamples, выборка.size))
    статистики = (
        выборка[индексы].mean(axis=1)
        if aggregate == "mean"
        else np.median(выборка[индексы], axis=1)
    )
    low, high = np.percentile(статистики, BOOTSTRAP_PERCENTILES)
    return float(low), float(high)


def _средние_ранги(значения: np.ndarray) -> np.ndarray:
    """Ранги от 1 до n; связанные значения получают свой средний ранг."""
    порядок = np.argsort(значения, kind="mergesort")
    ранги = np.empty(значения.size, dtype=float)
    ранги[порядок] = np.arange(1, значения.size + 1, dtype=float)

    отсортированные = значения[порядок]
    начало = 0
    while начало < значения.size:
        конец = начало
        while конец + 1 < значения.size and отсортированные[конец + 1] == отсортированные[начало]:
            конец += 1
        if конец > начало:
            ранги[порядок[начало : конец + 1]] = (начало + конец + 2) / 2
        начало = конец + 1
    return ранги


def spearman_rho(
    x: Sequence[float] | pd.Series | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Ранговая корреляция Спирмена: Пирсон по средним рангам.

    Живёт здесь, а не рядом с потребителем: ранги со связками уже реализованы в этом
    модуле для критерия Манна–Уитни, а связки в наших данных массовые — жёсткий скор
    принимает не более шести значений.

    Ряд, где все значения совпали, корреляции не имеет: дисперсия рангов равна нулю,
    и `StatsError` здесь честнее, чем `nan`, который ниже по течению превратится
    в пустую ячейку с непонятной причиной.
    """
    первый = np.asarray(x, dtype=float)
    второй = np.asarray(y, dtype=float)
    if первый.size != второй.size:
        raise StatsError(
            f"ряды разной длины: {первый.size} и {второй.size} — пары не построить"
        )
    if первый.size < 2:
        raise StatsError("корреляция по одной точке не определена")

    ранги_x = _средние_ранги(первый)
    ранги_y = _средние_ранги(второй)
    if ранги_x.std() == 0 or ранги_y.std() == 0:
        raise StatsError("все значения одного из рядов совпали: ранговой связи нет")
    return float(np.corrcoef(ранги_x, ранги_y)[0, 1])


def mann_whitney_u(
    x: Sequence[float] | pd.Series | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[float, float]:
    """U-статистика первой выборки и двустороннее p-value.

    Нормальная аппроксимация с поправкой на непрерывность и на связи — тот же
    способ, что `scipy.stats.mannwhitneyu(method="asymptotic")`; совпадение
    проверено на известном ответе в `tests/test_stats.py`.

    Выборки обязаны быть независимыми: топ сравнивается с отбракованными, а не
    со всем прогоном, куда он входит сам. Проверить это здесь нечем — за парность
    отвечает `summarize_run`.
    """
    первая = np.asarray(x, dtype=float)
    вторая = np.asarray(y, dtype=float)
    if первая.size == 0 or вторая.size == 0:
        raise StatsError("критерий Манна–Уитни по пустой выборке: сравнивать нечего")

    все = np.concatenate([первая, вторая])
    ранги = _средние_ранги(все)
    n1, n2, n = первая.size, вторая.size, все.size

    u = float(ранги[:n1].sum() - n1 * (n1 + 1) / 2)
    среднее = n1 * n2 / 2

    _, кратности = np.unique(все, return_counts=True)
    поправка_на_связи = float((кратности**3 - кратности).sum())
    дисперсия = n1 * n2 / 12 * ((n + 1) - поправка_на_связи / (n * (n - 1)))
    if дисперсия <= 0:
        # Все значения обеих выборок совпали: различать нечего, и это не ошибка.
        return u, 1.0

    отклонение = max(abs(u - среднее) - 0.5, 0.0)
    z = отклонение / math.sqrt(дисперсия)
    return u, math.erfc(z / math.sqrt(2))


def spearman(
    x: Sequence[float] | pd.Series | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Ранговая корреляция Спирмена; связанные значения получают средний ранг.

    Считается здесь, а не берётся из `scipy`, по причине, изложенной в шапке модуля.
    Копии этой же формулы живут в `src/audit/t1.py` и `src/audit/t10.py` — там они
    намеренные: разбор обязан воспроизводиться сам по себе, не завися от того, что
    производственный код с тех пор переписали. Третьей копии заводить незачем,
    и производственный путь ведёт сюда.

    Вырожденный случай — когда одна из выборок постоянна: ранги совпадают у всех,
    и корреляции нет не потому, что связь слабая, а потому, что её не из чего
    посчитать. Возвращается `nan`, а не ноль: ноль читался бы как измеренное
    отсутствие связи.
    """
    первый = np.asarray(x, dtype=float)
    второй = np.asarray(y, dtype=float)
    if первый.size != второй.size:
        raise StatsError(
            f"ряды разной длины: {первый.size} и {второй.size} — пары не определены"
        )
    if первый.size < 2:
        raise StatsError("ранговая корреляция по одной точке не определена")

    ранги_x = _средние_ранги(первый)
    ранги_y = _средние_ранги(второй)
    отклонения_x = ранги_x - ранги_x.mean()
    отклонения_y = ранги_y - ранги_y.mean()
    знаменатель = math.sqrt(float((отклонения_x**2).sum() * (отклонения_y**2).sum()))
    if знаменатель == 0.0:
        return float("nan")
    return float((отклонения_x * отклонения_y).sum() / знаменатель)


def spearman_permutation_p(
    x: Sequence[float] | pd.Series | np.ndarray,
    y: Sequence[float] | pd.Series | np.ndarray,
    *,
    resamples: int = PERMUTATION_RESAMPLES,
    seed: int = PERMUTATION_SEED,
) -> tuple[float, int]:
    """Двусторонний уровень значимости ρ перестановками; возвращает `p` и число перестановок.

    Один ряд перемешивается `resamples` раз, и считается доля перестановок, давших
    корреляцию не слабее наблюдённой по модулю. Ни одного допущения о распределении
    при этом не делается — в отличие от приближения через Стьюдента, которое требует
    двумерной нормальности, а у отклонений позы её нет.

    Оценка сдвинутая: к числителю и знаменателю добавляется единица. Без этого
    результат «ноль перестановок» дал бы `p = 0`, то есть утверждение о невозможном,
    тогда как измерено лишь «меньше одной перестановки из десяти тысяч». Число
    перестановок возвращается затем, чтобы эту границу можно было назвать в тексте.
    """
    наблюдённая = spearman(x, y)
    if math.isnan(наблюдённая):
        raise StatsError("ρ не определена: одна из выборок постоянна, перемешивать нечего")
    if resamples < 1:
        raise StatsError(f"число перестановок должно быть положительным, получено {resamples}")

    первый = np.asarray(x, dtype=float)
    второй = np.asarray(y, dtype=float)
    генератор = np.random.default_rng(seed)
    не_слабее = 0
    for _ in range(resamples):
        если_бы = spearman(первый, генератор.permutation(второй))
        if abs(если_бы) >= abs(наблюдённая):
            не_слабее += 1
    return (не_слабее + 1) / (resamples + 1), resamples


def _пары_условий(колонки: list[Column]) -> dict[int, int]:
    """Для колонки отобранного топа — индекс её прогона; остальные не сравниваются.

    Порядок задаёт `report._split_by_selection`: топ идёт сразу за своим прогоном
    и зовётся его подписью с суффиксом. Соответствие проверяется по подписи, а не
    по одному лишь порядку: молча сравнить чужие условия хуже, чем не сравнить.
    """
    пары: dict[int, int] = {}
    for индекс in range(1, len(колонки)):
        предыдущая = колонки[индекс - 1].condition
        if колонки[индекс].condition.startswith(f"{предыдущая}{TOP_SUFFIX}"):
            пары[индекс] = индекс - 1
    return пары


def _отбракованные(база: Column) -> pd.DataFrame:
    """Молекулы прогона, не попавшие в отобранный топ (`selected != 1`)."""
    if "selected" not in база.molecules.columns:
        raise StatsError(
            f"в условии {база.condition!r} нет колонки selected, "
            "а сравнить топ с отбракованными без неё нечем (формат таблицы ранжирования)"
        )
    отобран = база.molecules["selected"].fillna(0).astype(int) == 1
    return база.molecules[~отобран]


def _строка_условия(колонка: Column, отбракованные: pd.DataFrame | None) -> dict[str, object]:
    """Одна строка `metrics_summary.csv`: границы ДИ и p-value по каждой метрике.

    Пустая ячейка означает неприменимость: метрика не посчитана в этом условии
    либо сравнивать не с чем. Ноль здесь читался бы как измеренное значение.
    """
    строка: dict[str, object] = {"condition": колонка.condition}
    for метрика in SUMMARY_METRICS:
        значения = metric_values(колонка, метрика)
        low: object = ""
        high: object = ""
        p: object = ""
        if not значения.empty:
            low, high = bootstrap_ci(
                значения,
                aggregate="mean" if метрика.aggregation == "share" else "median",
            )
            if отбракованные is not None and метрика.column in отбракованные.columns:
                контроль = pd.to_numeric(
                    отбракованные[метрика.column], errors="coerce"
                ).dropna()
                if not контроль.empty:
                    p = mann_whitney_u(значения, контроль)[1]
        строка[f"{метрика.column}_lo"] = low
        строка[f"{метрика.column}_hi"] = high
        строка[f"{метрика.column}_p"] = p
    return строка


def summarize_run(run_dir: Path) -> Path:
    """Считает `metrics_summary.csv` для папки прогона и возвращает путь к нему.

    Строка на каждое условие прогона, подписи — те же, что в шапке таблицы
    «было/стало». У отобранного топа заполняется ещё и p-value: топ против
    отбракованных молекул того же прогона. Сравнение с прогоном целиком не
    делается намеренно — топ входит в него сам, выборки зависимы, и критерий
    Манна–Уитни на них неприменим.
    """
    колонки = load_run(run_dir)
    пары = _пары_условий(колонки)

    строки = [
        _строка_условия(
            колонка,
            _отбракованные(колонки[пары[индекс]]) if индекс in пары else None,
        )
        for индекс, колонка in enumerate(колонки)
    ]

    путь = run_dir / SUMMARY_CSV
    with путь.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(SUMMARY_COLUMNS), lineterminator=CSV_EOL)
        писатель.writeheader()
        писатель.writerows(строки)
    return путь
