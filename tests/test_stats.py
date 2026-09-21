"""Тесты статистики: бутстрэп-ДИ, критерий Манна–Уитни, `metrics_summary.csv`.

Числа в ожиданиях не выдуманы. Значения p сверены с
`scipy.stats.mannwhitneyu(alternative="two-sided", method="asymptotic")` на пяти
наборах, включая связи и полностью совпадающие выборки: совпадение до восьмого знака.
Сама сверка в тестах не воспроизводится — `scipy` попадает в окружение как зависимость
MDAnalysis и prolif, но в `pyproject.toml` не объявлен, и тест, опирающийся на него,
проверял бы чужой пакет вместо нашего.

Фикстуры те же, что у отчёта (`tests/fixtures/report/`): игрушечный прогон на пять
молекул, из которых две отобраны переранжированием.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation.stats import (
    SUMMARY_COLUMNS,
    StatsError,
    bootstrap_ci,
    mann_whitney_u,
    summarize_run,
)
from experiments.layout import SUMMARY_CSV
from experiments.report import MetricSpec, compute_cell, load_run

FIXTURES = Path(__file__).parent / "fixtures" / "report"
BASELINE = FIXTURES / "baseline"


def _прогон(tmp_path: Path) -> Path:
    """Копия фикстуры: `summarize_run` пишет в папку прогона, а фикстуры неизменны."""
    каталог = tmp_path / "run"
    каталог.mkdir()
    for имя in ("run.json", "metrics_per_molecule.csv", "ranking.csv"):
        (каталог / имя).write_bytes((BASELINE / имя).read_bytes())
    return каталог


def _строки_сводки(путь: Path) -> list[dict[str, str]]:
    with путь.open(encoding="utf-8", newline="") as файл:
        return list(csv.DictReader(файл))


def test_интервал_вырожденной_выборки_совпадает_со_значением() -> None:
    assert bootstrap_ci([1.0] * 20, aggregate="mean") == (1.0, 1.0)


def test_интервал_накрывает_долю_и_уже_отрезка() -> None:
    low, high = bootstrap_ci([1, 0] * 50, aggregate="mean")

    assert low < 0.5 < high
    assert (low, high) != (0.0, 1.0)


def test_интервал_воспроизводим_при_одном_зерне() -> None:
    выборка = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7]

    первый = bootstrap_ci(выборка, aggregate="median")
    второй = bootstrap_ci(выборка, aggregate="median")

    assert первый == второй


def test_зерно_действительно_участвует_в_розыгрыше() -> None:
    # Проверяется на среднем, а не на медиане: медиана короткой выборки принимает
    # всего несколько значений, и два разных розыгрыша дают одни и те же границы
    # даже при исправном генераторе.
    выборка = [0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7]

    assert bootstrap_ci(выборка, aggregate="mean", seed=0) != bootstrap_ci(
        выборка, aggregate="mean", seed=1
    )


def test_пустая_выборка_отвергается() -> None:
    with pytest.raises(StatsError, match="пустой выборке"):
        bootstrap_ci([], aggregate="mean")


def test_nan_в_выборке_отвергается() -> None:
    with pytest.raises(StatsError, match="NaN"):
        bootstrap_ci([0.1, float("nan"), 0.3], aggregate="median")


def test_критерий_на_полном_разделении() -> None:
    u, p = mann_whitney_u([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])

    assert u == 0.0
    assert p == pytest.approx(0.01218578, abs=1e-8)


def test_критерий_учитывает_связи() -> None:
    u, p = mann_whitney_u([1, 1, 2, 2, 3], [2, 2, 3, 3, 4])

    assert u == 5.0
    assert p == pytest.approx(0.12512239, abs=1e-8)


def test_совпадающие_выборки_дают_единицу() -> None:
    assert mann_whitney_u([1, 1, 1], [1, 1, 1])[1] == 1.0


def test_критерий_по_пустой_выборке_отвергается() -> None:
    with pytest.raises(StatsError, match="сравнивать нечего"):
        mann_whitney_u([1.0, 2.0], [])


def test_метрики_набора_в_сводку_не_идут() -> None:
    # `uniqueness` и `diversity` не имеют значения у отдельной молекулы,
    # пересэмплировать их нечем.
    assert not [имя for имя in SUMMARY_COLUMNS if имя.startswith(("uniqueness", "diversity"))]


def test_условия_сводки_совпадают_с_подписями_отчёта(tmp_path: Path) -> None:
    каталог = _прогон(tmp_path)

    путь = summarize_run(каталог)
    # Разбор именно модулем `csv`: подпись отобранного топа содержит запятую
    # («baseline, топ 20%»), и деление строки по запятой её разорвёт.
    условия = [строка["condition"] for строка in _строки_сводки(путь)]

    assert условия == [колонка.condition for колонка in load_run(каталог)]


def test_отчёт_подхватывает_посчитанные_интервалы(tmp_path: Path) -> None:
    # Главная проверка: файл, посчитанный здесь, должен читаться отчётом.
    # Промах по подписи условия выглядел бы как «файла нет», поэтому
    # проверяется не наличие файла, а появление ДИ в ячейке таблицы.
    каталог = _прогон(tmp_path)
    summarize_run(каталог)

    колонка = load_run(каталог)[0]
    ячейка = compute_cell(колонка, MetricSpec("hinge_hbond", "шарнир", "share"))

    assert ячейка.ci_low is not None and ячейка.ci_high is not None
    assert ячейка.ci_low <= ячейка.value <= ячейка.ci_high
    assert "ДИ" in ячейка.render()


def test_p_value_считается_только_для_отобранного_топа(tmp_path: Path) -> None:
    каталог = _прогон(tmp_path)
    summarize_run(каталог)

    прогон, топ = _строки_сводки(каталог / SUMMARY_CSV)

    assert прогон["ifp_tanimoto_p"] == ""
    assert топ["ifp_tanimoto_p"] != ""


def test_сводка_пишется_с_переводом_строки_unix(tmp_path: Path) -> None:
    # Прогоны коммитятся в git: CRLF от модуля `csv` показал бы
    # файл в diff изменённым целиком.
    summarize_run(_прогон(tmp_path))

    assert b"\r\n" not in (tmp_path / "run" / SUMMARY_CSV).read_bytes()
