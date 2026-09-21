"""Тесты сборки таблицы «было/стало».

Фикстуры — два игрушечных прогона на пять и две молекулы с `source=local-poses`,
собранные с паспортом, таблицей метрик и ранжированием. Каталог назван `report`, а не `runs`:
шаблон `runs/` в `.gitignore` срабатывает на любом уровне вложенности, и фикстуры
просто не попали бы в репозиторий.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.layout import SOURCE_DIFFSBDD
from experiments.report import (
    REPORT_METRICS,
    Column,
    MetricSpec,
    _interval,
    build_report,
    build_table,
    compute_cell,
    load_run,
)

FIXTURES = Path(__file__).parent / "fixtures" / "report"
BASELINE = FIXTURES / "baseline"
RERANK = FIXTURES / "rerank_top20"


def test_прогон_читается_и_даёт_колонку_на_условие() -> None:
    columns = load_run(BASELINE)

    # Условие одно на весь прогон, поэтому колонки две: до отбора и после.
    # Долю топа паспорт фикстуры не несёт, и подпись остаётся без числа.
    assert [column.condition for column in columns] == ["baseline", "baseline, топ"]
    column = columns[0]
    assert column.run_id == "2026-08-23-6tgu-s0-n5"
    assert column.source == "local-poses"
    assert len(column.molecules) == 5
    assert len(columns[1].molecules) == 2


def test_ranking_подмешивается_к_метрикам() -> None:
    column = load_run(BASELINE)[0]

    # ifp_score и ifp_score_norm приходят из ranking.csv, а не из таблицы метрик.
    assert "ifp_score_norm" in column.molecules.columns
    assert column.molecules["ifp_score_norm"].notna().all()


def test_прогон_без_паспорта_отвергается(tmp_path: Path) -> None:
    (tmp_path / "metrics_per_molecule.csv").write_text("mol_id,condition\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="run.json"):
        load_run(tmp_path)


def test_пустой_source_в_паспорте_отвергается(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({"run_id": "x", "source": ""}), encoding="utf-8")

    with pytest.raises(ValueError, match="source"):
        load_run(tmp_path)


def test_таблица_собирает_оба_условия() -> None:
    # Из BASELINE берётся колонка до отбора: сравнение идёт с отдельным прогоном
    # `rerank-top20`, а не с топом внутри самого BASELINE.
    columns = load_run(BASELINE)[:1] + load_run(RERANK)

    table, _ = build_table(columns)

    assert list(table.columns) == [column.header for column in columns]
    assert "IFP-Танимото к эталону" in table.index
    # Отбор в топ-2 поднял медиану Танимото: 0.33 против 0.465.
    baseline, rerank = table.loc["IFP-Танимото к эталону"]
    assert float(baseline.split()[0]) < float(rerank.split()[0])


def test_в_шапке_колонки_есть_run_id_и_source() -> None:
    columns = load_run(BASELINE)

    header = columns[0].header

    assert "2026-08-23-6tgu-s0-n5" in header
    assert "local-poses" in header


def test_вырожденные_метрики_не_выводятся_на_наборе_поз() -> None:
    columns = load_run(BASELINE) + load_run(RERANK)

    table, dropped = build_table(columns)

    for title in ("QED", "SA score", "Число тяжёлых атомов"):
        assert title not in table.index
        assert any(title in reason for reason in dropped)


def test_непосчитанная_метрика_попадает_в_список_невошедших() -> None:
    columns = load_run(BASELINE)

    _, dropped = build_table(columns)

    # vina_score в формате таблицы метрик присутствует и пуст: докинг не считался.
    assert any("Vina score" in reason for reason in dropped)


def test_метрики_для_diffsbdd_выводятся_при_своём_источнике() -> None:
    column = load_run(BASELINE)[0]
    generated = Column(
        condition="diffsbdd",
        run_id="2026-08-24-6tgu-s0-n5",
        source=SOURCE_DIFFSBDD,
        molecules=column.molecules.assign(qed=[0.4, 0.5, 0.6, 0.7, 0.8]),
        summary=None,
    )

    table, _ = build_table([generated])

    assert "QED" in table.index


def test_доля_и_медиана_считаются_по_разному() -> None:
    column = load_run(BASELINE)[0]
    share = compute_cell(column, MetricSpec("hinge_hbond", "hinge", "share"))
    median = compute_cell(column, MetricSpec("ifp_tanimoto", "танимото", "distribution"))

    assert share.value == pytest.approx(0.4)
    assert median.value == pytest.approx(0.33)
    assert median.iqr_low is not None and median.iqr_high is not None


def test_метрики_набора_считаются_по_smiles_а_не_по_колонке() -> None:
    """Uniqueness и diversity колонок в таблице метрик не имеют — они считаются по условию.

    Значения проверяются на наборе, где ответ известен заранее: три разные молекулы
    из четырёх записей дают uniqueness 0.75, а две копии одной — diversity 0.
    """
    column = load_run(BASELINE)[0]
    набор = Column(
        condition="diffsbdd",
        run_id="2026-08-24-6tgu-s0-n4",
        source=SOURCE_DIFFSBDD,
        molecules=pd.DataFrame(
            {"smiles": ["CCOc1ccccc1", "CCOc1ccccc1", "c1ccccc1", "CCO"]}
        ),
        summary=None,
    )

    уникальность = compute_cell(набор, MetricSpec("uniqueness", "Uniqueness", "set"))
    разнообразие = compute_cell(набор, MetricSpec("diversity", "Diversity", "set"))

    assert уникальность.value == pytest.approx(0.75)
    assert уникальность.n == 4
    assert разнообразие.value is not None and разнообразие.value > 0.5
    assert column.source != SOURCE_DIFFSBDD


def test_метрика_набора_на_одной_молекуле_даёт_пустую_ячейку() -> None:
    """Diversity от одной молекулы не определена; ноль читался бы как «нет разнообразия»."""
    одна = Column(
        condition="diffsbdd",
        run_id="2026-08-24-6tgu-s0-n1",
        source=SOURCE_DIFFSBDD,
        molecules=pd.DataFrame({"smiles": ["CCOc1ccccc1"]}),
        summary=None,
    )

    assert compute_cell(одна, MetricSpec("diversity", "Diversity", "set")).value is None


def test_ди_берётся_из_metrics_summary_если_он_есть(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ("run.json", "metrics_per_molecule.csv", "ranking.csv"):
        (run_dir / name).write_bytes((BASELINE / name).read_bytes())
    (run_dir / "metrics_summary.csv").write_text(
        "condition,hinge_hbond_lo,hinge_hbond_hi\nbaseline,0.12,0.74\n", encoding="utf-8"
    )

    column = load_run(run_dir)[0]
    cell = compute_cell(column, MetricSpec("hinge_hbond", "hinge", "share"))

    assert cell.ci_low == pytest.approx(0.12)
    assert "ДИ" in cell.render()


def test_отчёт_пишет_оба_файла(tmp_path: Path) -> None:
    markdown_path, csv_path = build_report([BASELINE, RERANK], tmp_path)

    text = markdown_path.read_text(encoding="utf-8")
    assert "2026-08-23-6tgu-s0-n5" in text and "local-poses" in text
    assert "Не вошло в таблицу" in text
    assert csv_path.read_text(encoding="utf-8").startswith("metric,aggregation,condition")


def test_отчёт_без_прогонов_падает(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ни одного прогона"):
        build_report([], tmp_path)


def test_все_метрики_отчёта_названы_по_разному() -> None:
    titles = [metric.title for metric in REPORT_METRICS]

    assert len(titles) == len(set(titles))


def _прогон_без_условия(
    каталог: Path, *, selected: tuple[int, ...], top_fraction: float | None = 0.2
) -> Path:
    """Прогон, сделанный одной командой: `condition` в паспорте нет, есть `ranking.csv`.

    Именно так выглядят живые прогоны — условие появляется только после отбора.
    """
    паспорт: dict[str, object] = {"run_id": "r1", "source": "diffsbdd"}
    if top_fraction is not None:
        паспорт["ranking"] = {"top_fraction": top_fraction, "n_selected": sum(selected)}
    (каталог / "run.json").write_text(json.dumps(паспорт), encoding="utf-8")

    строки = ["mol_id,condition,ifp_tanimoto"]
    строки += [f"m{i},,{0.9 if flag else 0.1}" for i, flag in enumerate(selected)]
    (каталог / "metrics_per_molecule.csv").write_text("\n".join(строки) + "\n", encoding="utf-8")

    ранги = ["mol_id,ifp_score,selected"]
    ранги += [f"m{i},{0.8 if flag else 0.2},{flag}" for i, flag in enumerate(selected)]
    (каталог / "ranking.csv").write_text("\n".join(ранги) + "\n", encoding="utf-8")
    return каталог


def test_прогон_без_условия_даёт_было_и_стало(tmp_path: Path) -> None:
    колонки = load_run(_прогон_без_условия(tmp_path, selected=(1, 1, 0, 0, 0)))

    assert [колонка.condition for колонка in колонки] == ["diffsbdd", "diffsbdd, топ 20%"]
    assert len(колонки[0].molecules) == 5
    assert len(колонки[1].molecules) == 2


def test_стало_считается_по_подмножеству_а_не_по_всем(tmp_path: Path) -> None:
    колонки = load_run(_прогон_без_условия(tmp_path, selected=(1, 1, 0, 0, 0)))

    было, стало = (колонка.molecules["ifp_tanimoto"] for колонка in колонки)
    assert стало.mean() > было.mean()
    assert set(стало) == {0.9}


def test_без_ranking_прогон_даёт_одну_колонку(tmp_path: Path) -> None:
    каталог = _прогон_без_условия(tmp_path, selected=(1, 0))
    (каталог / "ranking.csv").unlink()

    колонки = load_run(каталог)

    assert [колонка.condition for колонка in колонки] == ["diffsbdd"]


def test_отобранные_все_не_образуют_второго_условия(tmp_path: Path) -> None:
    """Отбор, забравший весь прогон, ничего не отбирает — второй колонки быть не должно."""
    колонки = load_run(_прогон_без_условия(tmp_path, selected=(1, 1, 1)))

    assert len(колонки) == 1


def test_доля_топа_без_паспорта_не_выдумывается(tmp_path: Path) -> None:
    колонки = load_run(_прогон_без_условия(tmp_path, selected=(1, 0, 0), top_fraction=None))

    assert колонки[1].condition == "diffsbdd, топ"


def test_одно_условие_делится_по_отбору(tmp_path: Path) -> None:
    """Помеченный прогон сравнивается сам с собой до отбора и после.

    Пока условий не проставляли, эти две колонки давала запасная ветка по `source`.
    Метка не отменяет сравнения: ради него сделан уровень A.
    """
    каталог = _прогон_без_условия(tmp_path, selected=(1, 1, 0))
    (каталог / "metrics_per_molecule.csv").write_text(
        "mol_id,condition,ifp_tanimoto\nm0,baseline,0.9\nm1,baseline,0.9\nm2,baseline,0.1\n",
        encoding="utf-8",
    )

    колонки = load_run(каталог)

    assert [колонка.condition for колонка in колонки] == ["baseline", "baseline, топ 20%"]
    assert len(колонки[0].molecules) == 3
    assert len(колонки[1].molecules) == 2


def test_несколько_условий_по_отбору_не_делятся(tmp_path: Path) -> None:
    """Условия внутри прогона сами задают колонки — делить их ещё и отбором нечем."""
    каталог = _прогон_без_условия(tmp_path, selected=(1, 1, 0))
    (каталог / "metrics_per_molecule.csv").write_text(
        "mol_id,condition,ifp_tanimoto\n"
        "m0,baseline,0.9\nm1,rerank-top20,0.9\nm2,baseline,0.1\n",
        encoding="utf-8",
    )

    колонки = load_run(каталог)

    assert [колонка.condition for колонка in колонки] == ["baseline", "rerank-top20"]


# --- доверительные интервалы: чей интервал попадает в ячейку ---
#
# `metrics_summary.csv` ещё не создан ни в одном прогоне, поэтому дефект
# ниже не был виден на живых данных и ждал бы дня сдачи.

ДОЛЯ = MetricSpec(column="valid", title="Доля валидных", aggregation="share")


def сводка(**колонки: list) -> pd.DataFrame:
    return pd.DataFrame(колонки)


def колонка_отчёта(summary: pd.DataFrame | None) -> Column:
    return Column(
        condition="diffsbdd",
        run_id="2026-08-24-6tgu-s0-n100",
        source=SOURCE_DIFFSBDD,
        molecules=pd.DataFrame(),
        summary=summary,
    )


def test_сводка_без_условия_не_даёт_интервал_молча() -> None:
    """Раньше фильтр просто пропускался, и все колонки получали первую строку файла."""
    без_условия = сводка(valid_lo=[0.1, 0.9], valid_hi=[0.2, 0.95])

    with pytest.raises(ValueError, match="нет колонки condition"):
        _interval(колонка_отчёта(без_условия), ДОЛЯ)


def test_две_строки_на_условие_отвергаются() -> None:
    """Взять первую — значит выбрать за автора файла и не сказать об этом."""
    дважды = сводка(
        condition=["diffsbdd", "diffsbdd"], valid_lo=[0.1, 0.9], valid_hi=[0.2, 0.95]
    )

    with pytest.raises(ValueError, match="не одна строка"):
        _interval(колонка_отчёта(дважды), ДОЛЯ)


def test_интервал_берётся_по_своему_условию() -> None:
    """Подпись условия строит _split_by_selection, и совпадать должна именно она."""
    обе = сводка(
        condition=["diffsbdd", "diffsbdd, топ 20%"],
        valid_lo=[0.1, 0.9],
        valid_hi=[0.2, 0.95],
    )

    assert _interval(колонка_отчёта(обе), ДОЛЯ) == (0.1, 0.2)


def test_чужое_условие_оставляет_ячейку_без_интервала() -> None:
    """Не ошибка: у прогона может не быть строки в сводке, и это видно по пустому ДИ."""
    чужое = сводка(condition=["local-poses"], valid_lo=[0.1], valid_hi=[0.2])

    assert _interval(колонка_отчёта(чужое), ДОЛЯ) is None


def test_без_колонок_границ_интервала_нет() -> None:
    """Сводка может считать не все метрики — это не повод падать."""
    другая = сводка(condition=["diffsbdd"], qed_lo=[0.1], qed_hi=[0.2])

    assert _interval(колонка_отчёта(другая), ДОЛЯ) is None
