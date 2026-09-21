"""Тесты свипа по размеру отбираемого топа.

Фикстуры те же, что у отчёта «было/стало»: игрушечные прогоны на пять и две молекулы
с паспортом, таблицей метрик и ранжированием. Своих заводить не нужно — перебор
читает ровно те же файлы.

Главное, что здесь проверяется, — не арифметика долей, а два свойства, без которых
кривая врёт: прогон при расчёте не меняется, и место разреза внутри группы молекул
с одинаковым скором видно в таблице.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.sweep import (
    DEFAULT_SWEEP_FRACTIONS,
    SweepError,
    build_sweep,
    cut_by_fraction,
    render_markdown,
    sweep_run,
)

FIXTURES = Path(__file__).parent / "fixtures" / "report"
BASELINE = FIXTURES / "baseline"


def набор(скоры: list[float]) -> pd.DataFrame:
    """Молекулы с заданными скорами и рангами по убыванию скора."""
    порядок = sorted(range(len(скоры)), key=lambda i: (-скоры[i], i))
    ранги = [0] * len(скоры)
    for место, индекс in enumerate(порядок, start=1):
        ранги[индекс] = место
    return pd.DataFrame(
        {
            "mol_id": [f"m{i}" for i in range(len(скоры))],
            "ifp_score": скоры,
            "rank": ранги,
        }
    )


def test_доля_режет_по_рангу_с_округлением_вверх() -> None:
    """Топ в 20 % от 95 молекул — это 19; на маленьком наборе топ не должен быть пустым."""
    cut = cut_by_fraction(набор([1.0] * 7), 0.2)

    assert len(cut.molecules) == 2


def test_полный_набор_отбором_не_считается() -> None:
    """У доли 100 % разреза нет, и число ничьих на границе — не ноль совпадений, а ноль вопросов."""
    cut = cut_by_fraction(набор([0.8, 0.6, 0.6, 0.4]), 1.0)

    assert len(cut.molecules) == 4
    assert cut.ties_at_cut == 0


def test_разрез_внутри_группы_одинаковых_виден() -> None:
    """Ради этого числа свип и написан: часть точек различается произволом разреза."""
    # Четыре молекулы со скором 0.6 и одна с 0.8; топ 40 % — это две молекулы,
    # то есть 0.8 и одна из четырёх равных.
    cut = cut_by_fraction(набор([0.8, 0.6, 0.6, 0.6, 0.6]), 0.4)

    assert len(cut.molecules) == 2
    assert cut.score_at_cut == pytest.approx(0.6)
    assert cut.ties_at_cut == 4


def test_доля_вне_диапазона_отвергается() -> None:
    with pytest.raises(SweepError, match="вне диапазона"):
        cut_by_fraction(набор([1.0, 0.5]), 1.5)


def test_прогон_без_ранжирования_не_считается(tmp_path: Path) -> None:
    """Свип по прогону без `ranking.csv` считать не по чему, и молчать об этом нельзя."""
    прогон = tmp_path / "run"
    прогон.mkdir()
    for имя in ("run.json", "metrics_per_molecule.csv"):
        (прогон / имя).write_bytes((BASELINE / имя).read_bytes())

    with pytest.raises(SweepError, match="не переранжирован"):
        sweep_run(прогон)


def test_свип_не_трогает_папку_прогона(tmp_path: Path) -> None:
    """Прогон, числа которого уже в тексте, обязан остаться байт в байт прежним."""
    прогон = tmp_path / "run"
    прогон.mkdir()
    for файл in BASELINE.iterdir():
        (прогон / файл.name).write_bytes(файл.read_bytes())
    было = {файл.name: файл.read_bytes() for файл in прогон.iterdir()}

    sweep_run(прогон, fractions=(0.5, 1.0))

    assert {файл.name: файл.read_bytes() for файл in прогон.iterdir()} == было


def test_кривая_даёт_точку_на_каждую_долю() -> None:
    таблица = sweep_run(BASELINE, fractions=(0.5, 1.0))

    assert sorted(таблица["top_fraction"].unique()) == [0.5, 1.0]
    assert set(таблица["run_id"]) == {"2026-08-23-6tgu-s0-n5"}


def test_точки_идут_от_полного_набора_к_самому_жёсткому_отбору() -> None:
    """Порядок в таблице — читаемость: слева «ничего не отбирали», справа «отобрали мало»."""
    таблица = sweep_run(BASELINE, fractions=DEFAULT_SWEEP_FRACTIONS)

    доли = list(dict.fromkeys(таблица["top_fraction"]))
    assert доли == sorted(доли, reverse=True)


def test_вырожденные_метрики_на_наборе_поз_не_считаются() -> None:
    """QED и разнообразие на ста положениях одной молекулы — не результат."""
    таблица = sweep_run(BASELINE, fractions=(1.0,))

    assert "qed" not in set(таблица["metric"])
    assert "ifp_score" in set(таблица["metric"])


def test_интервал_считается_для_каждой_точки() -> None:
    """Правило зоны: число без доверительного интервала в текст не идёт."""
    таблица = sweep_run(BASELINE, fractions=(1.0,))
    распределения = таблица[таблица["aggregation"] == "distribution"]

    assert not распределения.empty
    assert распределения["ci_low"].notna().all()
    assert распределения["ci_high"].notna().all()


def test_отчёт_и_таблица_пишутся_рядом(tmp_path: Path) -> None:
    md_path, csv_path = build_sweep([BASELINE], tmp_path, fractions=(0.5, 1.0))

    assert md_path.is_file() and csv_path.is_file()
    assert "Кривая компромисса" in md_path.read_text(encoding="utf-8")
    assert "top_fraction" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_csv_пишется_с_переводом_строки_проекта(tmp_path: Path) -> None:
    """Решение №28: CSV в репозитории идут с `\\n`, иначе diff шумит целыми файлами."""
    _, csv_path = build_sweep([BASELINE], tmp_path, fractions=(1.0,))

    assert b"\r\n" not in csv_path.read_bytes()


def test_в_отчёте_названы_run_id_и_source() -> None:
    """У числа должно быть видно, из какого прогона оно взято."""
    отчёт = render_markdown(sweep_run(BASELINE, fractions=(1.0,)))

    assert "2026-08-23-6tgu-s0-n5" in отчёт
    assert "local-poses" in отчёт
