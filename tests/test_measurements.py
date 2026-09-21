"""Реестр измерений по мишеням: ничего не теряется, свои строки заменяются."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from experiments.measurements import (
    COLUMNS,
    Measurement,
    MeasurementError,
    markdown_table,
    read_measurements,
    select,
    targets,
    upsert_measurements,
)


def измерение(**поля: str) -> Measurement:
    """Измерение с заполненными обязательными полями: тесту важны только ключ и значение."""
    основа = {
        "target": "6tgu",
        "measurement": "calibration_smoke",
        "variant": "explicit",
        "metric": "tanimoto_all",
        "value": "0.588",
    }
    основа.update(поля)
    return Measurement(**основа)  # type: ignore[arg-type]


def test_шапка_совпадает_с_полями_измерения() -> None:
    """Порядок колонок файла и полей записи — один и тот же: на это опираются читатели."""
    assert tuple(поле.name for поле in fields(Measurement)) == COLUMNS


def test_отсутствующий_файл_это_пустой_реестр(tmp_path: Path) -> None:
    assert read_measurements(tmp_path / "index.csv") == ()


def test_запись_и_чтение_возвращают_то_же(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    upsert_measurements([измерение()], путь)
    прочитанные = read_measurements(путь)

    assert len(прочитанные) == 1
    assert прочитанные[0].value == "0.588"
    assert прочитанные[0].key == ("6tgu", "calibration_smoke", "explicit", "tanimoto_all")


def test_повторная_запись_заменяет_свою_строку_а_не_дублирует(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    upsert_measurements([измерение()], путь)
    upsert_measurements([измерение(value="0.601")], путь)

    прочитанные = read_measurements(путь)
    assert [и.value for и in прочитанные] == ["0.601"]


def test_другая_мишень_не_вытесняет_прежнюю(tmp_path: Path) -> None:
    """Ровно та потеря, из-за которой реестр заведён: прогон по 6fnk стирал числа 6tgu."""
    путь = tmp_path / "index.csv"
    upsert_measurements([измерение()], путь)
    upsert_measurements([измерение(target="6fnk", value="0.214")], путь)

    значения = {и.target: и.value for и in read_measurements(путь)}
    assert значения == {"6tgu": "0.588", "6fnk": "0.214"}


def test_новый_вид_измерения_добавляется_к_прежним(tmp_path: Path) -> None:
    """Главное требование: вид теста, которого раньше не было, не трогает записанное."""
    путь = tmp_path / "index.csv"
    upsert_measurements(
        [измерение(), измерение(target="6fnk", value="0.214")],
        путь,
    )
    upsert_measurements(
        [
            измерение(
                measurement="pose_optimization",
                variant="all7",
                metric="score_after",
                value="0.439",
                target="6fnk",
            )
        ],
        путь,
    )

    все = read_measurements(путь)
    assert len(все) == 3
    assert {и.measurement for и in все} == {"calibration_smoke", "pose_optimization"}
    assert [и.value for и in select(все, measurement="calibration_smoke", target="6tgu")] == [
        "0.588"
    ]


def test_повтор_ключа_внутри_одной_записи_отвергается(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    with pytest.raises(MeasurementError, match="повторяются ключи"):
        upsert_measurements([измерение(), измерение(value="0.999")], путь)


def test_чужая_шапка_отвергается(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    путь.write_text("target,value\n6tgu,0.5\n", encoding="utf-8")
    with pytest.raises(MeasurementError, match="шапка"):
        read_measurements(путь)


def test_файл_пишется_с_переводом_строки_lf(tmp_path: Path) -> None:
    """Решение №28: CSV в этом репозитории пишутся с '\\n', иначе diff краснеет целиком."""
    путь = tmp_path / "index.csv"
    upsert_measurements([измерение()], путь)

    assert b"\r\n" not in путь.read_bytes()


def test_строки_сортируются_по_ключу(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    upsert_measurements(
        [измерение(target="6tgu"), измерение(target="3war"), измерение(target="6fnk")],
        путь,
    )

    assert [и.target for и in read_measurements(путь)] == ["3war", "6fnk", "6tgu"]


def test_список_мишеней_по_виду_измерения(tmp_path: Path) -> None:
    путь = tmp_path / "index.csv"
    upsert_measurements(
        [
            измерение(target="6tgu"),
            измерение(target="6fnk", measurement="pose_optimization"),
        ],
        путь,
    )
    все = read_measurements(путь)

    assert targets(все) == ("6fnk", "6tgu")
    assert targets(все, "calibration_smoke") == ("6tgu",)


def test_таблица_печатает_прочерк_на_неизмеренном(tmp_path: Path) -> None:
    """Неизмеренное и измеренный нуль — разные вещи, и в отчёте их путать нельзя."""
    путь = tmp_path / "index.csv"
    upsert_measurements(
        [
            измерение(),
            измерение(target="6fnk", value="0.214"),
            измерение(target="6fnk", metric="bits_ours", value="3"),
        ],
        путь,
    )

    таблица = markdown_table(
        read_measurements(путь),
        "calibration_smoke",
        [("tanimoto_all", "Танимото"), ("bits_ours", "Наших бит")],
    )

    assert таблица[0] == "| Мишень | Режим | Танимото | Наших бит |"
    assert таблица[2] == "| 6fnk | `explicit` | 0.214 | 3 |"
    # У 6tgu число бит не измерено — в ячейке прочерк, а не нуль.
    assert таблица[3] == "| 6tgu | `explicit` | 0.588 | — |"
