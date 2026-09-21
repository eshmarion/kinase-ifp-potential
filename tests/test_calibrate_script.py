"""Тесты CLI сверки с эталоном KLIFS.

Проверяется то, ради чего отчёт собирается слиянием, а не записью целиком: прогон
обязан обновлять свои разделы и не трогать чужие. Раньше здесь стоял запрет на прогон
`--target` поверх готовой выборки — он закрывал половину дыры (раздел 2) и не закрывал
вторую (раздел 3, разбор механизма `HYD`, дописанный руками). Теперь оба раздела
переживают прогон по построению, и проверяется именно это.

В KLIFS тест не ходит и выборку не считает: дымовая сверка идёт по фикстуре 6tgu.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_ifp.py"


def load_script() -> Any:
    """Загружает `scripts/calibrate_ifp.py` как модуль.

    Каталог `scripts/` намеренно не лежит на `pythonpath` (там тонкие CLI-входы,
    а не библиотека), поэтому модуль загружается по пути файла.
    """
    spec = importlib.util.spec_from_file_location("calibrate_ifp_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _аргументы(
    target_json: Path,
    report: Path,
    *,
    sample: bool,
    reselected: Path | None = None,
    csv: Path | None = None,
    measurements: Path | None = None,
) -> argparse.Namespace:
    # Реестр измерений по умолчанию кладётся рядом с отчётом теста, а не в `data/`:
    # прогон теста не вправе дописывать числа в рабочий реестр репозитория.
    return argparse.Namespace(
        target="6tgu",
        sample=sample,
        targets_dir=target_json.parent.parent,
        structures=Path("нет-такого.csv"),
        kinases=Path("нет-такого.csv"),
        fingerprints=Path("нет-такого.csv"),
        reselected=reselected,
        baseline_csv=report.parent / "e04b_per_structure.csv",
        report=report,
        csv=csv,
        measurements=measurements if measurements is not None else report.parent / "измерения.csv",
    )


РУЧНОЙ_РАЗДЕЛ = "## 3. Механизм расхождения по `HYD`"


def _отчёт_с_чужими_разделами(модуль: Any, path: Path) -> str:
    """Отчёт, где кроме дымовой сверки есть посчитанная выборка и ручной разбор."""
    текст = (
        "# Сверка отпечатка с эталоном KLIFS\n\n"
        "Раздел 2 посчитан 2026-09-03.\n\n"
        "## 1. Дымовая сверка на мишени 6tgu\n\nстарая дымовая сверка\n\n"
        f"{модуль.SAMPLE_SECTION}\n\nмедиана 0.750\n\n"
        f"{РУЧНОЙ_РАЗДЕЛ}\n\nправило «любой углерод», Жаккар 0.972\n"
    )
    path.write_text(текст, encoding="utf-8")
    return текст


def test_дымовой_прогон_не_стирает_чужие_разделы(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Раздел 2 стоит семи минут и доступа к KLIFS, раздел 3 написан руками."""
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    _отчёт_с_чужими_разделами(модуль, отчёт)
    monkeypatch.setattr(модуль, "parse_args", lambda: _аргументы(target_json, отчёт, sample=False))

    модуль.main()

    текст = отчёт.read_text(encoding="utf-8")
    assert "медиана 0.750" in текст
    assert "правило «любой углерод», Жаккар 0.972" in текст
    assert "старая дымовая сверка" not in текст
    assert "Дымовая сверка по мишеням" in текст
    assert "| 6tgu |" in текст


def test_преамбула_чужого_отчёта_не_переписывается(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Дата в шапке относится к разделу 2; прогон раздела 1 не вправе её сдвинуть."""
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    _отчёт_с_чужими_разделами(модуль, отчёт)
    monkeypatch.setattr(модуль, "parse_args", lambda: _аргументы(target_json, отчёт, sample=False))

    модуль.main()

    assert "Раздел 2 посчитан 2026-09-03." in отчёт.read_text(encoding="utf-8")


def test_порядок_разделов_сохраняется(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    _отчёт_с_чужими_разделами(модуль, отчёт)
    monkeypatch.setattr(модуль, "parse_args", lambda: _аргументы(target_json, отчёт, sample=False))

    модуль.main()

    текст = отчёт.read_text(encoding="utf-8")
    assert текст.index("## 1.") < текст.index(модуль.SAMPLE_SECTION) < текст.index(РУЧНОЙ_РАЗДЕЛ)


def test_отчёта_ещё_нет_пишется_своя_шапка(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    monkeypatch.setattr(модуль, "parse_args", lambda: _аргументы(target_json, отчёт, sample=False))

    модуль.main()

    текст = отчёт.read_text(encoding="utf-8")
    assert текст.startswith("# Сверка отпечатка с эталоном KLIFS")
    assert "Дымовая сверка по мишеням" in текст
    assert "| 6tgu |" in текст
    assert модуль.SAMPLE_SECTION not in текст


def test_повторный_прогон_не_плодит_разделы(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    monkeypatch.setattr(модуль, "parse_args", lambda: _аргументы(target_json, отчёт, sample=False))

    модуль.main()
    один = отчёт.read_text(encoding="utf-8")
    модуль.main()

    assert отчёт.read_text(encoding="utf-8") == один


def test_хелпер_аргументов_знает_все_ключи() -> None:
    """Забытый ключ ронял бы прогон на AttributeError посреди расчёта, а не на разборе."""
    модуль = load_script()
    настоящие = vars(модуль.build_parser().parse_args([]))
    наши = vars(_аргументы(Path("т/6tgu/target.json"), Path("о.md"), sample=False))

    assert set(наши) == set(настоящие)


class TestКудаПишутсяЧисла:
    """Прежний файл по структурам — это колонка «было»; затереть его нельзя."""

    def test_без_перевыбора_путь_прежний(self) -> None:
        модуль = load_script()
        args = _аргументы(Path("т/6tgu/target.json"), Path("о.md"), sample=True)

        assert модуль._куда_писать_csv(args) == модуль.SAMPLE_CSV

    def test_с_перевыбором_путь_другой(self) -> None:
        модуль = load_script()
        args = _аргументы(
            Path("т/6tgu/target.json"), Path("о.md"), sample=True, reselected=Path("состав.csv")
        )

        путь = модуль._куда_писать_csv(args)
        assert путь == модуль.RESELECTED_CSV
        assert путь != модуль.SAMPLE_CSV

    def test_явный_ключ_главнее_умолчания(self, tmp_path: Path) -> None:
        модуль = load_script()
        args = _аргументы(
            Path("т/6tgu/target.json"),
            Path("о.md"),
            sample=True,
            reselected=Path("состав.csv"),
            csv=tmp_path / "своё.csv",
        )

        assert модуль._куда_писать_csv(args) == tmp_path / "своё.csv"


def test_перевыбор_без_выборки_отвергается(
    tmp_path: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Иначе ключ молча не делает ничего, а прогон выглядит успешным."""
    модуль = load_script()
    отчёт = tmp_path / "calibration.md"
    monkeypatch.setattr(
        модуль,
        "parse_args",
        lambda: _аргументы(target_json, отчёт, sample=False, reselected=tmp_path / "состав.csv"),
    )

    with pytest.raises(SystemExit, match="--sample"):
        модуль.main()
