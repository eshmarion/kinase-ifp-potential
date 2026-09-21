"""Тесты CLI прогона одной командой (`scripts/run_experiment.py`).

Измерение показало, что у входа **шесть
отказов и ни одного теста**: из десяти неисполняемых веток сегмента семь лежат
в двух CLI-входах, причём именно `run_experiment.py` назван в критерии «готово,
когда». Критерий проверяли руками на живых прогонах, и с тех
пор его не держало ничто.

Второе, что здесь закреплено, — **печать медианы**. В этой же команде нашлась
дефект, уже починенный в соседней (`compute_metrics.py`): медиана
бралась элементом по индексу `len // 2`, то есть при чётном числе молекул выходила
верхняя медиана, а не медиана. Тест сравнивает напечатанное с `statistics.median`
по тому же файлу — на чётном числе молекул, иначе проверка ничего не значит.

Прогон берётся настоящий: `build_local_pose_run` по фикстуре 6tgu, как в тестах
сверки. Каталог прогонов всегда временный — `--runs-dir` указывает в `tmp_path`,
и рабочий `runs/` не трогается ни одним тестом.
"""

from __future__ import annotations

import importlib.util
import shutil
import statistics
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from experiments.local_poses import build_local_pose_run
from experiments.metrics import MetricsError, read_metrics
from kinase_ifp.scoring import ScoringError

КОРЕНЬ = Path(__file__).resolve().parents[1]
SCRIPT_PATH = КОРЕНЬ / "scripts" / "run_experiment.py"

#: Чётное число поз: на нечётном верхняя медиана совпадает с медианой, и проверка
#: печати была бы тавтологией — ровно так дефект и дожил до разбора.
ПОЗ_В_ПРОГОНЕ = 4


def load_script() -> Any:
    """Загружает вход как модуль: `scripts/` намеренно не лежит на `pythonpath`."""
    spec = importlib.util.spec_from_file_location("run_experiment_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def сырой_прогон(tmp_path_factory: pytest.TempPathFactory, target_json: Path) -> Path:
    """Набор поз без метрик и ранжирования — то, с чего команда начинает."""
    runs = tmp_path_factory.mktemp("runs-исходник")
    return build_local_pose_run(
        target_json, n=ПОЗ_В_ПРОГОНЕ, seed=3, runs_dir=runs, day="2026-08-24"
    )


@pytest.fixture()
def прогон(сырой_прогон: Path, tmp_path: Path) -> Path:
    """Своя копия прогона на каждый тест: команда пишет в паспорт и в реестр."""
    каталог = tmp_path / "runs"
    каталог.mkdir()
    папка = каталог / сырой_прогон.name
    shutil.copytree(сырой_прогон, папка)
    return папка


@pytest.fixture()
def запуск(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Зовёт `main()` входа с подставленной командной строкой."""
    модуль = load_script()

    def вызвать(*аргументы: str) -> None:
        monkeypatch.setattr(sys, "argv", ["run_experiment.py", *аргументы])
        модуль.main()

    вызвать.модуль = модуль  # type: ignore[attr-defined]
    yield вызвать


def test_цепочка_доводит_прогон_до_scored(
    запуск: Any, прогон: Path, target_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Критерий «готово, когда», проверенный кодом, а не руками."""
    запуск("--run", str(прогон), "--target", str(target_json), "--runs-dir", str(прогон.parent))

    строки = [
        строка.split(",")
        for строка in (прогон.parent / "index.csv").read_text(encoding="utf-8").splitlines()
    ]
    заголовок, запись = строки[0], строки[1]
    assert len(строки) == 2
    assert запись[заголовок.index("run_id")] == прогон.name
    assert запись[заголовок.index("status")] == "scored"
    assert (прогон / "ranking.csv").is_file()


def test_печатается_медиана_а_не_верхний_средний(
    запуск: Any, прогон: Path, target_json: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """При чётном числе молекул это разные числа."""
    запуск("--run", str(прогон), "--target", str(target_json), "--runs-dir", str(прогон.parent))

    напечатано = capsys.readouterr().out
    значения = sorted(float(строка["ifp_tanimoto"]) for строка in read_metrics(прогон))
    assert len(значения) % 2 == 0, "на нечётном числе поз проверка бессмысленна"
    медиана = statistics.median(значения)
    верхняя = значения[len(значения) // 2]

    assert f"медиана {медиана:.3f}" in напечатано
    if f"{верхняя:.3f}" != f"{медиана:.3f}":
        assert f"медиана {верхняя:.3f}" not in напечатано


def test_папка_без_паспорта_отвергается(запуск: Any, tmp_path: Path, target_json: Path) -> None:
    пусто = tmp_path / "runs" / "нет-паспорта"
    пусто.mkdir(parents=True)

    with pytest.raises(SystemExit, match="run.json"):
        запуск("--run", str(пусто), "--target", str(target_json), "--runs-dir", str(пусто.parent))


def test_отсутствующий_пакет_мишени_отвергается(запуск: Any, прогон: Path) -> None:
    with pytest.raises(SystemExit, match="Нет пакета мишени"):
        запуск(
            "--run", str(прогон),
            "--target", str(прогон.parent / "нет-такого.json"),
            "--runs-dir", str(прогон.parent),
        )


def test_сверка_без_чисел_не_ставит_штамп(
    запуск: Any, прогон: Path, target_json: Path
) -> None:
    """У прогона нет ни метрик, ни ранжирования — сверять нечем, штамп невозможен."""
    with pytest.raises(SystemExit, match="нечем сверить"):
        запуск(
            "--run", str(прогон),
            "--target", str(target_json),
            "--runs-dir", str(прогон.parent),
            "--stamp-target",
        )


def test_смена_условия_отвергается(запуск: Any, прогон: Path, target_json: Path) -> None:
    """Числа из прогона могли попасть в текст, и метка не переписывается."""
    общие = ("--run", str(прогон), "--target", str(target_json), "--runs-dir", str(прогон.parent))
    запуск(*общие, "--condition", "первое")

    with pytest.raises(SystemExit, match="уже помечен условием"):
        запуск(*общие, "--condition", "второе")


def test_отказ_метрик_становится_сообщением(
    запуск: Any, прогон: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Команду зовёт человек: наружу идёт строка, а не трейсбек."""
    def падать(*_: object, **__: object) -> None:
        raise MetricsError("подделанный отказ метрик")

    monkeypatch.setattr(запуск.модуль, "compute_run_metrics", падать)

    with pytest.raises(SystemExit, match="Метрики не посчитаны"):
        запуск("--run", str(прогон), "--target", str(target_json), "--runs-dir", str(прогон.parent))


def test_отказ_скора_становится_сообщением(
    запуск: Any, прогон: Path, target_json: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def падать(*_: object, **__: object) -> None:
        raise ScoringError("подделанный отказ скора")

    monkeypatch.setattr(запуск.модуль, "score_run", падать)

    with pytest.raises(SystemExit, match="Скор не посчитан"):
        запуск("--run", str(прогон), "--target", str(target_json), "--runs-dir", str(прогон.parent))
