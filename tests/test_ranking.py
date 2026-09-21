"""Тесты переранжирования прогона."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.local_poses import build_local_pose_run
from experiments.ranking import RANKING_COLUMNS, read_ranking, score_run
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.scoring import ScoringError


@pytest.fixture(scope="module")
def прогон(tmp_path_factory: pytest.TempPathFactory, target_json: Path) -> Path:
    """Небольшой набор поз: нативная плюс искажённые копии."""
    runs = tmp_path_factory.mktemp("runs")
    return build_local_pose_run(target_json, n=6, seed=1, runs_dir=runs, day="2026-08-24")


def test_кристаллическая_поза_получает_ранг_один(прогон: Path, target_json: Path) -> None:
    """Скор обязан ставить нативное положение первым.

    Если он этого не делает, переранжирование не имеет смысла — отбирать по нему
    хорошие позы нельзя, и вся таблица «было/стало» повисает.
    """
    score_run(прогон, target_json)

    строки = read_ranking(прогон)
    with open_sdf(прогон / "molecules.sdf") as поставщик:
        rmsd = {
            mol.GetProp("mol_id"): float(mol.GetProp("rmsd_to_ref"))
            for mol in поставщик
            if mol is not None
        }
    первая = min(строки, key=lambda строка: int(строка["rank"]))

    assert rmsd[первая["mol_id"]] == pytest.approx(0.0)
    assert float(первая["ifp_score"]) == pytest.approx(1.0)


def test_ranking_csv_по_заголовку(прогон: Path, target_json: Path) -> None:
    score_run(прогон, target_json)

    строки = read_ranking(прогон)
    assert строки
    assert tuple(строки[0]) == RANKING_COLUMNS
    # Ранги — перестановка 1..N: пропуск или повтор означал бы потерянную молекулу.
    assert sorted(int(строка["rank"]) for строка in строки) == list(range(1, len(строки) + 1))


def test_размер_топа_записан_в_паспорт(прогон: Path, target_json: Path) -> None:
    """Размер топа живёт в run.json, иначе таблицу не воспроизвести."""
    score_run(прогон, target_json, top_fraction=0.5)

    паспорт = json.loads((прогон / "run.json").read_text(encoding="utf-8"))
    assert паспорт["ranking"]["top_fraction"] == pytest.approx(0.5)
    assert паспорт["ranking"]["n_selected"] == sum(
        int(строка["selected"]) for строка in read_ranking(прогон)
    )


def test_пересчёт_с_другой_долей_меняет_только_отбор(прогон: Path, target_json: Path) -> None:
    """На пересчёте стоит перебор долей: скор и ранг от доли топа не зависят, отбор — да."""
    score_run(прогон, target_json, top_fraction=0.5)
    половина = {строка["mol_id"]: dict(строка) for строка in read_ranking(прогон)}

    score_run(прогон, target_json, top_fraction=0.25)
    четверть = {строка["mol_id"]: dict(строка) for строка in read_ranking(прогон)}

    assert {m: с["ifp_score"] for m, с in половина.items()} == {
        m: с["ifp_score"] for m, с in четверть.items()
    }
    assert {m: с["rank"] for m, с in половина.items()} == {
        m: с["rank"] for m, с in четверть.items()
    }

    def отобрано(таблица: dict[str, dict[str, str]]) -> int:
        return sum(int(с["selected"]) for с in таблица.values())

    assert отобрано(четверть) < отобрано(половина)


def test_прогон_без_молекул_отвергается(tmp_path: Path, target_json: Path) -> None:
    пустой = tmp_path / "2026-08-24-6tgu-s0-n0"
    пустой.mkdir()

    with pytest.raises(ScoringError, match="molecules.sdf"):
        score_run(пустой, target_json)


def test_чужой_пакет_мишени_роняет_расчёт(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, target_json: Path
) -> None:
    """Отпечатки разных режимов протонирования несравнимы.

    24.08 это стоило дня: на одной машине пакет стоял в `explicit`, на другой
    в `implicit-prolif`, числа разошлись вдвое и спокойно попали в таблицу
    как сопоставимые. Теперь расчёт по чужому пакету — ошибка, а не предупреждение.
    """
    runs = tmp_path_factory.mktemp("runs")
    прогон = build_local_pose_run(target_json, n=4, seed=1, runs_dir=runs, day="2026-08-24")

    подменённый = tmp_path / "6tgu"
    shutil.copytree(target_json.parent, подменённый)
    пакет = json.loads((подменённый / "target.json").read_text(encoding="utf-8"))
    пакет["protonation"] = "implicit-prolif"
    (подменённый / "target.json").write_text(
        json.dumps(пакет, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ScoringError, match="другом пакете мишени"):
        score_run(прогон, подменённый / "target.json")


def test_прогон_без_штампа_считается_и_записывает_пакет(
    tmp_path_factory: pytest.TempPathFactory, target_json: Path
) -> None:
    """Прогон, сделанный до появления штампа, не отвергается, но пакет расчёта записывается."""
    runs = tmp_path_factory.mktemp("runs")
    прогон = build_local_pose_run(target_json, n=4, seed=1, runs_dir=runs, day="2026-08-24")
    паспорт_путь = прогон / "run.json"
    паспорт = json.loads(паспорт_путь.read_text(encoding="utf-8"))
    паспорт["target"] = {"pdb_id": паспорт["target"]["pdb_id"]}
    паспорт_путь.write_text(json.dumps(паспорт, ensure_ascii=False), encoding="utf-8")

    score_run(прогон, target_json)

    записано = json.loads(паспорт_путь.read_text(encoding="utf-8"))
    assert записано["scoring"]["protonation"] == "explicit"
    assert len(записано["scoring"]["package_sha256"]) == 64
