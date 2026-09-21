"""Тесты метрик плана Б: ROC-AUC, ранг нативной позы, Спирмен.

Набор строится синтетическим и с известным ответом: скор задан функцией RMSD, поэтому
правильные значения метрик известны заранее и не берутся из самого расчёта. Прогон
пишется теми же функциями, какими его пишет производитель поз (`experiments.run_io`),
а не выкладывается текстом файла — иначе тест проверял бы согласие с собственной
копией формата папки прогона, а не с тем, что лежит в `runs/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from evaluation.discrimination import (
    DISCRIMINATION_COLUMNS,
    TARGET_SUMMARY_COLUMNS,
    DiscriminationError,
    discriminate_run,
    roc_auc,
    summarize_by_target,
    top1_rate,
    write_discrimination_csv,
    write_plan_b_metrics,
    write_target_summary,
)
from experiments.layout import (
    DISCRIMINATION_SUMMARY_CSV,
    DISCRIMINATION_SUMMARY_MD,
    PLAN_B_METRICS_JSON,
    SOURCE_DIFFSBDD,
    SOURCE_LOCAL_POSES,
)
from experiments.run_io import MoleculeRecord, write_molecules, write_run_json
from kinase_ifp.config import CORRECT_POSE_RMSD_A, NEAR_POSE_RMSD_A, RMSD_BIN_EDGES_A
from kinase_ifp.poses import POSE_KIND_NATIVE, POSE_KIND_RIGID


def _прогон(
    каталог: Path,
    позы: list[tuple[str, float, float]],
    *,
    source: str = SOURCE_LOCAL_POSES,
    native: str | None = "m1",
    seed: int = 0,
    pdb_id: str = "6tgu",
) -> Path:
    """Папка прогона из троек «mol_id, скор, rmsd_to_ref».

    Молекула берётся простейшая: метрики считаются по свойствам SDF и `ranking.csv`,
    а не по геометрии, и настоящая структура ничего к проверке не добавила бы.
    """
    каталог.mkdir(parents=True, exist_ok=True)
    молекула = Chem.MolFromSmiles("CCO")

    write_run_json(
        каталог,
        run_id=каталог.name,
        target={"pdb_id": pdb_id, "protonation": "explicit"},
        source=source,
        sampling={"n_requested": len(позы), "n_returned": len(позы), "seed": seed},
        duration_sec=1,
        project_commit="0" * 40,
    )
    write_molecules(
        каталог,
        [
            MoleculeRecord(
                mol=молекула,
                mol_id=mol_id,
                rmsd_to_ref=rmsd,
                props={
                    "pose_kind": POSE_KIND_NATIVE if mol_id == native else POSE_KIND_RIGID
                },
            )
            for mol_id, _, rmsd in позы
        ],
    )
    with (каталог / "ranking.csv").open("w", encoding="utf-8", newline="\n") as файл:
        писатель = csv.writer(файл)
        писатель.writerow(["mol_id", "ifp_score", "ifp_score_norm", "rank", "selected"])
        порядок = sorted(позы, key=lambda строка: -строка[1])
        for место, (mol_id, скор, _) in enumerate(порядок, start=1):
            писатель.writerow([mol_id, скор, скор / 20, место, 1 if место <= 2 else 0])
    return каталог


def _лестница(n: int = 11) -> list[tuple[str, float, float]]:
    """Набор, где скор строго убывает с ростом отклонения: известный ответ — AUC 1.0."""
    позы = [("m1", 1.0, 0.0)]
    for номер in range(2, n + 1):
        rmsd = 0.5 * (номер - 1)
        позы.append((f"m{номер}", round(1.0 - rmsd / 10, 3), rmsd))
    return позы


def test_скор_как_функция_rmsd_даёт_roc_auc_единицу(tmp_path: Path) -> None:
    измерение = discriminate_run(_прогон(tmp_path / "run", _лестница()))

    assert измерение.roc_auc == 1.0
    assert измерение.spearman_rho == pytest.approx(-1.0)
    assert измерение.native_rank == 1
    assert измерение.native_ties == 0
    assert измерение.native_is_best


def test_классы_считаются_по_границам_из_конфига(tmp_path: Path) -> None:
    """Поза из полосы между границами не входит ни в один класс ROC-AUC."""
    позы = [
        ("m1", 1.0, 0.0),
        ("m2", 0.8, NEAR_POSE_RMSD_A / 2),
        ("m3", 0.6, (NEAR_POSE_RMSD_A + CORRECT_POSE_RMSD_A) / 2),
        ("m4", 0.2, CORRECT_POSE_RMSD_A + 1.0),
        ("m5", 0.0, CORRECT_POSE_RMSD_A + 2.0),
    ]

    измерение = discriminate_run(_прогон(tmp_path / "run", позы))

    assert (измерение.n_near, измерение.n_far) == (2, 2)
    assert измерение.n_poses == 5
    # Граница «верной позы» и верхний край бинов рисунка — одно число, а не два
    # совпадающих: иначе рисунок и метрика разделят набор по-разному.
    assert RMSD_BIN_EDGES_A == (NEAR_POSE_RMSD_A, CORRECT_POSE_RMSD_A)


def test_ничья_на_вершине_не_считается_победой(tmp_path: Path) -> None:
    """Ранг 1 при ничьих означает «скор максимален», а не «нативная поза выделена»."""
    позы = [("m1", 1.0, 0.0), ("m2", 1.0, 3.0), ("m3", 0.4, 4.0), ("m4", 0.2, 5.0)]

    измерение = discriminate_run(_прогон(tmp_path / "run", позы))

    assert измерение.native_rank == 1
    assert измерение.native_ties == 1
    assert not измерение.native_is_best


def test_top1_считается_долей_наборов(tmp_path: Path) -> None:
    хороший = discriminate_run(_прогон(tmp_path / "r1", _лестница(), seed=1))
    ничья = discriminate_run(
        _прогон(
            tmp_path / "r2",
            [("m1", 1.0, 0.0), ("m2", 1.0, 3.0), ("m3", 0.2, 4.0), ("m4", 0.0, 5.0)],
            seed=2,
        )
    )

    assert top1_rate([хороший, ничья]) == (1, 2)


def test_прогон_генерации_отвергается(tmp_path: Path) -> None:
    каталог = _прогон(tmp_path / "run", _лестница(), source=SOURCE_DIFFSBDD)

    with pytest.raises(DiscriminationError, match="набору поз"):
        discriminate_run(каталог)


def test_набор_без_нативной_позы_отвергается(tmp_path: Path) -> None:
    позы = [("m1", 0.8, 1.5), ("m2", 0.6, 3.0), ("m3", 0.2, 4.0)]
    каталог = _прогон(tmp_path / "run", позы, native=None)

    with pytest.raises(DiscriminationError, match="нативная поза не найдена"):
        discriminate_run(каталог)


def test_один_класс_не_даёт_roc_auc() -> None:
    with pytest.raises(DiscriminationError, match="ROC-AUC не определён"):
        roc_auc([0.8, 0.6], [])


def test_метрики_пишутся_в_файл_прогона(tmp_path: Path) -> None:
    каталог = _прогон(tmp_path / "run", _лестница())
    измерение = discriminate_run(каталог)

    путь = write_plan_b_metrics(каталог, измерение)

    assert путь.name == PLAN_B_METRICS_JSON
    записано = json.loads(путь.read_text(encoding="utf-8"))
    assert записано["run_id"] == каталог.name
    assert float(записано["roc_auc"]) == 1.0


def test_сводка_пишется_колонками_модуля(tmp_path: Path) -> None:
    измерения = [
        discriminate_run(_прогон(tmp_path / "r1", _лестница(), seed=1)),
        discriminate_run(_прогон(tmp_path / "r2", _лестница(), seed=2)),
    ]

    путь = write_discrimination_csv(измерения, tmp_path / "свод" / "pose_discrimination.csv")

    текст = путь.read_text(encoding="utf-8")
    assert "\r\n" not in текст  # CSV_EOL: перевод строки в CSV только LF
    строки = list(csv.DictReader(текст.splitlines()))
    assert tuple(строки[0]) == DISCRIMINATION_COLUMNS
    assert [строка["seed"] for строка in строки] == ["1", "2"]


def test_сводка_по_мишеням_не_смешивает_их(tmp_path: Path) -> None:
    """Медиана считается внутри мишени: сходство меряется к своему эталону."""
    измерения = [
        discriminate_run(_прогон(tmp_path / f"a{сид}", _лестница(), seed=сид))
        for сид in (1, 2)
    ]
    чужая = discriminate_run(
        _прогон(
            tmp_path / "b1",
            [("m1", 1.0, 0.0), ("m2", 1.0, 3.0), ("m3", 0.2, 4.0), ("m4", 0.0, 5.0)],
            seed=1,
            pdb_id="3war",
        )
    )

    сводка = summarize_by_target([*измерения, чужая])

    assert [с.pdb_id for с in сводка] == ["3war", "6tgu"]
    по_мишеням = {с.pdb_id: с for с in сводка}
    assert по_мишеням["6tgu"].n_runs == 2
    assert по_мишеням["6tgu"].roc_auc == (1.0, 1.0, 1.0)
    assert по_мишеням["6tgu"].top1_strict == 2
    # У второй мишени нативная поза делит максимум, поэтому строгих побед нет,
    # а место всё равно первое — две разные величины, и обе видны в сводке.
    assert по_мишеням["3war"].top1_strict == 0
    assert по_мишеням["3war"].native_rank_max == 1


def test_сводка_пишется_парой_файлов_в_results(tmp_path: Path) -> None:
    """У числа одно машинное место и читаемый двойник."""
    измерения = [discriminate_run(_прогон(tmp_path / "run", _лестница()))]

    md, csv_путь = write_target_summary(summarize_by_target(измерения), tmp_path / "results")

    assert (md.name, csv_путь.name) == (DISCRIMINATION_SUMMARY_MD, DISCRIMINATION_SUMMARY_CSV)
    текст = csv_путь.read_text(encoding="utf-8")
    assert "\r\n" not in текст
    строки = list(csv.DictReader(текст.splitlines()))
    assert tuple(строки[0]) == TARGET_SUMMARY_COLUMNS
    assert float(строки[0]["roc_auc_median"]) == 1.0
    assert "6tgu" in md.read_text(encoding="utf-8")


def test_сводка_по_пустому_списку_отказывает() -> None:
    with pytest.raises(DiscriminationError, match="сводить нечего"):
        summarize_by_target([])
