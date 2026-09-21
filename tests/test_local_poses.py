"""Сборка прогона на локальных положениях лиганда.

Проверяется главное: папка прогона получается той же формы, что у генерации,
и расчёт скора читает её, не зная, откуда взялись молекулы.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from experiments.layout import SOURCE_LOCAL_POSES
from experiments.local_poses import build_local_pose_run
from experiments.run_io import FAILURES_CSV, MOLECULES_SDF, RUN_JSON
from kinase_ifp.molecule_io import open_sdf


@pytest.fixture(scope="module")
def прогон(target_json: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Небольшой прогон: двадцать поз проверяют всё то же, что и сотня, но быстрее."""
    return build_local_pose_run(
        target_json=target_json,
        n=20,
        seed=0,
        runs_dir=tmp_path_factory.mktemp("runs"),
        day="2026-08-23",
    )


def _молекулы(run_dir: Path) -> list[Chem.Mol]:
    """SDF читается через поток: путь-строку с кириллицей RDKit на Windows не открывает."""
    with open_sdf(run_dir / MOLECULES_SDF) as поставщик:
        return [m for m in поставщик if m is not None]


def test_прогон_назван_по_правилу(прогон: Path) -> None:
    """Имя папки — `<ГГГГ-ММ-ДД>-<pdb_id>-s<сила>-n<число>`."""
    assert прогон.name == "2026-08-23-6tgu-s0-n20"


def test_в_прогоне_три_обязательных_файла(прогон: Path) -> None:
    """Молекулы, паспорт и файл сбоев — последний создаётся даже пустым."""
    for имя in (MOLECULES_SDF, RUN_JSON, FAILURES_CSV):
        assert (прогон / имя).exists()


def test_записаны_все_запрошенные_молекулы(прогон: Path) -> None:
    """Запрошено n молекул — в SDF ровно n."""
    assert len(_молекулы(прогон)) == 20


def test_у_каждой_молекулы_есть_mol_id_и_rmsd(прогон: Path) -> None:
    """Оба свойства обязательны; без `rmsd_to_ref` различение положений не считается."""
    молекулы = _молекулы(прогон)

    assert [m.GetProp("mol_id") for m in молекулы][:2] == [
        "2026-08-23-6tgu-s0-n20-0001",
        "2026-08-23-6tgu-s0-n20-0002",
    ]
    assert all(m.GetProp("rmsd_to_ref") != "" for m in молекулы)
    assert float(молекулы[0].GetProp("rmsd_to_ref")) == 0.0


def test_у_каждой_молекулы_явные_водороды(прогон: Path) -> None:
    """У каждой молекулы есть водороды: `GetNumAtoms() > GetNumHeavyAtoms()`."""
    for mol in _молекулы(прогон):
        assert mol.GetNumAtoms() > mol.GetNumHeavyAtoms()


def test_у_каждой_молекулы_записан_способ_построения(прогон: Path) -> None:
    """`pose_kind` доезжает до SDF: сводка обязана отличать сдвиг от смены торсий.

    Внутри объекта `Pose` это поле было и раньше, но в файл не попадало, и потребитель
    видел лестницу RMSD без указания, чем именно испорчена поза.
    """
    молекулы = _молекулы(прогон)
    виды = [m.GetProp("pose_kind") for m in молекулы]

    assert виды[0] == "native"
    assert set(виды[1:]) <= {"rigid", "conformer"}
    assert "conformer" in виды[1:], "половина ступеней обязана идти на других конформерах"


def test_паспорт_помечает_источник_как_локальные_позы(прогон: Path) -> None:
    """Без `source` в отчёте не отличить набор поз от настоящей генерации.

    Это разные утверждения в тексте курсовой: одно про потенциал, другое про модель.
    """
    паспорт = json.loads((прогон / RUN_JSON).read_text(encoding="utf-8"))

    assert паспорт["source"] == SOURCE_LOCAL_POSES
    assert паспорт["target"]["pdb_id"] == "6tgu"
    assert паспорт["sampling"]["n_requested"] == 20
    assert паспорт["sampling"]["n_returned"] == 20
    assert паспорт["sampling"]["seed"] == 0
    assert паспорт["model"] == {"repo_commit": None, "checkpoint": None}


def test_повторный_прогон_не_затирает_прежний(прогон: Path, target_json: Path) -> None:
    """Запрет №2 действует и на уровне сборки, а не только внутри `run_io`."""
    from experiments.run_io import RunExistsError

    with pytest.raises(RunExistsError):
        build_local_pose_run(
            target_json=target_json,
            n=20,
            seed=0,
            runs_dir=прогон.parent,
            day="2026-08-23",
        )
