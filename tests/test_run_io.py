"""Запись папки прогона по форматам папки прогона и паспорта прогона.

Модуль общий для производителей SDF: локального набора поз и генерации
в Modal. Поэтому проверяется не «файл записался», а совместимость
с настоящим потребителем — сборкой отчёта `experiments.report`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from experiments.run_io import (
    FAILURES_CSV,
    MOLECULES_SDF,
    MoleculeRecord,
    RunExistsError,
    create_run_dir,
    make_run_id,
    write_failures,
    write_molecules,
    write_run_json,
)
from kinase_ifp.molecule_io import open_sdf


def _прочитать_sdf(path: Path) -> list[Chem.Mol]:
    """Читает SDF через поток: путь-строку с кириллицей RDKit на Windows не открывает."""
    with open_sdf(path) as поставщик:
        return [m for m in поставщик if m is not None]


def test_run_id_собран_по_правилу() -> None:
    """`<ГГГГ-ММ-ДД>-<pdb_id>-s<сила>-n<число>` — на этот вид опирается реестр прогонов."""
    assert make_run_id(pdb_id="6tgu", n=100, guidance_scale=0.0, day="2026-08-23") == (
        "2026-08-23-6tgu-s0-n100"
    )


def test_повторный_прогон_не_перезаписывает_существующий(tmp_path: Path) -> None:
    """Запрет №2: новый запуск — новая подпапка, старые результаты неприкосновенны."""
    папка = create_run_dir(tmp_path / "прогон")
    (папка / "molecules.sdf").write_text("", encoding="utf-8")

    with pytest.raises(RunExistsError):
        create_run_dir(tmp_path / "прогон")


def test_пустая_папка_после_вытеснения_прогону_не_мешает(tmp_path: Path) -> None:
    """Вытеснение контейнера в Modal оставляет пустой каталог и перезапускает функцию.

    16.09 на этом упал прогон `pocket-ids`: «Container terminated due to preemption»,
    перезапуск с тем же входом — и отказ на каталоге от первой попытки. Пустая папка
    результатом не является, поэтому неизменяемость её не защищает.
    """
    папка = create_run_dir(tmp_path / "прогон")

    assert create_run_dir(tmp_path / "прогон") == папка


def test_молекулы_несут_обязательные_sd_свойства(tmp_path: Path, crystal_ligand: Chem.Mol) -> None:
    """`mol_id` и `rmsd_to_ref` читаются обратно из SDF.

    Без `rmsd_to_ref` не считаются ROC-AUC, top-1 и монотонность, то есть план Б
    существует только на бумаге.
    """
    run_dir = create_run_dir(tmp_path / "прогон")
    write_molecules(
        run_dir,
        [
            MoleculeRecord(crystal_ligand, "прогон-0001", 0.0),
            MoleculeRecord(crystal_ligand, "прогон-0002", 1.25),
        ],
    )

    записанные = _прочитать_sdf(run_dir / MOLECULES_SDF)

    assert [m.GetProp("mol_id") for m in записанные] == ["прогон-0001", "прогон-0002"]
    assert [m.GetProp("rmsd_to_ref") for m in записанные] == ["0.0", "1.25"]


def test_водороды_переживают_запись_в_sdf(tmp_path: Path, crystal_ligand: Chem.Mol) -> None:
    """SDF читается обратно с явными водородами: без них ProLIF не отличит донор от акцептора."""
    run_dir = create_run_dir(tmp_path / "прогон")
    write_molecules(run_dir, [MoleculeRecord(crystal_ligand, "прогон-0001", 0.0)])

    прочитанная = _прочитать_sdf(run_dir / MOLECULES_SDF)[0]

    assert прочитанная.GetNumAtoms() > прочитанная.GetNumHeavyAtoms()


def test_молекула_без_эталона_несёт_пустой_rmsd(tmp_path: Path, crystal_ligand: Chem.Mol) -> None:
    """`source=diffsbdd`: эталонной позы нет, свойство присутствует и пусто.

    Так же выглядит запись без дополнительных свойств — путь генерации, у которой
    `pose_kind` не бывает.
    """
    run_dir = create_run_dir(tmp_path / "прогон")
    write_molecules(run_dir, [MoleculeRecord(crystal_ligand, "прогон-0001")])

    записанная = _прочитать_sdf(run_dir / MOLECULES_SDF)[0]

    assert записанная.GetProp("rmsd_to_ref") == ""
    assert not записанная.HasProp("pose_kind")


def test_дополнительные_свойства_доезжают_до_sdf(tmp_path: Path, crystal_ligand: Chem.Mol) -> None:
    """`props` пишутся рядом с обязательными: на них опирается сводка (`pose_kind`)."""
    run_dir = create_run_dir(tmp_path / "прогон")
    write_molecules(
        run_dir,
        [MoleculeRecord(crystal_ligand, "прогон-0001", 0.0, {"pose_kind": "native"})],
    )

    записанная = _прочитать_sdf(run_dir / MOLECULES_SDF)[0]

    assert записанная.GetProp("pose_kind") == "native"
    assert записанная.GetProp("mol_id") == "прогон-0001"


def test_файл_сбоев_создаётся_даже_пустым(tmp_path: Path) -> None:
    """«Файла нет» неотличимо от «сбоев не было», поэтому заголовок пишется всегда."""
    run_dir = create_run_dir(tmp_path / "прогон")
    write_failures(run_dir, [])

    строки = (run_dir / FAILURES_CSV).read_text(encoding="utf-8").splitlines()

    assert строки == ["mol_id,stage,reason"]


def test_паспорт_прогона_читается_сборкой_отчёта(tmp_path: Path) -> None:
    """`run.json` принимает `experiments.report.load_run` — настоящий потребитель паспорта."""
    import pandas as pd

    from experiments.report import METRICS_CSV, load_run

    run_dir = create_run_dir(tmp_path / "прогон")
    write_run_json(
        run_dir,
        run_id="2026-08-23-6tgu-s0-n2",
        target={"pdb_id": "6tgu", "klifs_structure_id": 12448},
        source="local-poses",
        sampling={"n_requested": 2, "n_returned": 2, "seed": 0, "guidance_scale": 0.0},
        duration_sec=1,
        project_commit="0000000",
    )
    pd.DataFrame(
        {
            "run_id": ["2026-08-23-6tgu-s0-n2"],
            "mol_id": ["m1"],
            "condition": ["baseline"],
            "valid": [1],
        }
    ).to_csv(run_dir / METRICS_CSV, index=False)

    колонки = load_run(run_dir)

    assert [c.run_id for c in колонки] == ["2026-08-23-6tgu-s0-n2"]
    assert [c.source for c in колонки] == ["local-poses"]


def test_повтор_прогона_различим_по_суффиксу() -> None:
    """Повторы одного дня с разными сидами обязаны получать разные `run_id`.

    В самом формате паспорта прогона сида нет, поэтому десять наборов за 23.08 столкнулись бы
    именами. Суффикс — то же расширение, которым уже пользуется сборка отчёта
    (`...-n5-top2` в его фикстурах).
    """
    базовый = make_run_id(pdb_id="6tgu", n=100, guidance_scale=0.0, day="2026-08-23")
    повтор = make_run_id(
        pdb_id="6tgu", n=100, guidance_scale=0.0, day="2026-08-23", suffix="r3"
    )

    assert базовый == "2026-08-23-6tgu-s0-n100"
    assert повтор == "2026-08-23-6tgu-s0-n100-r3"
