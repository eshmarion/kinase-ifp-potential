"""Прогон на локальных позах: набор поз плюс папка прогона.

Связывает генерацию поз (`kinase_ifp.poses`) с записью по форматам папки прогона и паспорта прогона
(`experiments.run_io`). Отдельный модуль, а не код внутри скрипта: критерий «готово»
модуля проверяется тестом, а логика в скрипте не тестируется.

Прогон намеренно не зависит от DiffSBDD и от GPU — именно это делает 23.08 независимым
от сборки образа Modal.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.layout import SOURCE_LOCAL_POSES
from experiments.run_io import (
    PLATFORM_LOCAL_CPU,
    MoleculeRecord,
    create_run_dir,
    make_run_id,
    write_failures,
    write_molecules,
    write_run_json,
)
from experiments.runs import target_stamp
from kinase_ifp.poses import generate_poses

# Сила guidance у набора поз всегда нулевая: модели, которую можно было бы направлять,
# в этом прогоне нет. Ноль пишется в `run_id`, чтобы имя читалось так же, как у прогонов
# генерации, и реестр не пришлось учить двум форматам.
LOCAL_POSES_GUIDANCE_SCALE: float = 0.0


def build_local_pose_run(
    target_json: Path,
    n: int,
    seed: int,
    runs_dir: Path,
    day: str | None = None,
) -> Path:
    """Собирает прогон на локальных позах и возвращает путь к его папке.

    Принимает `target.json` пакета мишени, размер набора, сид, каталог для прогонов
    и дату для `run_id` (по умолчанию сегодняшняя); возвращает созданную папку
    с `molecules.sdf`, `run.json` и `failures.csv`.

    Поднимает `RunExistsError`, если такая папка уже есть: перезаписывать прогоны
    запрещено.
    """
    package = json.loads(target_json.read_text(encoding="utf-8"))
    pdb_id = str(package["pdb_id"])
    ligand_path = target_json.parent / str(package["ligand_sdf"])

    run_id = make_run_id(
        pdb_id=pdb_id,
        n=n,
        guidance_scale=LOCAL_POSES_GUIDANCE_SCALE,
        day=day if day is not None else time.strftime("%Y-%m-%d"),
        # Нулевой сид даёт каноническое имя прогона, повторы различаются суффиксом:
        # иначе десять наборов за один день столкнулись бы именами папок.
        suffix=None if seed == 0 else f"r{seed}",
    )
    # Папка создаётся до расчёта: столкновение имён должно всплыть сразу, а не после
    # нескольких минут генерации.
    run_dir = create_run_dir(runs_dir / run_id)

    started = time.monotonic()
    pose_set = generate_poses(ligand_path, n=n, seed=seed)

    written = write_molecules(
        run_dir,
        (
            MoleculeRecord(
                mol=pose.mol,
                mol_id=f"{run_id}-{index:04d}",
                rmsd_to_ref=pose.rmsd_to_ref,
                # Способ построения доезжает до файла, а не остаётся в памяти: без него
                # сводка не отличит чувствительность к размещению от чувствительности
                # к торсиям, а лестница строится и тем, и другим приёмом.
                props={"pose_kind": pose.kind},
            )
            for index, pose in enumerate(pose_set.poses, start=1)
        ),
    )
    write_failures(
        run_dir,
        (
            (f"{run_id}-{failure.index:04d}", failure.stage, failure.reason)
            for failure in pose_set.failures
        ),
    )
    write_run_json(
        run_dir,
        run_id=run_id,
        target=target_stamp(target_json),
        source=SOURCE_LOCAL_POSES,
        sampling={
            "n_requested": n,
            "n_returned": written,
            "seed": seed,
            "guidance_scale": LOCAL_POSES_GUIDANCE_SCALE,
            "timesteps": None,
        },
        platform=PLATFORM_LOCAL_CPU,
        duration_sec=int(time.monotonic() - started),
    )
    return run_dir
