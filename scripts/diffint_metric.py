"""Метрика воспроизведения водородных связей в определении DiffInt.

    docker compose run --rm dev python scripts/diffint_metric.py \
        --run runs/2026-08-24-6tgu-s0-n100 --target data/targets/6tgu/target.json

Зачем отдельный вход. Наш `ifp_score` — доля ключевых взаимодействий эталона по шести
типам, а DiffInt (Sako et al., JCIM 2025) считает **только водородные
связи**: долю H-связей эталонного лиганда, воспроизведённых сгенерированной молекулой,
усреднённую по молекулам. Их число для DiffSBDD — 34.0 %, и сопоставлять с ним можно
лишь величину, посчитанную по тому же правилу. Новой логики здесь нет: это тот же
`tversky(эталон, молекула, alpha=1, beta=0)`, суженный до типов `HBOND_TYPES`.

Чего сравнение не даёт. Их 34.0 % получены на 100 карманах CrossDocked, наши — на
киназном кармане KLIFS, а H-связи у них размечает свой детектор, у нас — правила KLIFS
(3.5 A, отклонение D-H до 45°). Совпадение определения метрики не делает выборки
взаимозаменяемыми, и в тексте это оговаривается рядом с числом.

Знаменатель — число H-связевых бит эталона. Если их нет вовсе, мишень пропускается:
делить не на что, а вернуть 0 или 1 значило бы выдумать величину.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.layout import MOLECULES_SDF  # noqa: E402
from kinase_ifp.config import HBOND_TYPES  # noqa: E402
from kinase_ifp.fingerprint import bits_to_ifp  # noqa: E402
from kinase_ifp.fingerprint_klifs import (  # noqa: E402
    PocketRules,
    ifp_from_groups,
    load_pocket_rules,
)
from kinase_ifp.ligand_flags import groups_from_mol  # noqa: E402
from kinase_ifp.molecule_io import open_sdf  # noqa: E402
from kinase_ifp.similarity import select_types, tversky  # noqa: E402


def доли_по_молекулам(sdf: Path, карман: PocketRules, эталон_hb: np.ndarray) -> np.ndarray:
    """Доля H-связей эталона, воспроизведённая каждой молекулой файла."""
    доли: list[float] = []
    with open_sdf(sdf) as supplier:
        for mol in supplier:
            if mol is None:
                continue
            отпечаток = ifp_from_groups(groups_from_mol(mol), карман)
            доли.append(
                tversky(эталон_hb, select_types(отпечаток, HBOND_TYPES), alpha=1.0, beta=0.0)
            )
    return np.array(доли, dtype=float)


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument("--run", type=Path, required=True, help="папка прогона генерации")
    разбор.add_argument("--target", type=Path, required=True, help="target.json пакета мишени")
    разбор.add_argument(
        "--poses",
        type=Path,
        help="SDF оптимизированных поз; без него считается только исходный прогон",
    )
    аргументы = разбор.parse_args()

    пакет = json.loads(аргументы.target.read_text(encoding="utf-8"))
    эталон = bits_to_ifp(пакет["klifs_ifp_bits"])
    эталон_hb = select_types(эталон, HBOND_TYPES)
    связей = int(эталон_hb.sum())
    if связей == 0:
        print(
            f"у эталона {аргументы.target.parent.name} нет H-связей: метрика не определена",
            file=sys.stderr,
        )
        return 1

    карман = load_pocket_rules(аргументы.target)
    до = доли_по_молекулам(аргументы.run / MOLECULES_SDF, карман, эталон_hb)
    print(f"мишень {аргументы.target.parent.name}, H-связей у эталона {связей}")
    print(f"исходные молекулы DiffSBDD: n={len(до)}, метрика DiffInt {до.mean():.3f}")
    print(f"  воспроизвели хотя бы одну: {(до > 0).mean():.3f}; все: {(до == 1).mean():.3f}")

    if аргументы.poses is not None:
        после = доли_по_молекулам(аргументы.poses, карман, эталон_hb)
        print(f"после дискретного потенциала: n={len(после)}, метрика {после.mean():.3f}")
        print(
            f"  воспроизвели хотя бы одну: {(после > 0).mean():.3f}; "
            f"все: {(после == 1).mean():.3f}"
        )
        if len(до) == len(после):
            прирост = после - до
            print(f"  прирост на молекулу: среднее {прирост.mean():+.3f}, "
                  f"улучшилось {(прирост > 0).sum()}, ухудшилось {(прирост < 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
