"""CLI: фиксация режима протонирования белка мишени.

Запуск (после scripts/select_target.py):
    uv run python scripts/annotate_protonation.py --target data/targets/6tgu/target.json

Скрипт перестраивает PDB-файлы пакета из лежащих рядом mol2 (сеть не нужна),
создаёт белок без водородов для второго режима сверки и проставляет в `target.json`
поля `protonation` и `protonation_tool`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kinase_ifp.config import DATA_DIR
from kinase_ifp.protonation import annotate_protonation
from kinase_ifp.structure_io import PROTEIN_NOH_PDB, PROTEIN_PDB, count_hydrogens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help=f"путь к target.json мишени, например {DATA_DIR / 'targets' / '6tgu' / 'target.json'}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.target.is_file():
        raise SystemExit(
            f"Нет файла {args.target}. Сначала запустите scripts/select_target.py"
        )

    target_dir = args.target.parent
    package = annotate_protonation(args.target)

    print(f"Мишень: {package['pdb_id']}, цепь {package['chain']}")
    print(f"Водородов в {PROTEIN_PDB}: {count_hydrogens(target_dir / PROTEIN_PDB)}")
    print(f"Водородов в {PROTEIN_NOH_PDB}: {count_hydrogens(target_dir / PROTEIN_NOH_PDB)}")
    print(f"protonation: {package['protonation']}")
    print(f"protonation_tool: {package['protonation_tool']}")
    print(f"Пакет мишени обновлён: {args.target}")


if __name__ == "__main__":
    main()
