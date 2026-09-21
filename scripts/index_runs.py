"""CLI: пересборка реестра прогонов `runs/index.csv`.

Запуск:
    docker compose run --rm dev python scripts/index_runs.py

Реестр производный: он собирается по содержимому `runs/`, поэтому пересобрать его
можно в любой момент, и это не считается перезаписью прогона (неизменяемость — про сами
папки, а не про их опись).

Прогон без обязательных файлов папки или без паспорта попадает в реестр
со статусом `failed`,
а не пропускается: молча исчезнувшая папка выглядит как «эксперимента не было».

Прогон целиком одной командой собирает `scripts/run_experiment.py`; здесь только
пересборка описи.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.registry import INDEX_CSV, RegistryError, rebuild_index
from kinase_ifp.config import RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="каталог прогонов (по умолчанию runs/ в корне репозитория)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        строки = rebuild_index(args.runs_dir)
    except RegistryError as ошибка:
        raise SystemExit(
            f"{ошибка}. Прогоны создаются генерацией или набором положений, "
            "а забираются на диск через scripts/pull_run.py"
        ) from ошибка

    for строка in строки:
        print(f"{строка['status']:<10} {строка['run_id']:<34} {строка['source']}")
    print(f"\nвсего прогонов: {len(строки)}, реестр: {args.runs_dir / INDEX_CSV}")


if __name__ == "__main__":
    main()
