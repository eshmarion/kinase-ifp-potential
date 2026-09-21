"""CLI: манифест выгрузки KLIFS — сборка и сверка с файлами каталога.

Запуск:
    docker compose run --rm dev python scripts/klifs_manifest.py --check
    docker compose run --rm dev python scripts/klifs_manifest.py --write

`--check` ничего не меняет: говорит, описывает ли манифест то, что лежит в каталоге.
Ту же сверку делает `pytest` (`tests/test_manifest.py`), поэтому отставший манифест
валит проверки, а не обнаруживается через неделю.

`--write` пересобирает манифест по фактическому содержимому каталога. Запускать его
нужно после каждой новой выгрузки — и в том же коммите, что и сами файлы.

Логики здесь нет: разбор аргументов и вызов `kinase_ifp.manifest`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kinase_ifp.manifest import KLIFS_DIR, check_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    режим = parser.add_mutually_exclusive_group(required=True)
    режим.add_argument("--write", action="store_true", help="пересобрать и записать манифест")
    режим.add_argument("--check", action="store_true", help="сверить манифест с файлами каталога")
    parser.add_argument(
        "--klifs-dir",
        type=Path,
        default=KLIFS_DIR,
        help=f"каталог выгрузки (по умолчанию {KLIFS_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write:
        путь = write_manifest(args.klifs_dir)
        print(f"Манифест записан: {путь}")
        return

    расхождения = check_manifest(args.klifs_dir)
    if not расхождения:
        print(f"Манифест соответствует каталогу {args.klifs_dir}")
        return

    print(f"Манифест разошёлся с каталогом {args.klifs_dir}:")
    for строка in расхождения:
        print(f"  - {строка}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
