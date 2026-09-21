"""Расхождение альтернативных моделей кристалла и доли заселённости.

    docker compose run --rm dev python scripts/alternate_models.py --pdb-id 6tgu \
        --target data/targets/6tgu/target.json

Зачем нужен отдельный вход. Решение №16 («из альтернативных моделей берётся первая
по алфавиту») обосновано числом: у 6tgu боковая цепь HIS161 между моделями A и B
разворачивается на несколько ангстрем. Само число посчитали 14.09 вручную и нигде
не сохранили ни его, ни способ расчёта, поэтому «Материалы и методы» ссылались
на величину, которую нельзя было ни проверить, ни пересчитать.

Исходная запись PDB в git не хранится (`data/rcsb-cache/` закрыт `.gitignore`):
она скачивается командой, а в репозиторий идут **числа** — строки реестра измерений.
Так же устроены физичность и докинг.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.measurements import Measurement, upsert_measurements  # noqa: E402
from kinase_ifp.alternates import AlternateGroup, read_alternates  # noqa: E402
from kinase_ifp.config import MEASUREMENTS_CSV  # noqa: E402
from kinase_ifp.pocket_content import fetch_pdb  # noqa: E402

# Куда складывается скачанная запись PDB. Каталог вне git: файл весит под мегабайт,
# воспроизводится одной командой и нужен только для пересчёта.
CACHE_DIR: Final = Path("data/rcsb-cache")

# Вид измерения в реестре.
MEASUREMENT: Final[str] = "alternates"


def измерения(
    мишень: str, pdb_id: str, группы: list[AlternateGroup], источник: str
) -> list[Measurement]:
    """Строки реестра: сводка по структуре плюс расхождение каждой группы."""
    дата = datetime.now(timezone.utc).date().isoformat()

    def строка(вариант: str, величина: str, значение: str, единица: str) -> Measurement:
        return Measurement(
            target=мишень,
            measurement=MEASUREMENT,
            variant=вариант,
            metric=величина,
            value=значение,
            unit=единица,
            measured_utc=дата,
            source=источник,
            notes="расхождение — наибольшее расстояние между одноимёнными атомами A и B",
        )

    расходятся = sum(1 for г in группы if not г.first_is_major)
    строки = [
        строка(pdb_id, "alternate_groups", str(len(группы)), "групп"),
        строка(pdb_id, "major_differs_from_a", str(расходятся), "групп"),
    ]
    for группа in группы:
        вариант = f"{pdb_id}:{группа.label}"
        строки.append(строка(вариант, "shift_ab", f"{группа.shift:.4f}", "A"))
        for метка, доля in sorted(группа.occupancy.items()):
            строки.append(строка(вариант, f"occupancy_{метка.lower()}", f"{доля:.2f}", "доля"))
    return строки


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument("--pdb-id", required=True, help="код структуры, например 6tgu")
    разбор.add_argument(
        "--target",
        type=Path,
        help="target.json пакета мишени: включает запись чисел в реестр измерений",
    )
    разбор.add_argument("--cache", type=Path, default=CACHE_DIR, help="куда класть запись PDB")
    разбор.add_argument("--measurements", type=Path, default=MEASUREMENTS_CSV)
    разбор.add_argument("--top", type=int, default=8, help="сколько групп напечатать")
    аргументы = разбор.parse_args()

    текст = fetch_pdb(аргументы.pdb_id, аргументы.cache)
    группы = read_alternates(текст)
    if not группы:
        print(f"в записи {аргументы.pdb_id} нет альтернативных положений", file=sys.stderr)
        return 1

    расходятся = [г for г in группы if not г.first_is_major]
    print(f"=== {аргументы.pdb_id}: групп с альтернативами {len(группы)} ===")
    print(f"правило «берём A» расходится с «берём мажорный» в {len(расходятся)} из {len(группы)}")
    print("\nнаибольшие расхождения:")
    for группа in группы[: аргументы.top]:
        доли = ", ".join(f"{м}={д:.2f}" for м, д in sorted(группа.occupancy.items()))
        отметка = " " if группа.first_is_major else "*"
        print(f" {отметка}{группа.label:<9} цепь {группа.chain}  {группа.shift:6.4f} A   {доли}")
    print("\n* — мажорный конформер не A, то есть наш выбор берёт минорный")

    if аргументы.target is None:
        print("\n--target не задан: числа напечатаны, но в реестр не записаны")
        return 0

    мишень = str(json.loads(аргументы.target.read_text(encoding="utf-8"))["pdb_id"])
    строки = измерения(
        мишень,
        аргументы.pdb_id,
        группы,
        f"python scripts/alternate_models.py --pdb-id {аргументы.pdb_id}",
    )
    upsert_measurements(строки, аргументы.measurements)
    print(f"\nв реестр измерений записано строк: {len(строки)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
