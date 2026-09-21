"""CLI: измерения по базе KLIFS в `results/`.

    docker compose run --rm dev python scripts/build_klifs_base_report.py

Считает то же, что печатает `scripts/measure_klifs_base.py`, но кладёт числа в файлы:
`klifs_base.md` для чтения и `klifs_base.csv` для сверки. Без файла число из этого
измерения нельзя привести в курсовой — `scripts/check_numbers.py` берёт источники
закрытым списком и всё остальное считает неподтверждённым.

Логики здесь нет: вызов `kinase_ifp.base_stats` и запись через `experiments.base_report`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.base_report import записать, строки  # noqa: E402
from kinase_ifp.base_stats import (  # noqa: E402
    BaseStatsError,
    dataset_composition,
    reference_bits,
)

КОРЕНЬ = Path(__file__).resolve().parents[1]
КУДА = КОРЕНЬ / "results"


def main() -> None:
    try:
        собранные = строки(dataset_composition(), reference_bits())
    except BaseStatsError as ошибка:
        raise SystemExit(f"измерение не выполнено: {ошибка}") from ошибка

    путь_md, путь_csv = записать(собранные, КУДА)
    print(f"показателей записано: {len(собранные)}")
    print(f"  {путь_md.relative_to(КОРЕНЬ)}")
    print(f"  {путь_csv.relative_to(КОРЕНЬ)}")


if __name__ == "__main__":
    main()
