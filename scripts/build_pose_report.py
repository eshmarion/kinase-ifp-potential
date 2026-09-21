"""CLI: бенчмарк набора поз в `results/`.

    docker compose run --rm dev python scripts/build_pose_report.py \\
        --run 2026-08-24-6tgu-s0-n100 --run 2026-09-17-6fnk-s0-n100

Кладёт в `results/` числа, которыми держится утверждение «скор различает
положение лиганда»: ранговую корреляцию скора с отклонением от кристаллической позы,
уровень её значимости, скор и ранг нативной позы вместе с числом поз, делящих
максимум, и доли поз с ключевой связью по бинам отклонения.

Прогонов можно дать несколько — тогда строки получают приставкой мишень, и числа
двух киназ стоят в одном файле рядом. Ради этого сравнения второй набор поз
и считается (пункт `C.2` плана подачи): без него «все результаты на одной
структуре» остаётся первым же возражением к работе.

Без этого файла ни одно число раздела 5.2 не имеет машинного источника: сверка
 Сверка находила их только в прозе документа и ставила «источник документ».

Логики здесь нет: вызов `experiments.pose_report`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.pose_report import PoseReportError, записать, строки  # noqa: E402
from experiments.results_io import Строка, как_текст  # noqa: E402
from experiments.runs import read_passport  # noqa: E402

КОРЕНЬ = Path(__file__).resolve().parents[1]
ПРОГОНЫ = КОРЕНЬ / "runs"
КУДА = КОРЕНЬ / "results"


def _мишень(run_dir: Path) -> str:
    """`pdb_id` мишени из паспорта прогона, а не из имени папки."""
    паспорт = read_passport(run_dir)
    return str(паспорт.get("target", {}).get("pdb_id", run_dir.name))


def _показать(путь: Path) -> str:
    """Путь относительно корня, если он внутри него, иначе как есть."""
    try:
        return str(путь.relative_to(КОРЕНЬ))
    except ValueError:
        return str(путь)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="идентификатор прогона с набором поз; можно указать несколько раз",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=ПРОГОНЫ, help="каталог с папками прогонов"
    )
    parser.add_argument("--out", type=Path, default=КУДА, help="куда класть отчёт")
    args = parser.parse_args()

    несколько = len(args.run) > 1
    собранные: list[Строка] = []
    for идентификатор in args.run:
        run_dir = args.runs_dir / идентификатор
        if not run_dir.is_dir():
            raise SystemExit(f"нет папки прогона {run_dir}")
        try:
            собранные += строки(run_dir, _мишень(run_dir) if несколько else "")
        except PoseReportError as ошибка:
            raise SystemExit(f"бенчмарк не построен: {ошибка}") from ошибка

    путь_md, путь_csv = записать(собранные, args.out)
    print("прогоны: " + ", ".join(args.run))
    for строка in собранные:
        print(f"  {строка.показатель:48} {как_текст(строка.значение)}")
    print(f"\n  {_показать(путь_md)}")
    print(f"  {_показать(путь_csv)}")


if __name__ == "__main__":
    main()
