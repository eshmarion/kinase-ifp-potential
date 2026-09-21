"""Внешняя оценка поз докингом: `vina_score` и `ligand_efficiency`.

    docker compose run --rm dev python scripts/score_docking.py \
        --run runs/2026-08-24-6tgu-s0-n100 --target data/targets/6tgu/target.json \
        --poses results/pose_optimization-2026-08-24-6tgu-s0-n100-all7.sdf

Считает энергию позы как есть — без поиска и без минимизации (`docs/metrics.md` 3.8).
С `--poses` сравнивает одни и те же молекулы до и после дискретного потенциала, и это
главный смысл входа: величина посчитана мерой, которая в потенциал не входит, поэтому
её изменение — независимое свидетельство, а не пересказ нашего же скора.

Молекула, которую Vina или Meeko не берут, в таблицу не попадает вовсе — ни с пустой
ячейкой, ни с нулём, — а её причина печатается. Пустая ячейка здесь означала бы
неприменимость метрики, чего в данном случае нет: метрика применима, это расчёт
не удался (`docs/metrics.md`, раздел 4).

GPU не нужен: оценка позы без поиска считается миллисекунды.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.stats import wilcoxon  # noqa: E402

from experiments.layout import MOLECULES_SDF  # noqa: E402
from experiments.measurements import (  # noqa: E402
    SOURCE_VARIANT,
    Measurement,
    upsert_measurements,
    variant_from_poses,
)
from kinase_ifp.config import MEASUREMENTS_CSV  # noqa: E402
from kinase_ifp.docking import DockingError, PoseScorer  # noqa: E402
from kinase_ifp.molecule_io import open_sdf, read_mol  # noqa: E402
from kinase_ifp.protonate import prepare_ligand  # noqa: E402

РЕЗУЛЬТАТЫ = Path("results")

# Условие кристаллического лиганда: он не принадлежит прогону, поэтому у него
# собственное значение `variant` — та же договорённость, что в `check_physics.py`.
КРИСТАЛЛ = "crystal"


def сводка_докинга(
    значения: dict[str, np.ndarray], мишень: str, вариант: str, источник: str
) -> list[Measurement]:
    """Строки реестра по одному условию: медианы, счётчики и число нефизичных поз.

    Метрика `positive_energy` — число поз, у которых энергия положительна. Она здесь
    не ради полноты: положительная энергия означает грубое столкновение с белком
    (отталкивание у Vina растёт неограниченно), и это **независимый от PoseBusters**
    признак нефизичности. Он и снимает возражение о круге для варианта с членом клэша:
    тот оптимизирует расстояние до белка, которое PoseBusters и проверяет, а в оценочную
    функцию Vina наш порог не входит вовсе.
    """
    дата = datetime.now(timezone.utc).date().isoformat()
    числа: dict[str, tuple[str, str]] = {}
    for имя, значение in значения.items():
        числа[f"{имя}_median"] = (f"{np.median(значение):.3f}", "ккал/моль")
    числа["positive_energy"] = (str(int((значения["vina_score"] > 0).sum())), "поз")
    числа["molecules"] = (str(len(значения["vina_score"])), "")
    return [
        Measurement(
            target=мишень,
            measurement="docking",
            variant=вариант,
            metric=имя,
            value=значение,
            unit=единица,
            measured_utc=дата,
            source=источник,
        )
        for имя, (значение, единица) in числа.items()
    ]


def оценить_файл(sdf: Path, оценщик: PoseScorer) -> dict[int, tuple[float, float, float]]:
    """Оценивает все позы файла; ключ — **номер молекулы в файле**, а не её имя.

    Имя тут ключом быть не может: у молекул прогона DiffSBDD свойство `_Name` пустое
    у всех сразу, и словарь по именам схлопывается в одну запись. Порядок же общий
    по построению — `optimize_poses.py` пишет позы в том порядке, в каком читал
    молекулы, и пропущенных среди них нет.
    """
    оценки: dict[int, tuple[float, float, float]] = {}
    with open_sdf(sdf) as supplier:
        for номер, mol in enumerate(supplier):
            if mol is None:
                continue
            try:
                оценка = оценщик.score(mol)
                минимум = оценщик.minimize(mol)
            except DockingError as причина:
                print(f"  молекула №{номер} не оценена: {причина}", file=sys.stderr)
                continue
            оценки[номер] = (оценка.total, оценка.ligand_efficiency, минимум.total)
    return оценки


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    разбор.add_argument("--run", type=Path, required=True, help="папка прогона генерации")
    разбор.add_argument("--target", type=Path, required=True, help="target.json пакета мишени")
    разбор.add_argument("--poses", type=Path, help="SDF оптимизированных поз для сравнения")
    разбор.add_argument(
        "--label",
        default="",
        help="метка в имени таблицы: без неё расчёт по другим позам затрёт прежний",
    )
    разбор.add_argument(
        "--measurements",
        type=Path,
        default=MEASUREMENTS_CSV,
        help="реестр измерений: туда идут медианы, счётчики и число нефизичных поз",
    )
    аргументы = разбор.parse_args()
    мишень = str(json.loads(аргументы.target.read_text(encoding="utf-8"))["pdb_id"])

    начало = time.time()
    with tempfile.TemporaryDirectory() as временный:
        оценщик = PoseScorer(аргументы.target, Path(временный))
        эталон = read_mol(аргументы.target.parent / "ligand.sdf")
        if эталон is None:
            print(f"не читается ligand.sdf пакета {аргументы.target.parent}", file=sys.stderr)
            return 1
        # Кристаллический лиганд лежит в пакете без явных водородов, а типы атомов
        # PDBQT без них не назначить; молекулы прогона приходят уже протонированными.
        оценка_эталона = оценщик.score(prepare_ligand(эталон))
        print(
            f"кристаллический лиганд: vina_score {оценка_эталона.total:.2f}, "
            f"эффективность {оценка_эталона.ligand_efficiency:.3f}"
        )

        до = оценить_файл(аргументы.run / MOLECULES_SDF, оценщик)
        после = оценить_файл(аргументы.poses, оценщик) if аргументы.poses else {}

    if not до:
        print(f"в прогоне {аргументы.run} не оценено ни одной молекулы", file=sys.stderr)
        return 1

    общие = [имя for имя in до if имя in после] if после else list(до)
    строки = []
    for имя in общие:
        строка = {
            "mol_id": f"{имя:04d}",
            "vina_score_before": round(до[имя][0], 3),
            "ligand_efficiency_before": round(до[имя][1], 4),
            "vina_min_before": round(до[имя][2], 3),
        }
        if после:
            строка |= {
                "vina_score_after": round(после[имя][0], 3),
                "ligand_efficiency_after": round(после[имя][1], 4),
                "vina_min_after": round(после[имя][2], 3),
            }
        строки.append(строка)

    РЕЗУЛЬТАТЫ.mkdir(parents=True, exist_ok=True)
    суффикс = f"-{аргументы.label}" if аргументы.label else ""
    таблица = РЕЗУЛЬТАТЫ / f"docking-{аргументы.run.name}{суффикс}.csv"
    # CSV с '\n', а не с '\r\n'.
    with таблица.open("w", encoding="utf-8", newline="\n") as поток:
        писатель = csv.DictWriter(поток, fieldnames=list(строки[0]), lineterminator="\n")
        писатель.writeheader()
        писатель.writerows(строки)

    print(f"молекул {len(строки)}, {time.time() - начало:.0f} с")
    for величина, подпись in (("vina_score", "поза как есть"), ("vina_min", "после минимизации")):
        сд = np.array([с[f"{величина}_before"] for с in строки], dtype=float)
        строка = f"{подпись:20s}: медиана до {np.median(сд):+7.2f}"
        if после:
            сп = np.array([с[f"{величина}_after"] for с in строки], dtype=float)
            улучшилось = int((сп < сд).sum())
            строка += (
                f", после {np.median(сп):+7.2f}, сдвиг {np.median(сп) - np.median(сд):+6.2f}"
                f", улучшилось {улучшилось}/{len(строки)}"
            )
            if (сп != сд).any():
                строка += f", Уилкоксон p={wilcoxon(сд, сп, zero_method='wilcox').pvalue:.1e}"
        print(строка)
    лучше = int((np.array([с["vina_min_before"] for с in строки]) < оценка_эталона.total).sum())
    print(f"молекул лучше кристаллического лиганда (по минимизированной): {лучше}/{len(строки)}")
    print(f"таблица: {таблица}")

    # Числа идут и в реестр измерений: без машинного источника сверка `check_numbers.py`
    # их не подтверждает, и написанный по ним подраздел приходится откатывать
    #. Исходные молекулы получают собственный ключ `:source`,
    # поэтому в тексте исходные, М0 и М1 берутся одной выборкой по ключу.
    команда = f"python scripts/score_docking.py --run runs/{аргументы.run.name}"
    накопленные = сводка_докинга(
        {
            "vina_score": np.array([с["vina_score_before"] for с in строки], dtype=float),
            "vina_min": np.array([с["vina_min_before"] for с in строки], dtype=float),
            "ligand_efficiency": np.array(
                [с["ligand_efficiency_before"] for с in строки], dtype=float
            ),
        },
        мишень,
        f"{аргументы.run.name}:{SOURCE_VARIANT}",
        команда,
    )
    дата = datetime.now(timezone.utc).date().isoformat()
    накопленные.append(
        Measurement(
            target=мишень,
            measurement="docking",
            variant=КРИСТАЛЛ,
            metric="vina_score_median",
            value=f"{оценка_эталона.total:.3f}",
            unit="ккал/моль",
            measured_utc=дата,
            source=команда,
            notes="кристаллический лиганд пакета мишени: калибровка шкалы",
        )
    )
    if после:
        вариант = variant_from_poses(аргументы.poses)
        накопленные += сводка_докинга(
            {
                "vina_score": np.array([с["vina_score_after"] for с in строки], dtype=float),
                "vina_min": np.array([с["vina_min_after"] for с in строки], dtype=float),
                "ligand_efficiency": np.array(
                    [с["ligand_efficiency_after"] for с in строки], dtype=float
                ),
            },
            мишень,
            вариант,
            f"{команда} --poses {аргументы.poses}",
        )
        for величина in ("vina_score", "vina_min"):
            сд = np.array([с[f"{величина}_before"] for с in строки], dtype=float)
            сп = np.array([с[f"{величина}_after"] for с in строки], dtype=float)
            p = wilcoxon(сд, сп, zero_method="wilcox").pvalue if (сп != сд).any() else float("nan")
            накопленные += [
                Measurement(
                    target=мишень,
                    measurement="docking",
                    variant=вариант,
                    metric=f"{величина}_improved",
                    value=str(int((сп < сд).sum())),
                    unit="молекул",
                    measured_utc=дата,
                    source=команда,
                ),
                Measurement(
                    target=мишень,
                    measurement="docking",
                    variant=вариант,
                    metric=f"{величина}_wilcoxon_p",
                    value=f"{p:.3e}",
                    unit="",
                    measured_utc=дата,
                    source=команда,
                    notes="парный двусторонний тест против исходных молекул",
                ),
            ]
    upsert_measurements(накопленные, аргументы.measurements)
    print(f"в реестр измерений записано строк: {len(накопленные)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
