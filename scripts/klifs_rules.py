"""CLI: воспроизведение эталонных отпечатков KLIFS по правилам FingerPrintLib.

    python scripts/klifs_rules.py --best-per-kinase              # по структуре на киназу, 12 киназ
    python scripts/klifs_rules.py --best-per-kinase --sweep      # с перебором порогов
    python scripts/klifs_rules.py --all --limit 200 --workers 32 # проверка объёмом

Отвечает на один вопрос: совпадает ли расчёт по опубликованным правилам KLIFS
(Marcou & Rognan 2007, таблицы 2–4) с эталонными отпечатками базы бит в бит.
Это не наш отпечаток: наш считается ProLIF по нашим порогам и подгонке под чужую
геометрию не подлежит.

Пакеты мишеней собираются во временный каталог, `data/targets/` не меняется.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timezone  # noqa: E402

from experiments.measurements import Measurement, upsert_measurements  # noqa: E402
from kinase_ifp.config import CALIBRATION_KINASES, MEASUREMENTS_CSV  # noqa: E402
from kinase_ifp.klifs import select_best_per_kinase  # noqa: E402

# Вид измерения в реестре: расчёт по опубликованным правилам FingerPrintLib,
# а не наш отпечаток ProLIF. Разные вещи, и смешивать их в одном виде нельзя.
RULES_MEASUREMENT = "klifs_rules"

# Мишень для сводки по всей выборке: сумма по двенадцати структурам своей не имеет.
ВЫБОРКА = "e04b"

# Вариант для чисел, не разложенных по типам взаимодействий.
ВСЕ_ТИПЫ = "all"

# Величины, которые считаются в битах, а не в структурах.
ТИПОВЫЕ = frozenset({"ours", "reference", "shared"})
from kinase_ifp.klifs_rules import (  # noqa: E402
    ПРАВИЛА_KLIFS,
    FingerprintComparison,
    PreparedPackage,
    RuleThresholds,
    aggregate_types,
    compare_prepared,
    hbond_candidates,
    prepare_package,
    ring_pairs,
)
from kinase_ifp.package_batch import build_packages  # noqa: E402

КОРЕНЬ = Path(__file__).resolve().parents[1]
СТРУКТУРЫ = КОРЕНЬ / "data" / "klifs" / "structures.csv"
ОТПЕЧАТКИ = КОРЕНЬ / "data" / "klifs" / "klifs_ifp.csv"

# Сетка перебора. Порог колец идёт мелким шагом: именно он в статье напечатан
# заведомо коротким (4.0 A), а расстояния между центрами колец в кармане киназы —
# 4.5…5.5 A, то есть решается всё внутри одного ангстрема.
ПОРОГИ_КОЛЕЦ = (4.0, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 6.5)
ПОРОГИ_ИОННЫЕ = (4.0, 4.5, 5.0)
УГЛЫ_КОЛЕЦ = (20.0, 30.0, 40.0, 50.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    режим = parser.add_mutually_exclusive_group(required=True)
    режим.add_argument(
        "--best-per-kinase", action="store_true", help="по лучшей структуре на киназу"
    )
    режим.add_argument("--all", action="store_true", help="все структуры киназ выборки")
    parser.add_argument("--limit", type=int, default=0, help="взять не больше N структур")
    parser.add_argument(
        "--measurements",
        type=Path,
        nargs="?",
        const=MEASUREMENTS_CSV,
        default=None,
        help="записать числа выборки в реестр измерений (по умолчанию общий)",
    )
    parser.add_argument("--seed", type=int, default=0, help="сид подвыборки при --limit")
    parser.add_argument("--workers", type=int, default=16, help="сколько структур качать разом")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="перебрать пороги колец, ионный порог и угловую границу",
    )
    parser.add_argument(
        "--aromatic", type=float, default=None, help="порог расстояния между центрами колец"
    )
    parser.add_argument("--ionic", type=float, default=None, help="порог ионного контакта")
    parser.add_argument("--hbond", type=float, default=None, help="порог ‖DA‖ водородной связи")
    parser.add_argument(
        "--hbond-angle",
        type=float,
        default=None,
        help="допустимое отклонение от линейности D-H...A, градусы",
    )
    parser.add_argument("--csv", type=Path, default=None, help="куда записать числа по структурам")
    parser.add_argument(
        "--packages",
        type=Path,
        default=None,
        help="каталог для пакетов мишеней: уже собранные переиспользуются, "
        "новые докачиваются. Без него пакеты собираются во временный каталог",
    )
    parser.add_argument(
        "--dump-pairs",
        type=Path,
        default=None,
        help="выгрузить геометрию всех пар колец «остаток — лиганд» рядом "
        "с битами эталона: по ней подбирается правило",
    )
    parser.add_argument(
        "--dump-hbonds",
        type=Path,
        default=None,
        help="выгрузить лучшие пары «донор — акцептор» по позициям рядом "
        "с битами DON и ACC эталона",
    )
    return parser.parse_args()


def _прочитать() -> tuple[pd.DataFrame, dict[int, str]]:
    структуры = pd.read_csv(СТРУКТУРЫ)
    отпечатки = pd.read_csv(ОТПЕЧАТКИ, dtype={"structure_id": int})
    пары = zip(отпечатки["structure_id"], отпечатки["bits"], strict=True)
    return структуры, dict(пары)


def _выборка(структуры: pd.DataFrame, эталоны: dict[int, str]) -> pd.DataFrame:
    свои = структуры[структуры["kinase.klifs_name"].isin(CALIBRATION_KINASES)]
    return свои[свои["structure.klifs_id"].isin(эталоны)]


def _прогресс(готово: int, всего: int) -> None:
    if готово % 50 == 0:
        print(f"  обработано {готово} из {всего}", flush=True)


def _строки(args: argparse.Namespace, выборка: pd.DataFrame, эталоны: dict[int, str]) -> list:
    if args.best_per_kinase:
        отобранные = select_best_per_kinase(выборка, CALIBRATION_KINASES, set(эталоны))
        return [строка for _, строка in отобранные.iterrows()]
    строки = [строка for _, строка in выборка.iterrows()]
    if args.limit and args.limit < len(строки):
        # Первые N подряд — выборка, смещённая к одной киназе: таблица KLIFS
        # отсортирована по структурам. Случайная подвыборка с объявленным сидом.
        строки = random.Random(args.seed).sample(строки, args.limit)
    return строки


def _пороги(args: argparse.Namespace) -> RuleThresholds:
    пороги = ПРАВИЛА_KLIFS
    if args.aromatic is not None:
        пороги = replace(пороги, aromatic=args.aromatic)
    if args.ionic is not None:
        пороги = replace(пороги, ionic=args.ionic)
    if args.hbond is not None:
        пороги = replace(пороги, hbond=args.hbond)
    if args.hbond_angle is not None:
        пороги = replace(пороги, hbond_deviation=args.hbond_angle)
    return пороги


def _печать_по_структурам(итоги: list[FingerprintComparison]) -> None:
    print("\nПо структурам: совпадение расчёта по правилам KLIFS с эталоном\n")
    print(f"{'структура':10} {'эталон':>7} {'пропущено':>10} {'лишних':>7}  расхождения")
    for итог in sorted(итоги, key=lambda и: и.label):
        пропущено = [р for р in итог.differences if р.reference]
        лишние = [р for р in итог.differences if р.ours]
        подробно = ", ".join(
            f"{'-' if р.reference else '+'}{р.interaction}@{р.position}"
            for р in (*пропущено, *лишние)
        )
        print(
            f"{итог.label:10} {итог.reference_total:>7} {len(пропущено):>10} "
            f"{len(лишние):>7}  {подробно or 'совпадает бит в бит'}"
        )


def _печать_по_типам(итоги: list[FingerprintComparison]) -> None:
    сводка = aggregate_types(итоги)
    всего_эталон = sum(с.reference for с in сводка)
    всего_наших = sum(с.ours for с in сводка)
    всего_общих = sum(с.shared for с in сводка)
    print(f"\nПо типам взаимодействий (эталонных бит {всего_эталон})\n")
    print(f"{'тип':6} {'наших':>7} {'эталон':>7} {'общих':>7} {'пропущено':>10} {'лишних':>7}")
    for с in сводка:
        print(
            f"{с.interaction:6} {с.ours:>7} {с.reference:>7} {с.shared:>7} "
            f"{с.reference - с.shared:>10} {с.ours - с.shared:>7}"
        )
    совпало = sum(1 for и in итоги if и.identical)
    print(
        f"\nСтруктур совпало бит в бит: {совпало} из {len(итоги)}; "
        f"бит общих {всего_общих} из {всего_эталон} эталонных при {всего_наших} наших"
    )


def _перебор(готовые: list[PreparedPackage]) -> None:
    print("\nПеребор порога расстояния между центрами колец (ионный порог 4.0 A)\n")
    print(f"{'порог, A':>9} {'пропущено':>10} {'лишних':>8} {'структур совпало':>18}")
    for порог in ПОРОГИ_КОЛЕЦ:
        итоги = [compare_prepared(г, replace(ПРАВИЛА_KLIFS, aromatic=порог)) for г in готовые]
        _строка_перебора(f"{порог:>9}", итоги)

    print("\nПеребор ионного порога (порог колец 5.5 A)\n")
    print(f"{'порог, A':>9} {'пропущено':>10} {'лишних':>8} {'структур совпало':>18}")
    for порог in ПОРОГИ_ИОННЫЕ:
        пороги = replace(ПРАВИЛА_KLIFS, aromatic=5.5, ionic=порог)
        _строка_перебора(f"{порог:>9}", [compare_prepared(г, пороги) for г in готовые])

    print("\nПеребор угловой границы «лицом / ребром» (кольца 5.5 A, ионный 4.5 A)\n")
    print(f"{'угол, °':>9} {'пропущено':>10} {'лишних':>8} {'структур совпало':>18}")
    for угол in УГЛЫ_КОЛЕЦ:
        пороги = replace(ПРАВИЛА_KLIFS, aromatic=5.5, ionic=4.5, aromatic_angle=угол)
        _строка_перебора(f"{угол:>9}", [compare_prepared(г, пороги) for г in готовые])


def _строка_перебора(подпись: str, итоги: list[FingerprintComparison]) -> None:
    пропущено = sum(1 for и in итоги for р in и.differences if р.reference)
    лишних = sum(1 for и in итоги for р in и.differences if р.ours)
    совпало = sum(1 for и in итоги if и.identical)
    print(f"{подпись} {пропущено:>10} {лишних:>8} {совпало:>13} из {len(итоги)}")


def _записать_csv(путь: Path, итоги: list[FingerprintComparison]) -> None:
    путь.parent.mkdir(parents=True, exist_ok=True)
    with путь.open("w", encoding="utf-8", newline="") as f:
        писатель = csv.writer(f, lineterminator="\n")
        писатель.writerow(
            ["label", "reference_bits", "missed", "extra", "identical", "differences"]
        )
        for итог in sorted(итоги, key=lambda и: и.label):
            пропущено = sum(1 for р in итог.differences if р.reference)
            лишних = sum(1 for р in итог.differences if р.ours)
            писатель.writerow(
                [
                    итог.label,
                    итог.reference_total,
                    пропущено,
                    лишних,
                    int(итог.identical),
                    " ".join(
                        f"{'-' if р.reference else '+'}{р.interaction}@{р.position}"
                        for р in итог.differences
                    ),
                ]
            )
    print(f"\nЧисла записаны: {путь}")


def _собрать(
    args: argparse.Namespace, строки: list, каталог: Path
) -> tuple[dict[Path, int], list[str]]:
    """Пакеты мишеней: уже лежащие в каталоге берутся с диска, остальные качаются.

    Паспорт ищется в глубину: `build_target_package` кладёт файлы не прямо в
    переданный каталог, а в подкаталог с `pdb_id` внутри него.
    """
    готовые: dict[Path, int] = {}
    к_загрузке = []
    for строка in строки:
        klifs_id = int(строка["structure.klifs_id"])
        паспорта = sorted((каталог / str(klifs_id)).glob("*/target.json"))
        if паспорта:
            готовые[паспорта[0]] = klifs_id
        else:
            к_загрузке.append(строка)
    if готовые:
        print(f"Взято с диска: {len(готовые)}")
    новые, отказы = build_packages(
        к_загрузке, каталог, workers=args.workers, progress=_прогресс
    )
    return {**готовые, **новые}, отказы


def _записать_пары(путь: Path, готовые: list[PreparedPackage]) -> None:
    """Выгрузка геометрии пар колец: по ней подбирается правило KLIFS."""
    путь.parent.mkdir(parents=True, exist_ok=True)
    with путь.open("w", encoding="utf-8", newline="") as f:
        писатель = csv.writer(f, lineterminator="\n")
        писатель.writerow(
            [
                "label", "position", "residue", "centroid_distance", "plane_angle",
                "normal_to_centroid_angle", "min_atom_distance",
                "reference_face", "reference_edge",
            ]
        )
        for пакет in готовые:
            for пара in ring_pairs(пакет):
                писатель.writerow(
                    [
                        пара.label, пара.position, пара.residue,
                        f"{пара.centroid_distance:.3f}", f"{пара.plane_angle:.1f}",
                        f"{пара.normal_to_centroid_angle:.1f}",
                        f"{пара.min_atom_distance:.3f}",
                        int(пара.reference_face), int(пара.reference_edge),
                    ]
                )
    print(f"\nГеометрия пар колец записана: {путь}")


def _записать_связи(путь: Path, готовые: list[PreparedPackage]) -> None:
    """Выгрузка кандидатов в водородную связь: по ней проверяются пороги DON и ACC."""
    путь.parent.mkdir(parents=True, exist_ok=True)
    with путь.open("w", encoding="utf-8", newline="") as f:
        писатель = csv.writer(f, lineterminator="\n")
        писатель.writerow(
            ["label", "position", "residue", "direction", "distance", "deviation", "reference"]
        )
        for пакет in готовые:
            for к in hbond_candidates(пакет):
                писатель.writerow(
                    [
                        к.label, к.position, к.residue, к.direction,
                        f"{к.distance:.3f}", f"{к.deviation:.1f}", int(к.reference),
                    ]
                )
    print(f"\nКандидаты водородных связей записаны: {путь}")


def _измерения(
    итоги: list[FingerprintComparison], пороги: RuleThresholds, источник: str
) -> list[Measurement]:
    """Строки реестра по выборке: сводка, разбивка по типам и счёт по структурам.

    Зачем это здесь. Раздел 5.1 «Результатов» и «Выводы» держатся на трёх числах —
    191 эталонный бит, 190 воспроизведённых, 143 гидрофобных, — и до 19.09 у них
    не было машинного источника: сверка находила их только в прозе `docs/calibration.md`
    Числа считаются здесь, поэтому здесь же и записываются.

    Мишень строки — код структуры, а для сводки по всей выборке `ВЫБОРКА`: реестр
    ключуется четвёркой, и сумма по двенадцати структурам своей мишени не имеет.
    """
    дата = datetime.now(timezone.utc).date().isoformat()
    примечание = (
        f"правила FingerPrintLib, пороги: кольца {пороги.aromatic} A и "
        f"{пороги.aromatic_angle}°, гидрофобный {пороги.hydrophobe} A, "
        f"водородная связь {пороги.hbond} A и {пороги.hbond_deviation}°, "
        f"ионный {пороги.ionic} A"
    )

    def строка(мишень: str, вариант: str, величина: str, значение: int) -> Measurement:
        return Measurement(
            target=мишень,
            measurement=RULES_MEASUREMENT,
            variant=вариант,
            metric=величина,
            value=str(значение),
            unit="бит" if "bits" in величина or величина in ТИПОВЫЕ else "структур",
            measured_utc=дата,
            source=источник,
            notes=примечание,
        )

    сводка = aggregate_types(итоги)
    строки = [
        строка(ВЫБОРКА, ВСЕ_ТИПЫ, "reference_bits", sum(с.reference for с in сводка)),
        строка(ВЫБОРКА, ВСЕ_ТИПЫ, "shared_bits", sum(с.shared for с in сводка)),
        строка(ВЫБОРКА, ВСЕ_ТИПЫ, "our_bits", sum(с.ours for с in сводка)),
        строка(ВЫБОРКА, ВСЕ_ТИПЫ, "structures", len(итоги)),
        строка(ВЫБОРКА, ВСЕ_ТИПЫ, "identical", sum(1 for и in итоги if и.identical)),
    ]
    for счёт in сводка:
        строки += [
            строка(ВЫБОРКА, счёт.interaction, "reference", счёт.reference),
            строка(ВЫБОРКА, счёт.interaction, "ours", счёт.ours),
            строка(ВЫБОРКА, счёт.interaction, "shared", счёт.shared),
        ]
    for итог in итоги:
        общих = sum(с.shared for с in итог.by_type)
        строки += [
            строка(итог.label, ВСЕ_ТИПЫ, "reference_bits", итог.reference_total),
            строка(итог.label, ВСЕ_ТИПЫ, "shared_bits", общих),
        ]
    return строки


def main() -> None:
    args = parse_args()
    структуры, эталоны = _прочитать()
    строки = _строки(args, _выборка(структуры, эталоны), эталоны)

    with tempfile.TemporaryDirectory() as врем:
        каталог = args.packages or Path(врем)
        каталог.mkdir(parents=True, exist_ok=True)
        собранные, отказы = _собрать(args, строки, каталог)
        print(f"Пакетов всего: {len(собранные)}; отказов: {len(отказы)}")
        метки = {
            int(строка["structure.klifs_id"]): str(строка["structure.pdb_id"])
            for строка in строки
        }
        готовые = [
            prepare_package(
                путь,
                reference=эталоны[klifs_id],
                label=метки.get(klifs_id, str(klifs_id)),
            )
            for путь, klifs_id in собранные.items()
        ]
        пороги = _пороги(args)
        итоги = [compare_prepared(г, пороги) for г in готовые]
        if args.sweep:
            _перебор(готовые)
        if args.dump_pairs:
            _записать_пары(args.dump_pairs, готовые)
        if args.dump_hbonds:
            _записать_связи(args.dump_hbonds, готовые)

    _печать_по_структурам(итоги)
    _печать_по_типам(итоги)
    if args.measurements:
        строки_реестра = _измерения(
            итоги, пороги, "python scripts/klifs_rules.py --best-per-kinase"
        )
        upsert_measurements(строки_реестра, args.measurements)
        print()
        print(f"в реестр измерений записано строк: {len(строки_реестра)}")
    print(
        f"\nПороги: кольца {пороги.aromatic} A и {пороги.aromatic_angle}°, "
        f"гидрофобный {пороги.hydrophobe} A, водородная связь {пороги.hbond} A "
        f"и {пороги.hbond_deviation}°, ионный {пороги.ionic} A"
    )
    if отказы:
        print(f"\nОтказы ({len(отказы)}), первые пять:")
        for строка in отказы[:5]:
            print(" ", строка)
    if args.csv:
        _записать_csv(args.csv, итоги)


if __name__ == "__main__":
    main()
