"""CLI: реконструкция правила `apolar` KLIFS по эталонным отпечаткам.

    python scripts/apolar_rule.py --models          # сравнение альтернативных моделей, без сети
    python scripts/apolar_rule.py --best-per-kinase # по структуре на киназу
    python scripts/apolar_rule.py --all --limit 200 # все структуры киназ выборки

Режим `--models` отвечает на отдельный вопрос: считает KLIFS эталонный отпечаток
на альтернативную модель или на структуру целиком. Он работает по локальной выгрузке
и структуры не качает.

Режимы `--best-per-kinase` и `--all` собирают пакеты мишеней во временный каталог:
`data/targets/` не меняется, в репозиторий попадают только числа.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kinase_ifp.apolar_rule import (  # noqa: E402
    ApolarRuleError,
    aggregate,
    compare_alternate_models,
    score_rules_per_structure,
)
from kinase_ifp.config import (  # noqa: E402
    APOLAR_CONTACT_DISTANCE,
    APOLAR_RULE_CANDIDATES,
    CALIBRATION_KINASES,
)
from kinase_ifp.klifs import rank_structures  # noqa: E402
from kinase_ifp.package_batch import build_packages  # noqa: E402

КОРЕНЬ = Path(__file__).resolve().parents[1]
СТРУКТУРЫ = КОРЕНЬ / "data" / "klifs" / "structures.csv"
ОТПЕЧАТКИ = КОРЕНЬ / "data" / "klifs" / "klifs_ifp.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    режим = parser.add_mutually_exclusive_group(required=True)
    режим.add_argument("--models", action="store_true", help="сравнить альтернативные модели")
    режим.add_argument(
        "--best-per-kinase", action="store_true", help="по лучшей структуре на киназу"
    )
    режим.add_argument("--all", action="store_true", help="все структуры киназ выборки")
    parser.add_argument("--limit", type=int, default=0, help="взять не больше N структур")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="сид случайной подвыборки при --limit; выборка воспроизводима",
    )
    parser.add_argument(
        "--distance", type=float, default=APOLAR_CONTACT_DISTANCE, help="порог контакта"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="сколько структур качать одновременно; узкое место — сеть KLIFS, "
        "а не расчёт (замер 16.09: из 13 с на структуру 11 с — скачивание mol2)",
    )
    parser.add_argument(
        "--per-structure",
        action="store_true",
        help="печатать разбивку по структурам: сколько бит apolar у эталона "
        "и сколько из них даёт каждое правило",
    )
    parser.add_argument("--csv", type=Path, default=None, help="куда записать числа")
    return parser.parse_args()


def _выборка(структуры: pd.DataFrame, эталоны: dict[int, str]) -> pd.DataFrame:
    свои = структуры[структуры["kinase.klifs_name"].isin(CALIBRATION_KINASES)]
    return свои[свои["structure.klifs_id"].isin(эталоны)]


def _прочитать() -> tuple[pd.DataFrame, dict[int, str]]:
    структуры = pd.read_csv(СТРУКТУРЫ)
    отпечатки = pd.read_csv(ОТПЕЧАТКИ, dtype={"structure_id": int})
    пары = zip(отпечатки["structure_id"], отпечатки["bits"], strict=True)
    return структуры, dict(пары)


def _режим_моделей() -> None:
    структуры, эталоны = _прочитать()
    выборка = _выборка(структуры, эталоны)
    записи = [
        (
            int(строка["structure.klifs_id"]),
            str(строка["structure.pdb_id"]),
            str(строка["structure.chain"]),
            str(строка["structure.alternate_model"]),
        )
        for _, строка in выборка.iterrows()
    ]
    итог = compare_alternate_models(эталоны, записи)
    print(f"Киназ в выборке: {len(CALIBRATION_KINASES)}; структур с эталоном: {len(записи)}")
    for ключ, значение in итог.items():
        print(f"  {ключ} ... {значение}")
    пар = итог["пар с несколькими моделями"]
    if not пар:
        return
    if итог["отпечатки разошлись"] == 0:
        print(
            "\nЭталон KLIFS считается на структуру, а не на альтернативную модель:\n"
            "все пары моделей дали побитово одинаковые отпечатки."
        )
        return
    доля = итог["отпечатки разошлись"] / пар
    print(
        f"\nЭталон KLIFS считается на альтернативную модель, а не на структуру:\n"
        f"у {итог['отпечатки разошлись']} пар из {пар} ({доля:.1%}) отпечатки моделей\n"
        "различаются. Совпадение у остальных означает лишь, что альтернатива\n"
        "не затронула карман: по одной-двум структурам об этом судить нельзя."
    )


def _прогресс(готово: int, всего: int) -> None:
    """Отметка каждые пятьдесят структур: длинный прогон иначе молчит до конца."""
    if готово % 50 == 0:
        print(f"  обработано {готово} из {всего}", flush=True)


def _печать_по_структурам(по_структурам: list) -> None:
    """Таблица «структура — эталонных бит — сколько дало каждое правило»."""
    структуры = sorted({и.label for и in по_структурам})
    правила = list(APOLAR_RULE_CANDIDATES)
    показать = [п for п in правила if п != "любой тяжёлый атом"]
    print("\nПо структурам: эталонных бит apolar и сколько из них даёт правило\n")
    шапка = f"{'структура':10} {'позиций':>8} {'эталон':>7}"
    for правило in показать:
        шапка += f" {правило[:14]:>15}"
    print(шапка)
    for метка in структуры:
        свои = {и.rule: и for и in по_структурам if и.label == метка}
        первый = next(iter(свои.values()))
        строка = f"{метка:10} {первый.positions:>8} {первый.reference_total:>7}"
        for правило in показать:
            итог = свои[правило]
            лишние = f"+{итог.extra}" if итог.extra else ""
            строка += f" {f'{итог.reproduced}{лишние}':>15}"
        print(строка)


def _собрать_и_посчитать(args: argparse.Namespace) -> None:
    структуры, эталоны = _прочитать()
    выборка = _выборка(структуры, эталоны)
    if args.best_per_kinase:
        строки = [
            rank_structures(выборка[выборка["kinase.klifs_name"] == киназа]).iloc[0]
            for киназа in CALIBRATION_KINASES
            if not выборка[выборка["kinase.klifs_name"] == киназа].empty
        ]
    else:
        строки = [строка for _, строка in выборка.iterrows()]
    if args.limit and args.limit < len(строки):
        # Первые N подряд — это выборка, смещённая к одной киназе: таблица KLIFS
        # отсортирована по структурам, а не перемешана. Берётся случайная
        # подвыборка с объявленным сидом, чтобы прогон был воспроизводим.
        сид = random.Random(args.seed)
        строки = сид.sample(строки, args.limit)

    отказы: list[str] = []
    with tempfile.TemporaryDirectory() as врем:
        собранные, отказы = build_packages(
            строки, Path(врем), workers=args.workers, progress=_прогресс
        )
        пакеты = {путь: эталоны[klifs_id] for путь, klifs_id in собранные.items()}
        print(f"Пакетов собрано: {len(пакеты)}; отказов: {len(отказы)}")
        try:
            по_структурам = score_rules_per_structure(
                пакеты, APOLAR_RULE_CANDIDATES, args.distance
            )
        except ApolarRuleError as ошибка:
            raise SystemExit(f"перебор правил не выполнен: {ошибка}") from ошибка
        итоги = aggregate(по_структурам, APOLAR_RULE_CANDIDATES)
    if args.per_structure:
        _печать_по_структурам(по_структурам)

    эталонных = итоги[0].reference_total if итоги else 0
    print(f"\nЭталонных бит apolar: {эталонных}; порог {args.distance} A\n")
    print(f"{'правило':32} {'воспр.':>8} {'лишних':>8} {'полнота':>9} {'точность':>9}")
    for итог in итоги:
        print(
            f"{итог.rule:32} {итог.reproduced:>8} {итог.extra:>8} "
            f"{итог.recall:>9.3f} {итог.precision:>9.3f}"
        )
    if отказы:
        print(f"\nОтказы ({len(отказы)}), первые пять:")
        for строка in отказы[:5]:
            print(" ", строка)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            писатель = csv.writer(f, lineterminator="\n")
            писатель.writerow(
                ["rule", "reproduced", "extra", "reference_total", "recall", "precision"]
            )
            for итог in итоги:
                писатель.writerow(
                    [
                        итог.rule,
                        итог.reproduced,
                        итог.extra,
                        итог.reference_total,
                        f"{итог.recall:.4f}",
                        f"{итог.precision:.4f}",
                    ]
                )
        print(f"\nЧисла записаны: {args.csv}")


def main() -> None:
    args = parse_args()
    if args.models:
        _режим_моделей()
        return
    _собрать_и_посчитать(args)


if __name__ == "__main__":
    main()
