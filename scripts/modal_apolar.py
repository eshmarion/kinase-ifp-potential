"""Приложение Modal: расчёт по выгрузке KLIFS без GPU, кусками.

Тонкий вход. Своей научной логики здесь нет: правила и их подсчёт живут
в `kinase_ifp.apolar_rule`, сборка пакетов — в `kinase_ifp.package_batch`,
выборка структур — в `scripts/apolar_rule.py`. Здесь только раздача работы.

**Почему в облаке.** Узкое место — канал до KLIFS: около 11 секунд на структуру
из 13, и на домашнем канале 16 потоков не дали и 50 структур за десять минут
GPU не нужен вовсе.

**Почему кусками, а не одной командой.** Первая попытка была именно одной командой
и не дошла ни разу: 16.09 контейнер вытеснили дважды, а третья попытка упёрлась
в часовой предел (`FunctionTimeoutError`). Скачанные пакеты живут во временном
каталоге, поэтому каждое вытеснение выбрасывало всю работу. Двухчасовая загрузка
при вытеснении примерно раз в час не заканчивается никогда — это свойство задачи,
а не невезение.

**Почему нарезка не портит ответ.** `apolar_rule.aggregate` складывает разбивку
по структурам суммами, а `score_rules_per_structure` считает каждую структуру
независимо от остальных. Значит расчёт раскладывается на куски **точно**: собрать
разбивку по частям и сложить один раз — то же число до бита, а не приближение.
Это проверяется тестом, а не утверждается (`tests/test_modal_apolar.py`).

Запуск — из контейнера через сервис `cloud`, у которого примонтирован токен:

    docker compose run --rm cloud modal run scripts/modal_apolar.py --limit 120
    docker compose run --rm cloud modal run scripts/modal_apolar.py

Результат кладётся на диск в `data/calibration/`: числа не должны существовать
только в облаке.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import modal

# Modal увозит в облако только файл точки входа, поэтому соседние модули из `scripts/`
# там не импортируются сами: каталоги кладутся в образ ниже и добавляются в путь здесь.
# Первый путь работает локально, второй — в облаке.
PROJECT_SCRIPTS_REMOTE = "/opt/project/scripts"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, PROJECT_SCRIPTS_REMOTE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/project"

# Образ описан Dockerfile'ом, а не кодом Modal, по той же причине, что у `modal_app.py`:
# тот же файл собирается локально, поэтому «окружение не собралось» выясняется
# на своей машине. Берётся `base`, а не `diffsbdd`: модель не нужна.
image = (
    modal.Image.from_dockerfile(
        str(PROJECT_ROOT / "docker" / "Dockerfile.base"),
        context_dir=str(PROJECT_ROOT),
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path=PROJECT_SCRIPTS_REMOTE)
    # Выборка читается по пути `<корень>/data/klifs`. Сами структуры качаются из KLIFS,
    # в образ идут только таблицы.
    .add_local_dir(str(PROJECT_ROOT / "data" / "klifs"), remote_path=f"{REMOTE_ROOT}/data/klifs")
)

app = modal.App("kinase-ifp-apolar")

# Размер куска подобран под вытеснение, а не под скорость: кусок обязан успевать
# закончиться заметно раньше, чем контейнер отберут. Потери при вытеснении равны
# одному куску, и Modal перезапускает его сам (`retries`).
РАЗМЕР_КУСКА: int = 40

# Потоков загрузки внутри куска. Меньше, чем 32 из заявки, потому что кусков много
# и работают они одновременно: KLIFS видит произведение, а не это число.
ПОТОКОВ_В_КУСКЕ: int = 8

# Сколько кусков считается одновременно. Ограничение стоит ради KLIFS, а не ради
# Modal: база видит произведение числа контейнеров на число потоков, и без предела
# сорок три куска ушли бы в неё разом. Проба на трёх кусках дала ноль отказов
# при двадцати четырёх одновременных запросах; здесь взято вдвое больше с запасом.
ОДНОВРЕМЕННО_КУСКОВ: int = 10

ЗАГОЛОВОК_СВОДКИ = ["rule", "reproduced", "extra", "reference_total", "recall", "precision"]
ЗАГОЛОВОК_РАЗБИВКИ = ["label", "rule", "reproduced", "extra", "reference_total", "positions"]


def выборка_структур() -> list[int]:
    """Идентификаторы структур заявки: киназы сверки, у которых есть эталон.

    Правило отбора здесь не повторяется, а берётся у самого расчёта: копия правила
    разошлась бы с оригиналом — ровно тот случай, ради которого заведён
    тест-синхронизатор. Обе функции приватные —
    они живут в скрипте, а не в `src/`, и это долг, записанный отдельной заявкой.
    """
    from apolar_rule import _выборка, _прочитать

    структуры, эталоны = _прочитать()
    return [int(и) for и in _выборка(структуры, эталоны)["structure.klifs_id"]]


def куски(идентификаторы: list[int], размер: int) -> list[list[int]]:
    """Режет выборку на куски подряд.

    Порядок не важен: сводка складывается суммами, и от порядка слагаемых не зависит.
    """
    if размер < 1:
        raise ValueError(f"размер куска должен быть положительным, получено {размер}")
    return [идентификаторы[н : н + размер] for н in range(0, len(идентификаторы), размер)]


@app.function(
    image=image,
    cpu=4.0,
    memory=4096,
    timeout=1800,
    retries=3,
    max_containers=ОДНОВРЕМЕННО_КУСКОВ,
)
def кусок(идентификаторы: list[int], distance: float | None = None) -> dict[str, object]:
    """Считает правила на одном куске выборки и возвращает разбивку по структурам.

    Возвращает плоские записи, а не объекты: так результат не зависит от того,
    совпали ли версии кода в облаке и на машине, откуда запущено.
    """
    import tempfile

    sys.path.insert(0, f"{REMOTE_ROOT}/src")

    from apolar_rule import _прочитать

    from kinase_ifp.apolar_rule import score_rules_per_structure
    from kinase_ifp.config import APOLAR_CONTACT_DISTANCE, APOLAR_RULE_CANDIDATES
    from kinase_ifp.package_batch import build_packages

    структуры, эталоны = _прочитать()
    нужные = структуры[структуры["structure.klifs_id"].isin(идентификаторы)]
    строки = [строка for _, строка in нужные.iterrows()]

    порог = APOLAR_CONTACT_DISTANCE if distance is None else distance
    with tempfile.TemporaryDirectory() as врем:
        собранные, отказы = build_packages(строки, Path(врем), workers=ПОТОКОВ_В_КУСКЕ)
        пакеты = {путь: эталоны[klifs_id] for путь, klifs_id in собранные.items()}
        по_структурам = score_rules_per_structure(пакеты, APOLAR_RULE_CANDIDATES, порог)

    return {
        "разбивка": [
            {
                "label": и.label,
                "rule": и.rule,
                "reproduced": и.reproduced,
                "extra": и.extra,
                "reference_total": и.reference_total,
                "positions": и.positions,
            }
            for и in по_структурам
        ],
        "собрано": len(собранные),
        "отказы": отказы,
    }


def _записать(путь: Path, заголовок: list[str], строки: list[list[object]]) -> None:
    """CSV с переводом строки, а не с возвратом каретки."""
    with путь.open("w", encoding="utf-8", newline="") as файл:
        писатель = csv.writer(файл, lineterminator="\n")
        писатель.writerow(заголовок)
        писатель.writerows(строки)


@app.local_entrypoint()
def main(limit: int = 0, chunk: int = РАЗМЕР_КУСКА, distance: float | None = None) -> None:
    """Раздаёт выборку кусками, складывает разбивку один раз и кладёт числа на диск."""
    from kinase_ifp.apolar_rule import StructureRuleScore, aggregate
    from kinase_ifp.config import APOLAR_RULE_CANDIDATES

    идентификаторы = выборка_структур()
    if limit:
        идентификаторы = идентификаторы[:limit]
    части = куски(идентификаторы, chunk)
    print(f"структур в выборке: {len(идентификаторы)}, кусков: {len(части)} по {chunk}")

    разбивка: list[StructureRuleScore] = []
    отказы: list[str] = []
    собрано = 0
    for номер, итог in enumerate(кусок.map(части, kwargs={"distance": distance}), start=1):
        разбивка += [StructureRuleScore(**запись) for запись in итог["разбивка"]]
        отказы += итог["отказы"]
        собрано += итог["собрано"]
        сколько, сбоев = итог["собрано"], len(итог["отказы"])
        print(f"  кусок {номер}/{len(части)}: собрано {сколько}, отказов {сбоев}")

    if not разбивка:
        raise SystemExit("ни один кусок не дал разбивки — считать нечего")

    итоги = aggregate(разбивка, APOLAR_RULE_CANDIDATES)
    эталонных = итоги[0].reference_total if итоги else 0

    print(f"\nПакетов собрано: {собрано}; отказов: {len(отказы)}")
    print(f"Эталонных бит apolar: {эталонных}\n")
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

    куда = PROJECT_ROOT / "data" / "calibration"
    куда.mkdir(parents=True, exist_ok=True)
    имя = "apolar_rule_all.csv" if not limit else f"apolar_rule_limit{limit}.csv"

    _записать(
        куда / имя,
        ЗАГОЛОВОК_СВОДКИ,
        [
            [
                итог.rule,
                итог.reproduced,
                итог.extra,
                итог.reference_total,
                f"{итог.recall:.4f}",
                f"{итог.precision:.4f}",
            ]
            for итог in итоги
        ],
    )
    print(f"\nЧисла записаны: {куда / имя}")

    # Разбивка по структурам — то, ради чего `score_rules_per_structure` вообще есть:
    # сводное число скрывает, что решающих наблюдений у части набора элементов единицы.
    подробно = куда / имя.replace(".csv", "_per_structure.csv")
    _записать(
        подробно,
        ЗАГОЛОВОК_РАЗБИВКИ,
        [
            [и.label, и.rule, и.reproduced, и.extra, и.reference_total, и.positions]
            for и in разбивка
        ],
    )
    print(f"Разбивка по структурам: {подробно}")
