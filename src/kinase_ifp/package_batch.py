"""Сборка пакетов мишеней пачкой: параллельная загрузка из KLIFS с повтором попыток.

Нужна измерениям, которым требуется не одна мишень, а выборка: сверка правил
(`klifs_rules`), перебор правила `apolar` (`apolar_rule`). Сам пакет собирает
`klifs.build_target_package`; здесь только организация — потоки, повторы и учёт
отказов.

**Почему потоки, а не процессы и не GPU.** Замер 16.09: из ~13 секунд на структуру
около 11 уходит на скачивание mol2 из KLIFS, а на разбор и запись — две. Узкое место
сетевое, поэтому ускорение почти линейно по числу потоков, а вычислитель задаче
не нужен вовсе.

**Почему попытка повторяется.** KLIFS обрывает часть параллельных запросов: 16.09
на двенадцати структурах так терялись то две, то одна, причём при одиночном запуске
те же структуры собирались. Потеря молча меняет знаменатель измерения — 143 эталонных
бита превращались в 122 и 136. Поэтому отказ либо переживает три попытки, либо
попадает в список отказов и печатается, а не растворяется.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

from kinase_ifp.klifs import build_target_package

ПОПЫТОК: Final[int] = 3
ПАУЗА_СЕК: Final[float] = 2.0


def build_packages(
    rows: Sequence[Any],
    destination: Path,
    *,
    workers: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[Path, int], list[str]]:
    """Собирает пакеты мишеней по строкам таблицы структур KLIFS.

    Возвращает пару: путь к `target.json` → `structure.klifs_id` и список причин
    отказа. Каждый пакет кладётся в свой подкаталог `destination`, названный по
    `klifs_id`, — `data/targets/` при этом не трогается.

    `progress` зовётся после каждой готовой структуры с парой «сделано, всего».
    """
    локальные = threading.local()

    def сессия() -> Any:
        # Своя сессия `opencadd` на поток: потокобезопасность клиента не объявлена,
        # а общая сессия при параллельных запросах давала пустые отказы.
        if not hasattr(локальные, "session"):
            from opencadd.databases.klifs import setup_remote

            локальные.session = setup_remote()
        return локальные.session

    def собрать(строка: Any) -> tuple[Path | None, int, str | None]:
        klifs_id = int(строка["structure.klifs_id"])
        подкаталог = destination / str(klifs_id)
        подкаталог.mkdir(parents=True, exist_ok=True)
        последняя: Exception | None = None
        for попытка in range(ПОПЫТОК):
            try:
                return build_target_package(сессия(), строка, подкаталог, None), klifs_id, None
            except Exception as ошибка:  # noqa: BLE001 - причина сохраняется и печатается
                последняя = ошибка
                if попытка + 1 < ПОПЫТОК:
                    time.sleep(ПАУЗА_СЕК * (попытка + 1))
        # Тип исключения печатается всегда: у сетевых сбоев KLIFS текст пустой,
        # и без типа отказ выглядел бы безымянным.
        причина = (
            f"{строка['structure.pdb_id']} (id {klifs_id}): "
            f"{type(последняя).__name__}: {последняя} (попыток: {ПОПЫТОК})"
        )
        return None, klifs_id, причина

    пакеты: dict[Path, int] = {}
    отказы: list[str] = []
    готово = 0
    with ThreadPoolExecutor(max_workers=workers) as пул:
        for путь, klifs_id, причина in пул.map(собрать, rows):
            готово += 1
            if причина is not None:
                отказы.append(причина)
            elif путь is not None:
                пакеты[путь] = klifs_id
            if progress is not None:
                progress(готово, len(rows))
    return пакеты, отказы
