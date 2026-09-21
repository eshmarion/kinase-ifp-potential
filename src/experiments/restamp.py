"""Сверка прогона с пакетом мишени и штамп задним числом.

Штамп пакета (`target.protonation` и `target.package_sha256`) появился 24.08 вечером,
и прогоны старше него `compare_target_stamp` отмечает как `unstamped`: расчёт их
принимает, но сказать, на каком пакете они посчитаны, нельзя.

`runs.stamp_target` штамп ставит, но ничего не доказывает — основание требуется
внешнее: пересчёт прогона на этом пакете даёт те же числа. 25.08 такую сверку делали
руками для трёх прогонов, и уже 26.08 выяснилось, чего стоит процедура,
записанная только в тексте: оставался незарегистрированный прогон, штамповать
который он не стал именно потому, что сверку делать было некогда.
Здесь та же процедура выполняется кодом, и штамп ставится только после неё.

Модуль отделён от `runs.py` намеренно: тот обходится стандартной библиотекой
и `hashlib`, а сверка тянет за собой `metrics` и `ranking` со всем RDKit.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from experiments.layout import METRICS_CSV, RANKING_CSV
from experiments.metrics import MetricsError, compute_run_metrics
from experiments.ranking import score_run
from experiments.runs import RunError, compare_target_stamp, read_passport, stamp_target
from kinase_ifp.klifs import KlifsDataError
from kinase_ifp.scoring import ScoringError

# Файлы, по которым сверяется прогон. Оба производные: считаются из `molecules.sdf`
# и пакета мишени, поэтому совпадение любого из них означает, что пакет тот же.
VERIFIED_FILES: Final[tuple[str, ...]] = (METRICS_CSV, RANKING_CSV)

# Сколько символов строки показывать вокруг места расхождения.
WINDOW: Final[int] = 60


@dataclass(frozen=True)
class StampReport:
    """Что сверялось и чем кончилось.

    `verified` — имена файлов, которые удалось сравнить; пустым не бывает, иначе
    сверка не состоялась и до отчёта дело не доходит.
    """

    run_id: str
    verified: tuple[str, ...]
    stamped: bool
    protonation: str
    package_sha256: str

    def describe(self) -> str:
        """Строка для вывода в терминал."""
        файлы = ", ".join(self.verified)
        действие = "штамп проставлен" if self.stamped else "штамп уже стоял"
        return (
            f"сверка по {файлы}: совпало побайтово; {действие} — "
            f"режим {self.protonation!r}, хэш {self.package_sha256[:16]}"
        )


def verify_and_stamp(run_dir: Path, target_json: Path) -> StampReport:
    """Пересчитывает прогон в копии, сверяет результат и штампует при совпадении.

    Оригинал не изменяется до самого конца: пересчёт идёт во временном каталоге,
    и в папку прогона запись происходит только штампом, только после совпадения.

    Поднимает `RunError`, если сверять нечего, если пересчёт разошёлся с тем, что
    лежит, или если прогон несёт штамп другого пакета.
    """
    доступные = tuple(имя for имя in VERIFIED_FILES if (run_dir / имя).is_file())
    if not доступные:
        raise RunError(
            f"Прогон {run_dir.name} нечем сверить: нет ни {METRICS_CSV}, ни {RANKING_CSV}. "
            "Штамп без сверки — это запись в паспорт непроверенного утверждения; "
            "посчитайте прогон обычной цепочкой, и штамп встанет сам"
        )

    # Префикс каталога латинский, хотя весь модуль написан по-русски: внутрь этой
    # копии уходит `Chem.SDMolSupplier(str(...))`, а он передаёт путь в C++ через ANSI
    # и на Windows не открывает ничего с кириллицей — сверка падала бы с «Bad input
    # file» и обвиняла мишень. То же правило, что в `run_io.write_molecules`.
    with tempfile.TemporaryDirectory(prefix="sverka-") as временный:
        копия = Path(временный) / run_dir.name
        shutil.copytree(run_dir, копия)
        try:
            _пересчитать(копия, target_json, доступные)
        except (KlifsDataError, MetricsError, ScoringError) as ошибка:
            # Пересчёт может не состояться вовсе: пакет неполон, не проходит проверку
            # целостности или штамп прогона указывает на другой пакет. Всё это значит
            # «сверить не удалось», а не «сверка не сошлась», и разница важна —
            # во втором случае числа сравнивать было не с чем. Сообщение при этом
            # должно оставаться сообщением, а не трейсбеком: команду зовёт человек.
            raise RunError(
                f"Прогон {run_dir.name} не пересчитывается на пакете {target_json}: "
                f"{ошибка}"
            ) from ошибка
        for имя in доступные:
            _сравнить(run_dir / имя, копия / имя)

    паспорт = read_passport(run_dir)
    было, свежий = compare_target_stamp(паспорт, target_json)
    stamp_target(run_dir, target_json)
    return StampReport(
        run_id=run_dir.name,
        verified=доступные,
        stamped=было != "match",
        protonation=str(свежий["protonation"]),
        package_sha256=str(свежий["package_sha256"]),
    )


def _пересчитать(копия: Path, target_json: Path, файлы: tuple[str, ...]) -> None:
    """Считает в копии ровно то, что в прогоне уже есть.

    Считать лишнее нельзя: у прогона без метрик их отсутствие — это его состояние,
    а не пробел, и появившийся при сверке файл сравнивать было бы не с чем.
    """
    if METRICS_CSV in файлы:
        compute_run_metrics(копия, target_json)
    if RANKING_CSV in файлы:
        # Доля топа берётся из паспорта: она записана туда прошлым переранжированием
        #, и пересчёт с другой долей изменил бы колонку `selected`.
        доля = read_passport(копия).get("ranking", {}).get("top_fraction")
        if доля is None:
            score_run(копия, target_json)
        else:
            score_run(копия, target_json, top_fraction=float(доля))


def _сравнить(было: Path, стало: Path) -> None:
    """Побайтовое сравнение; при расхождении называет первую несовпавшую строку.

    Сравнение точное, а не «с точностью до формата»: с 25.08 наши CSV пишутся
    с `\\n` (`CSV_EOL` в `layout`), поэтому пересчёт идемпотентен побайтово,
    и любое различие — это различие в числах.
    """
    if было.read_bytes() == стало.read_bytes():
        return

    строки_было = было.read_text(encoding="utf-8").splitlines()
    строки_стало = стало.read_text(encoding="utf-8").splitlines()
    for номер, (левая, правая) in enumerate(zip(строки_было, строки_стало, strict=False), 1):
        if левая != правая:
            столбец = _первое_различие(левая, правая)
            raise RunError(
                f"Пересчёт разошёлся с {было.name}, строка {номер}, столбец {столбец}:\n"
                f"  в прогоне: {_окно(левая, столбец)}\n"
                f"  пересчёт:  {_окно(правая, столбец)}\n"
                "Значит прогон посчитан на другом пакете мишени — штамп ставить нельзя"
            )
    raise RunError(
        f"Пересчёт разошёлся с {было.name} длиной: было строк {len(строки_было)}, "
        f"стало {len(строки_стало)}. Штамп ставить нельзя"
    )


def _первое_различие(левая: str, правая: str) -> int:
    """Номер первого несовпавшего символа, считая с единицы."""
    for индекс, (а, б) in enumerate(zip(левая, правая, strict=False), 1):
        if а != б:
            return индекс
    return min(len(левая), len(правая)) + 1


def _окно(строка: str, столбец: int) -> str:
    """Фрагмент строки вокруг места расхождения.

    Показывать начало строки бесполезно: в `metrics_per_molecule.csv` первые полторы
    сотни символов — это `run_id`, `mol_id` и SMILES, одинаковые у обеих сторон,
    и сообщение выглядело бы как «строки совпадают, но разошлись».
    """
    начало = max(0, столбец - 1 - WINDOW // 2)
    конец = начало + WINDOW
    кусок = строка[начало:конец]
    return ("…" if начало else "") + кусок + ("…" if конец < len(строка) else "")
