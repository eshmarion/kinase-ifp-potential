"""Запись папки прогона: молекулы, отбраковка и паспорт.

Модуль общий для обоих производителей SDF: локального набора поз и генерации
в Modal. Две реализации одного формата разошлись бы в мелочах — в имени
свойства, в порядке колонок, в том, пишется ли пустой файл сбоев, — и расхождение
всплыло бы уже на сборке отчёта.

Внутри только стандартная библиотека: модуль обязан работать и в образе DiffSBDD,
куда pandas не ставится.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from rdkit import Chem

from experiments.layout import CSV_EOL, FAILURES_CSV, MOLECULES_SDF, RUN_JSON
from kinase_ifp.molecule_io import open_sdf

FAILURES_HEADER: Final[tuple[str, str, str]] = ("mol_id", "stage", "reason")

PLATFORM_LOCAL_CPU: Final[str] = "local-cpu"
PLATFORM_MODAL_GPU: Final[str] = "modal-gpu"


class RunExistsError(FileExistsError):
    """Папка прогона уже существует."""


class MoleculesReadError(RuntimeError):
    """`molecules.sdf` прочитать не удалось или в нём нет обязательного свойства."""


def read_molecule_props(run_dir: Path, names: Iterable[str]) -> dict[str, dict[str, str]]:
    """SD-свойства молекул прогона по `mol_id`: `{mol_id: {имя: значение}}`.

    Читатель того же формата папки прогона, который пишет `write_molecules`, поэтому живёт
    рядом с писателем: имя свойства, объявленное в одном модуле, а прочитанное
    в другом, расходится при первом же переименовании.

    Молекулы читаются без санитизации: нужны только свойства, и разбор валентностей
    был бы лишней работой и лишним источником отказов. Отсутствующее свойство
    в словарь молекулы не попадает — это не ошибка: `rmsd_to_ref` пуст у генерации,
    `pose_kind` отсутствует у прогонов, записанных до его появления, и «нет сведений»
    обязано отличаться от нуля.

    Санитизация отключена ещё и потому, что набор поз содержит намеренно испорченные
    положения: разбор химии отверг бы ровно те записи, ради которых набор и сделан.
    Запись без `mol_id` роняет чтение, а не пропускается молча: привязать свойство
    не к чему, а тихий пропуск дал бы выборку меньше заявленной.
    """
    путь = run_dir / MOLECULES_SDF
    if not путь.is_file():
        # Причин ровно две, и они требуют разных действий, поэтому названы обе:
        # выгрузка оборвалась — повторить `pull_run.py`; молекулы намеренно оставлены
        # у автора — взять числа из `ranking.csv` и `metrics_per_molecule.csv`
        # либо попросить автора выложить файл. Колонка `molecules` в реестре говорит, что
        # именно из двух.
        raise MoleculesReadError(
            f"Нет файла {путь} (формат папки прогона): выгрузка прогона неполна либо молекулы "
            "оставлены у автора — смотри колонку molecules в runs/index.csv"
        )

    запрошенные = tuple(names)
    значения: dict[str, dict[str, str]] = {}
    # Поток, а не путь строкой: путь RDKit передаёт в C++ через ANSI и на Windows
    # не открывает ничего с кириллицей — обёртка в `kinase_ifp.molecule_io`.
    with open_sdf(путь, sanitize=False) as поставщик:
        for номер, молекула in enumerate(поставщик, start=1):
            if молекула is None:
                raise MoleculesReadError(
                    f"{путь}: запись №{номер} не читается даже без санитизации — файл повреждён"
                )
            if not молекула.HasProp("mol_id"):
                raise MoleculesReadError(
                    f"{путь}: у записи №{номер} нет свойства mol_id (формат папки прогона)"
                )
            свойства = {
                имя: молекула.GetProp(имя).strip()
                for имя in запрошенные
                if молекула.HasProp(имя) and молекула.GetProp(имя).strip()
            }
            значения[молекула.GetProp("mol_id")] = свойства
    return значения


def read_rmsd(run_dir: Path) -> dict[str, float]:
    """`rmsd_to_ref` по `mol_id` из `molecules.sdf`.

    В `metrics_per_molecule.csv` этой колонки нет — её заголовок закрыт, а RMSD
    живёт свойством SDF. Пустое свойство означает `source=diffsbdd`, где эталонной
    позы нет вовсе; такая молекула просто не попадает в словарь.
    """
    return {
        mol_id: float(свойства["rmsd_to_ref"])
        for mol_id, свойства in read_molecule_props(run_dir, ("rmsd_to_ref",)).items()
        if "rmsd_to_ref" in свойства
    }


@dataclass(frozen=True)
class MoleculeRecord:
    """Строка `molecules.sdf`: молекула, её `mol_id`, RMSD к эталону и прочие свойства.

    `rmsd_to_ref` равный `None` означает «эталонной позы нет» (`source=diffsbdd`)
    и записывается пустым, а не пропускается.

    `props` — дополнительные SD-свойства сверх обязательных по формату папки прогона. Контракт
    их не запрещает, а производителю поз они нужны: без `pose_kind` потребитель
 не отличит неверное размещение от неверных торсий. Отдельный объект,
    а не кортеж из четырёх элементов: параллельный список свойств рано или поздно
    разъедется с молекулами.
    """

    mol: Chem.Mol
    mol_id: str
    rmsd_to_ref: float | None = None
    props: Mapping[str, str] = field(default_factory=dict)


def make_run_id(
    pdb_id: str, n: int, guidance_scale: float, day: str, suffix: str | None = None
) -> str:
    """Собирает `run_id` вида `2026-08-23-6tgu-s0-n100`.

    Сила guidance пишется без дробной части, когда она целая: на этот вид опирается
    реестр прогонов и заголовки колонок отчёта.

    `suffix` дописывается через дефис и различает прогоны, которые правило именования
    иначе назвало бы одинаково: сида в паспорте нет, а десять наборов положений
    за один день нужны, чтобы top-1 была долей, а не единственным испытанием.
    Тем же приёмом пользуется сборка отчёта — в её фикстурах лежит `...-n5-top2`.
    """
    сила = f"{guidance_scale:g}"
    run_id = f"{day}-{pdb_id}-s{сила}-n{n}"
    return f"{run_id}-{suffix}" if suffix else run_id


def create_run_dir(run_dir: Path) -> Path:
    """Создаёт папку прогона и возвращает её путь.

    Поднимает `RunExistsError`, если в папке уже что-то лежит: перезаписывать
    результаты запрещено — новый запуск получает новую подпапку.

    **Пустая папка результатом не является и отказа не вызывает.** Её оставляет
    вытеснение контейнера в Modal: «Container terminated due to preemption. Your Function
    will be restarted with the same input» — платформа перезапускает функцию с тем же
    входом, а каталог от первой попытки остаётся. 16.09 прогон `pocket-ids` на этом
    и упал: защита от перезаписи сработала против штатного для облака перезапуска,
    и GPU-минуты ушли в отказ. Запрет №2 защищает молекулы и историю прогона,
    а не само имя каталога.
    """
    if run_dir.exists():
        if any(run_dir.iterdir()):
            raise RunExistsError(
                f"папка прогона {run_dir} уже существует и не пуста; перезапись "
                "результатов запрещена, новый запуск — новая подпапка"
            )
        return run_dir
    run_dir.mkdir(parents=True)
    return run_dir


def write_molecules(run_dir: Path, molecules: Iterable[MoleculeRecord]) -> int:
    """Пишет `molecules.sdf` и возвращает число записанных молекул.

    Принимает записи `MoleculeRecord`. Обязательные свойства — `mol_id`
    и `rmsd_to_ref`; последний при `None` записывается пустым, потому что у
    сгенерированной молекулы эталонной позы нет, а свойство обязано присутствовать.
    Всё из `props` дописывается сверх них.
    """
    written = 0
    # SDWriter получает открытый поток, а не строку пути: на Windows он передаёт путь
    # в C++ через ANSI и падает с «Bad output file» на любом пути с кириллицей.
    # Каталоги прогонов латинские, но домашний каталог пользователя может быть любым.
    with (run_dir / MOLECULES_SDF).open("w", encoding="utf-8", newline="") as stream:
        writer = Chem.SDWriter(stream)
        for molecule in molecules:
            record = Chem.Mol(molecule.mol)
            record.SetProp("mol_id", molecule.mol_id)
            record.SetProp(
                "rmsd_to_ref",
                "" if molecule.rmsd_to_ref is None else str(molecule.rmsd_to_ref),
            )
            for имя, значение in molecule.props.items():
                record.SetProp(имя, значение)
            writer.write(record)
            written += 1
        writer.close()
    return written


def write_failures(run_dir: Path, failures: Iterable[tuple[str, str, str]]) -> None:
    """Пишет `failures.csv` с колонками `mol_id, stage, reason`.

    Файл создаётся всегда, даже когда сбоев не было: иначе «файла нет» неотличимо
    от «всё записалось», и потерянные молекулы исчезают молча.
    """
    with (run_dir / FAILURES_CSV).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator=CSV_EOL)
        writer.writerow(FAILURES_HEADER)
        writer.writerows(failures)


def append_failures(run_dir: Path, failures: Iterable[tuple[str, str, str]]) -> None:
    """Дописывает строки в `failures.csv`, не трогая уже записанные.

    Нужна этапам после производства молекул: скоринг отбраковывает своё,
    и перезапись файла стёрла бы отказы производителя — то есть потеряла бы ровно те
    сведения, ради которых файл заведён. Заголовок пишется, только если файла ещё нет.
    """
    путь = run_dir / FAILURES_CSV
    существует = путь.is_file()
    with путь.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator=CSV_EOL)
        if not существует:
            writer.writerow(FAILURES_HEADER)
        writer.writerows(failures)


def current_project_commit(repo: Path | None = None, *, mark_dirty: bool = True) -> str:
    """Текущий коммит проекта для паспорта прогона.

    К хэшу дописывается `-dirty`, если в дереве есть незакоммиченные правки: без этой
    пометки паспорт утверждал бы, что результат получен кодом из названного коммита,
    хотя запускался изменённый файл, и повторить прогон по `run.json` было бы нельзя
    (`docs/questions.md`, В-14).

    `mark_dirty=False` отключает пометку и возвращает голый хэш. Так спрашивают про чужой
    клон: его мы не правим, а `git status` внутри контейнера всё равно счёл бы
    его изменённым целиком — у чужого репозитория нет нашего `.gitattributes`, и файлы,
    выкачанные на Windows с CRLF, расходятся с индексом при взгляде из Linux.

    `repo` задаёт каталог репозитория; по умолчанию берётся текущий. Возвращает
    `unknown`, если git недоступен: в образе Modal репозитория может не быть,
    а прогон из-за отсутствия истории падать не должен.
    """
    команда = ["git"] + (["-C", str(repo)] if repo is not None else [])
    try:
        хэш = subprocess.run(
            [*команда, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        правки = ""
        if mark_dirty:
            правки = subprocess.run(
                [*команда, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not хэш:
        return "unknown"
    return f"{хэш}-dirty" if правки else хэш


def write_run_json(
    run_dir: Path,
    *,
    run_id: str,
    target: dict[str, Any],
    source: str,
    sampling: dict[str, Any],
    duration_sec: int,
    platform: str = PLATFORM_LOCAL_CPU,
    model: dict[str, Any] | None = None,
    project_commit: str | None = None,
) -> Path:
    """Пишет паспорт прогона `run.json` и возвращает путь к нему.

    Прогон без паспорта считается несуществующим и в таблицы не идёт, поэтому поля
    паспорта заполняются целиком, а не по мере наличия. `model` пустой у набора положений:
    модели в таком прогоне нет, и это не то же самое, что неизвестная модель.
    """
    passport = {
        "run_id": run_id,
        # timezone.utc, а не datetime.UTC: проект живёт на Python 3.10,
        # где псевдонима UTC ещё нет.
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": target,
        "source": source,
        "model": model if model is not None else {"repo_commit": None, "checkpoint": None},
        "sampling": sampling,
        "platform": platform,
        "duration_sec": duration_sec,
        "project_commit": (
            project_commit if project_commit is not None else current_project_commit()
        ),
    }
    path = run_dir / RUN_JSON
    path.write_text(json.dumps(passport, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
