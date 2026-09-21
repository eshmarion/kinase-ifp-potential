"""Манифест выгрузки KLIFS: чем получен каждый файл, когда и в каком состоянии.

Заведён по итогам методологического разбора
(`docs/audit-findings.md`). До него в `data/klifs/` лежали семь файлов и ни одного
описания: дата существовала только как время изменения файла на диске, а git его
не хранит — в свежем клоне даты выгрузки не было вообще.

**Дата обращения к базе записывается верхней границей, а не точным моментом.**
Точного момента не знает никто: он нигде не сохранялся. Известно проверяемое — дата
коммита, последним изменившего файл: позже неё к базе не обращались. Поле названо
`retrieved_not_later_than` именно поэтому. Правдоподобная дата была бы хуже пустого
места: проверить её нечем, а выглядит она как факт.

**Версию базы KLIFS манифест не несёт.** Не по недосмотру: `setup_remote()` отдаёт
объект с таблицами и ничем больше — ни версии, ни даты среза (проверено 14.09.2026
перечислением атрибутов сессии). Вместо версии базы фиксируются версия клиента
`opencadd` и хеши файлов: по ним видно, та ли это выгрузка, даже когда база уже
отвечала бы иначе.

Происхождение каждого файла объявлено здесь, а не выводится из имени: три файла
каталога — не выгрузка, а результат нашего расчёта поверх исходных файлов RCSB
, и смешивать их с прямой выгрузкой значит приписывать KLIFS то,
чего он не говорил.

**Поле `expected_command` — ожидание, а не протокол.** Строки объявлены константами
и одинаковы при любом прогоне; с какими ключами команду звали на самом деле, манифест
не знает. Поле названо так намеренно: прежнее имя `command` читалось как
запись факта, и в одной строке эта мнимая запись была прямо неверна. Записывать
фактический вызов (`sys.argv`) в момент получения файла — отдельная работа, она
осмысленна только для будущих выгрузок.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from kinase_ifp.config import DATA_DIR, PROJECT_ROOT

KLIFS_DIR: Final[Path] = DATA_DIR / "klifs"
MANIFEST_NAME: Final[str] = "manifest.json"
MANIFEST_SCHEMA: Final[int] = 1

# Прямая выгрузка из KLIFS.
ORIGIN_DOWNLOAD: Final[str] = "klifs-download"
# Наш расчёт поверх исходных файлов RCSB: KLIFS отдаёт белок очищенным, и по пакету
# нельзя спросить, что ещё занимало карман в кристалле.
ORIGIN_DERIVED: Final[str] = "derived"


class ManifestError(RuntimeError):
    """Манифест нельзя собрать или он разошёлся с файлами каталога."""


@dataclass(frozen=True)
class Origin:
    """Откуда взялся файл каталога: вид источника и ожидаемая команда получения."""

    kind: str
    expected_command: str


# Происхождение файлов `data/klifs/`. Команды взяты из докстрингов скриптов и раздела
# Записаны по факту, а не восстановлены по памяти.
#
# `python scripts/fetch_klifs.py` зовётся **без** `--limit`: ключ объявлен
# с `default=None`, а `fetch_structures` проверяет `if limit is not None`, поэтому
# `--limit 0` означает не «все», а `df.head(0)` — пустую таблицу. У `fetch_klifs_ifp.py`
# наоборот: там `NO_LIMIT = 0`, и ноль означает именно «все». Одинаковый на вид ключ
# у двух скриптов работает по-разному; ошибка в этой строке прожила сутки.
FILE_ORIGINS: Final[dict[str, Origin]] = {
    "structures_all.csv": Origin(ORIGIN_DOWNLOAD, "python scripts/fetch_klifs.py"),
    "structures.csv": Origin(ORIGIN_DOWNLOAD, "python scripts/fetch_klifs.py"),
    "kinases.csv": Origin(ORIGIN_DOWNLOAD, "python scripts/fetch_klifs.py"),
    "klifs_ifp.csv": Origin(ORIGIN_DOWNLOAD, "python scripts/fetch_klifs_ifp.py --limit 0"),
    "pocket_content.csv": Origin(
        ORIGIN_DERIVED,
        "python scripts/classify_complexes.py --sample 60 --out data/klifs/pocket_content.csv",
    ),
    "pocket_content_calibration.csv": Origin(
        ORIGIN_DERIVED,
        "python scripts/classify_complexes.py --calibration "
        "--out data/klifs/pocket_content_calibration.csv",
    ),
    "calibration_diffsbdd_ready.csv": Origin(
        ORIGIN_DERIVED,
        "python scripts/classify_complexes.py --reselect-calibration "
        "--out data/klifs/calibration_diffsbdd_ready.csv",
    ),
}


def file_digest(path: Path) -> str:
    """Возвращает sha256 файла в шестнадцатеричном виде."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_records(path: Path) -> int:
    """Считает строки данных в CSV: без заголовка и без пустого хвоста.

    Читается разборщиком `csv`, а не подсчётом переводов строки: поле с переносом
    внутри кавычек дало бы завышенное число, и расхождение выглядело бы как порча файла.
    """
    with path.open(encoding="utf-8", newline="") as файл:
        строк = sum(1 for строка in csv.reader(файл) if строка)
    return max(строк - 1, 0)


def _git_field(path: Path, формат: str, repo_root: Path) -> str | None:
    """Поле последнего коммита, изменившего файл; `None` — файл вне git.

    Путь за пределами репозитория до git не доходит вовсе: `git log` на таком пути
    завершается кодом 128, и разбирать его сообщение пришлось бы по тексту, который
    зависит от локали. Принадлежность каталогу проверяется здесь и сразу.
    """
    try:
        внутри = path.resolve().is_relative_to(repo_root.resolve())
    except OSError:
        return None
    if not внутри:
        return None

    результат = subprocess.run(
        ["git", "log", "-1", f"--format={формат}", "--", str(path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return результат.stdout.strip() or None


def last_commit_date(path: Path, *, repo_root: Path = PROJECT_ROOT) -> str | None:
    """Дата последнего коммита, изменившего файл, в виде `ГГГГ-ММ-ДД`.

    `None` означает, что файл в git не отслеживается: тогда верхней границы обращения
    к базе у нас нет, и манифест обязан сказать об этом, а не молчать.
    """
    return _git_field(path, "%cs", repo_root)


def last_commit_hash(path: Path, *, repo_root: Path = PROJECT_ROOT) -> str | None:
    """Короткий хеш последнего коммита, изменившего файл."""
    return _git_field(path, "%h", repo_root)


def client_version() -> str | None:
    """Версия установленного `opencadd` или `None`, если пакета в окружении нет."""
    try:
        import opencadd
    except ImportError:
        return None
    версия = getattr(opencadd, "__version__", None)
    return str(версия) if версия else None


def data_files(klifs_dir: Path) -> list[Path]:
    """Файлы каталога, кроме самого манифеста, в алфавитном порядке."""
    return sorted(
        путь for путь in klifs_dir.iterdir() if путь.is_file() and путь.name != MANIFEST_NAME
    )


def build_manifest(
    klifs_dir: Path = KLIFS_DIR, *, repo_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Собирает манифест по фактическому содержимому каталога выгрузки.

    Файл без объявленного происхождения обрывает сборку: приписать ему команду
    по догадке значит завести в репозиторий непроверяемое утверждение — ровно то,
    против чего манифест и заводится.
    """
    if not klifs_dir.is_dir():
        raise ManifestError(f"Каталог выгрузки не найден: {klifs_dir}")

    файлы: dict[str, Any] = {}
    for путь in data_files(klifs_dir):
        источник = FILE_ORIGINS.get(путь.name)
        if источник is None:
            raise ManifestError(
                f"{путь.name}: происхождение не объявлено в FILE_ORIGINS. "
                "Добавьте файл туда вместе с командой, которой он получен"
            )
        дата = last_commit_date(путь, repo_root=repo_root)
        коммит = last_commit_hash(путь, repo_root=repo_root)
        запись: dict[str, Any] = {
            "origin": источник.kind,
            "expected_command": источник.expected_command,
            "retrieved_not_later_than": дата,
            "date_basis": (
                f"коммит {коммит}, последний изменивший файл"
                if дата
                else "файл не отслеживается git: верхней границы нет"
            ),
            "records": count_records(путь),
            "bytes": путь.stat().st_size,
            "sha256": file_digest(путь),
        }
        файлы[путь.name] = запись

    return {
        "schema": MANIFEST_SCHEMA,
        "generator": "python scripts/klifs_manifest.py --write",
        "source": {
            "database": "KLIFS",
            "access": "opencadd.databases.klifs.setup_remote()",
            "client_version": client_version(),
            "database_version": None,
            "database_version_note": (
                "KLIFS через opencadd версию выгрузки не сообщает: объект сессии несёт "
                "только таблицы. Проверено 14.09.2026 перечислением её атрибутов"
            ),
            "command_note": (
                "expected_command — команда, которой файл должен получаться, а не запись "
                "того, чем он получен: строки объявлены константами в FILE_ORIGINS "
                "и одинаковы при любом прогоне. Фактическая строка вызова не сохраняется "
                "(ожидание, а не протокол)"
            ),
        },
        "files": файлы,
    }


def manifest_path(klifs_dir: Path = KLIFS_DIR) -> Path:
    """Путь к файлу манифеста внутри каталога выгрузки."""
    return klifs_dir / MANIFEST_NAME


def write_manifest(klifs_dir: Path = KLIFS_DIR, *, repo_root: Path = PROJECT_ROOT) -> Path:
    """Пересобирает манифест и записывает его. Возвращает путь к файлу."""
    манифест = build_manifest(klifs_dir, repo_root=repo_root)
    путь = manifest_path(klifs_dir)
    путь.write_text(json.dumps(манифест, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return путь


def read_manifest(klifs_dir: Path = KLIFS_DIR) -> dict[str, Any]:
    """Читает манифест каталога выгрузки."""
    путь = manifest_path(klifs_dir)
    if not путь.is_file():
        raise ManifestError(
            f"Манифеста нет: {путь}. Соберите его командой "
            "`python scripts/klifs_manifest.py --write`"
        )
    данные: dict[str, Any] = json.loads(путь.read_text(encoding="utf-8"))
    return данные


def check_manifest(klifs_dir: Path = KLIFS_DIR) -> list[str]:
    """Сверяет манифест с файлами каталога и возвращает список расхождений.

    Пустой список означает, что манифест описывает ровно то, что лежит на диске.
    Состав каталога, размер, число записей и sha256 проверяются по отдельности:
    «файл подменён» и «файл дописан» чинятся по-разному, и сообщение должно
    различать эти случаи, а не говорить «не совпало».
    """
    манифест = read_manifest(klifs_dir)
    объявленные: dict[str, Any] = манифест.get("files", {})
    расхождения: list[str] = []

    на_диске = {путь.name: путь for путь in data_files(klifs_dir)}
    for имя in sorted(set(объявленные) - set(на_диске)):
        расхождения.append(f"{имя}: объявлен в манифесте, но на диске его нет")
    for имя in sorted(set(на_диске) - set(объявленные)):
        расхождения.append(f"{имя}: лежит в каталоге, но в манифесте не объявлен")

    for имя in sorted(set(объявленные) & set(на_диске)):
        путь = на_диске[имя]
        запись = объявленные[имя]

        размер = путь.stat().st_size
        if запись.get("bytes") != размер:
            расхождения.append(f"{имя}: в манифесте {запись.get('bytes')} байт, на диске {размер}")

        записей = count_records(путь)
        if запись.get("records") != записей:
            расхождения.append(
                f"{имя}: в манифесте {запись.get('records')} записей, на диске {записей}"
            )

        дайджест = file_digest(путь)
        if запись.get("sha256") != дайджест:
            расхождения.append(
                f"{имя}: sha256 не совпал — в манифесте {str(запись.get('sha256'))[:12]}, "
                f"на диске {дайджест[:12]}"
            )

    return расхождения
