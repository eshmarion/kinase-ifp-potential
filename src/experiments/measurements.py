"""Реестр измерений по мишеням: одна строка на одно измеренное число.

Зачем понадобился. Отчёты, которые собирает код, до сих пор хранили числа только
в markdown, и формат каждого был заточен под один вид измерения на одну мишень.
`scripts/calibrate_ifp.py` из-за этого **терял** числа: раздел дымовой сверки узнаётся
слиянием по номеру раздела, а не по мишени, поэтому прогон по 6fnk 18.09 стёр из
отчёта числа 6tgu, на которые опирается выбор порога.

Устройство. Таблица длинная (long), а не широкая: колонка на мишень, вид измерения,
вариант, величину и само значение. Новый вид теста поэтому не требует ни новой колонки,
ни правки формата — он приходит новыми значениями поля `measurement`, а прежние строки
остаются на месте. Широкая таблица («колонка на метрику») этого не позволяет: каждый
новый вид измерения менял бы схему файла и ломал бы уже записанное.

Ключ строки — четвёрка `(target, measurement, variant, metric)`. Повторный прогон
того же измерения заменяет **только** свои строки: в этом и состоит защита от потери
чужих чисел. Значения хранятся строками, как пришли из расчёта: реестр не считает
и не округляет, иначе сравнение «было/стало» пошло бы по округлённым числам.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final

from experiments.layout import CSV_EOL, MOLECULES_SDF
from kinase_ifp.config import MEASUREMENTS_CSV

# Наборы типов взаимодействий, с которых начинается метка условия в имени файла поз.
# Список закрыт и совпадает с тем, что пишет `scripts/optimize_poses.py`: по нему имя
# прогона отделяется от метки условия, а иначе их не различить — в имени прогона дефисов
# столько же, сколько в метке.
TYPE_SET_MARKERS: Final[tuple[str, ...]] = ("all7", "scoring6")

# Префикс файлов оптимизированных поз (`scripts/optimize_poses.py`).
POSES_PREFIX: Final[str] = "pose_optimization-"

# Метка варианта для молекул прогона, не тронутых потенциалом. Отдельное значение,
# а не пустая строка: в реестре рядом стоят три условия — исходные, М0 и М1, — и
# сравнивать их можно только если у исходных есть свой ключ, а не отсутствие ключа.
SOURCE_VARIANT: Final[str] = "source"

# Порядок колонок файла. Он же порядок полей `Measurement`, и тест это держит:
# читатели реестра рассчитывают на стабильную шапку.
COLUMNS: Final[tuple[str, ...]] = (
    "target",
    "measurement",
    "variant",
    "metric",
    "value",
    "unit",
    "measured_utc",
    "source",
    "notes",
)

# Поля, которые вместе задают одну измеренную величину. Повторная запись с той же
# четвёркой — обновление, а не вторая строка.
KEY_COLUMNS: Final[tuple[str, ...]] = ("target", "measurement", "variant", "metric")


class MeasurementError(RuntimeError):
    """Реестр измерений устроен не так, как ожидает запись или чтение."""


@dataclass(frozen=True)
class Measurement:
    """Одно измеренное число с указанием, что, где и чем измерено.

    `variant` — условие измерения: режим протонирования, набор типов взаимодействий,
    идентификатор прогона. Пустая строка означает, что у измерения условий нет.
    `source` — чем получено число: команда или модуль, по которому его можно повторить.
    """

    target: str
    measurement: str
    variant: str
    metric: str
    value: str
    unit: str = ""
    measured_utc: str = ""
    source: str = ""
    notes: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Четвёрка, по которой строка опознаётся при обновлении."""
        return (self.target, self.measurement, self.variant, self.metric)

    def as_row(self) -> dict[str, str]:
        """Строка для записи в CSV в порядке `COLUMNS`."""
        return {поле.name: str(getattr(self, поле.name)) for поле in fields(self)}


def read_measurements(path: Path = MEASUREMENTS_CSV) -> tuple[Measurement, ...]:
    """Читает реестр измерений; отсутствующий файл — пустой реестр, а не ошибка.

    Принимает путь к CSV, возвращает строки в том порядке, в каком они лежат в файле.
    Поднимает `MeasurementError`, если шапка не совпадает с `COLUMNS`: молча прочитать
    файл чужого формата опаснее, чем остановиться.
    """
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8", newline="") as поток:
        читатель = csv.DictReader(поток)
        if tuple(читатель.fieldnames or ()) != COLUMNS:
            raise MeasurementError(
                f"{path}: шапка {читатель.fieldnames} не совпадает с ожидаемой {list(COLUMNS)}"
            )
        return tuple(
            Measurement(**{колонка: (строка.get(колонка) or "") for колонка in COLUMNS})
            for строка in читатель
        )


def variant_from_poses(path: Path) -> str:
    """Ключ условия `<прогон>:<метка>` по имени файла молекул или оптимизированных поз.

    Принимает `runs/<прогон>/molecules.sdf` либо
    `results/pose_optimization-<прогон>-<метка>.sdf`; возвращает строку для поля
    `variant` реестра измерений.

    Нужна затем, чтобы физичность, докинг и оптимизация позы ложились в реестр под
    **одним** ключом: тогда в тексте исходные молекулы, М0 и М1 берутся одной выборкой
    по ключу, а не сшиваются глазами из трёх сводок.

    Имя прогона от метки условия отделяется по набору типов (`all7`, `scoring6`),
    а не по последнему дефису: дефисов в имени прогона столько же, сколько в метке,
    и позиционный разбор рано или поздно разрежет не там. Поднимает `MeasurementError`,
    если имя не опознано — молчаливый возврат пустого ключа склеил бы в реестре разные
    условия в одну строку.
    """
    if path.name == MOLECULES_SDF:
        return f"{path.parent.name}:{SOURCE_VARIANT}"
    основа = path.stem
    if not основа.startswith(POSES_PREFIX):
        raise MeasurementError(
            f"не опознано имя файла поз {path.name}: ожидался {MOLECULES_SDF} "
            f"или {POSES_PREFIX}<прогон>-<метка>.sdf"
        )
    остаток = основа[len(POSES_PREFIX) :]
    for набор in TYPE_SET_MARKERS:
        маркер = f"-{набор}"
        if маркер in остаток:
            граница = остаток.index(маркер)
            return f"{остаток[:граница]}:{остаток[граница + 1 :]}"
    raise MeasurementError(
        f"в имени {path.name} нет набора типов из {list(TYPE_SET_MARKERS)}: "
        "имя прогона от метки условия отделить нечем"
    )


def upsert_measurements(
    rows: Iterable[Measurement], path: Path = MEASUREMENTS_CSV
) -> tuple[Measurement, ...]:
    """Дописывает измерения в реестр, заменяя только строки с теми же ключами.

    Принимает новые измерения и путь к реестру; возвращает реестр целиком после записи.
    Строки, чьих ключей среди новых нет, сохраняются дословно — ради этого реестр
    и заведён. Порядок итогового файла — сортировка по ключу, чтобы diff показывал
    изменённые числа, а не переехавшие строки.
    """
    новые = list(rows)
    ключи = {измерение.key for измерение in новые}
    if len(ключи) != len(новые):
        повторы = sorted({и.key for и in новые if sum(1 for д in новые if д.key == и.key) > 1})
        raise MeasurementError(f"в одной записи повторяются ключи: {повторы}")

    прежние = [измерение for измерение in read_measurements(path) if измерение.key not in ключи]
    итог = tuple(sorted(прежние + новые, key=lambda измерение: измерение.key))

    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV пишется с '\n': модуль csv по умолчанию ставит CRLF, и файл,
    # записанный кодом, отличался бы в diff от того же файла, сохранённого git.
    with path.open("w", encoding="utf-8", newline="") as поток:
        писатель = csv.DictWriter(поток, fieldnames=list(COLUMNS), lineterminator=CSV_EOL)
        писатель.writeheader()
        писатель.writerows(измерение.as_row() for измерение in итог)
    return итог


def select(
    rows: Iterable[Measurement],
    *,
    measurement: str | None = None,
    target: str | None = None,
    variant: str | None = None,
) -> tuple[Measurement, ...]:
    """Отбирает измерения по виду, мишени и варианту; `None` — не фильтровать."""
    return tuple(
        измерение
        for измерение in rows
        if (measurement is None or измерение.measurement == measurement)
        and (target is None or измерение.target == target)
        and (variant is None or измерение.variant == variant)
    )


def targets(rows: Iterable[Measurement], measurement: str | None = None) -> tuple[str, ...]:
    """Мишени, по которым есть измерения данного вида, в алфавитном порядке."""
    return tuple(sorted({измерение.target for измерение in select(rows, measurement=measurement)}))


def markdown_table(
    rows: Iterable[Measurement],
    measurement: str,
    metrics: Sequence[tuple[str, str]],
    *,
    row_header: str = "Мишень",
    variant_header: str | None = "Режим",
    date_header: str | None = None,
) -> list[str]:
    """Собирает markdown-таблицу «строка на мишень, колонка на величину».

    Принимает измерения, вид измерения и пары «имя величины — заголовок колонки»;
    возвращает строки таблицы. Отсутствующее значение печатается прочерком, а не нулём:
    неизмеренное и измеренный нуль — разные вещи, и в отчёте их нельзя путать.

    `date_header` добавляет колонку с датой измерения. Дата стоит в строке, а не в шапке
    отчёта: мишени считаются в разные дни, и общая дата раздела говорила бы, что все
    числа получены разом.
    """
    отобранные = select(rows, measurement=measurement)
    заголовки = [row_header]
    if variant_header is not None:
        заголовки.append(variant_header)
    заголовки += [заголовок for _, заголовок in metrics]
    if date_header is not None:
        заголовки.append(date_header)

    таблица = ["| " + " | ".join(заголовки) + " |", "|" + "---|" * len(заголовки)]
    пары = sorted({(измерение.target, измерение.variant) for измерение in отобранные})
    for мишень, вариант in пары:
        группа = [
            измерение
            for измерение in отобранные
            if измерение.target == мишень and измерение.variant == вариант
        ]
        значения = {измерение.metric: измерение.value for измерение in группа}
        ячейки = [мишень]
        if variant_header is not None:
            ячейки.append(f"`{вариант}`" if вариант else "—")
        ячейки += [значения.get(имя, "—") for имя, _ in metrics]
        if date_header is not None:
            даты = {измерение.measured_utc for измерение in группа if измерение.measured_utc}
            ячейки.append(max(даты) if даты else "—")
        таблица.append("| " + " | ".join(ячейки) + " |")
    return таблица
