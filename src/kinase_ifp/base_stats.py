"""Описательная статистика выгрузки KLIFS: состав набора и вырожденность скора.

Отвечает на два вопроса текста курсовой, которые до сих пор считались разово в чьей-то
сессии и командой нигде не воспроизводились: из чего состоит набор данных (пункт B.2
плана подачи) и почему скор принимает мало значений (пункт B.4).

Второй вопрос важнее первого. В «Выводах» вырожденность скора записана как ограничение
работы — «скор принимает не более шести значений». Здесь измеряется, что это свойство
не нашей мишени, а схемы KLIFS в приложении к киназам: направленных бит у эталона мало
у подавляющего большинства структур базы, а не у 6tgu по невезению.

Расчёта взаимодействий здесь нет: таблицы читаются функциями `klifs.py`, пороги берутся
из `config.py`. Модуль намеренно только считает уже выгруженное и в сеть не ходит.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from kinase_ifp.config import (
    DATA_DIR,
    INHIBITOR_TYPE_I,
    KLIFS_IFP_LENGTH,
    KLIFS_INTERACTION_TYPES,
    LIGAND_CLASS_NONE,
    LIGAND_CLASS_NUCLEOTIDE,
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
    N_KLIFS_INTERACTION_TYPES,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.klifs import LIGAND_CLASS_COLUMN, annotate_ligand_class, read_fingerprint_table

KLIFS_DIR: Final[Path] = DATA_DIR / "klifs"

#: Колонки таблицы структур, по которым идёт отбор мишени. Объявлены здесь, а не
#: по месту использования: их читают и состав выгрузки, и рисунок состава базы,
#: а разъехавшиеся имена колонок дали бы два разных распределения под одной подписью.
RESOLUTION_COLUMN: Final[str] = "structure.resolution"
QUALITY_COLUMN: Final[str] = "structure.qualityscore"

#: Порог, до которого считается доля структур с малым числом направленных бит.
#: Пять — потому что столько их у эталона 6tgu: вопрос текста звучит как «повезло ли нам
#: с бедной мишенью», и ответом служит доля структур базы, у которых бит не больше.
FEW_DIRECTED_BITS: Final[int] = 5


class BaseStatsError(RuntimeError):
    """Выгрузка на диске не та, по которой считают: нет файла, колонки или строк."""


@dataclass(frozen=True)
class Distribution:
    """Распределение числовой колонки с отметкой порога отбора.

    Порог хранится рядом с квартилями намеренно: число «медиана разрешения 2.1 Å»
    ничего не говорит, пока не сказано, что отбор шёл по 3.0 Å.
    """

    name: str
    count: int
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    threshold: float
    passing: int

    @property
    def passing_share(self) -> float:
        """Доля значений, проходящих порог; ноль при пустом распределении."""
        return self.passing / self.count if self.count else 0.0


@dataclass(frozen=True)
class DatasetComposition:
    """Состав выгрузки по шагам отбора — таблица для раздела «Материалы и методы»."""

    downloaded: int
    selected: int
    kinases: int
    fingerprints: int
    with_ligand: int
    nucleotide: int
    resolution: Distribution
    quality: Distribution

    @property
    def nucleotide_share(self) -> float:
        """Доля нуклеотидных комплексов среди структур типа I с лигандом."""
        return self.nucleotide / self.with_ligand if self.with_ligand else 0.0


@dataclass(frozen=True)
class ReferenceBits:
    """Сколько бит несут эталонные отпечатки базы и сколько из них идёт в скор."""

    structures: int
    directed_median: float
    directed_mean: float
    directed_minimum: int
    directed_maximum: int
    few_directed: int
    bits_by_type: tuple[tuple[str, int], ...]
    positions_median: float
    directed_positions_median: float

    @property
    def total_bits(self) -> int:
        """Сколько единиц во всех эталонных отпечатках вместе."""
        return sum(количество for _, количество in self.bits_by_type)

    @property
    def few_directed_share(self) -> float:
        """Доля структур, у которых направленных бит не больше `FEW_DIRECTED_BITS`."""
        return self.few_directed / self.structures if self.structures else 0.0

    @property
    def hydrophobic_share(self) -> float:
        """Доля гидрофобных бит во всей базе — величина, из-за которой `HYD` исключён."""
        всего = self.total_bits
        if not всего:
            return 0.0
        гидрофобных = dict(self.bits_by_type).get("HYD", 0)
        return гидрофобных / всего


def _read_table(path: Path) -> pd.DataFrame:
    """Читает таблицу структур; отсутствие файла — отказ, а не пустая таблица.

    Пустая таблица здесь опаснее исключения: все доли посчитались бы от нуля и статистика
    вышла бы правдоподобной и бессмысленной.
    """
    if not path.exists():
        raise BaseStatsError(f"нет файла выгрузки {path}")
    таблица = pd.read_csv(path)
    if таблица.empty:
        raise BaseStatsError(f"{path}: таблица пуста")
    return таблица


def _distribution(
    values: pd.Series, name: str, threshold: float, *, keep_below: bool
) -> Distribution:
    """Квартили колонки и число значений по нужную сторону порога.

    `keep_below` разводит два смысла порога: разрешение проходит, когда оно **не больше**
    порога, оценка качества — когда **не меньше**. Границы включительны, как в
    `filter_structures`.
    """
    числа = [float(значение) for значение in values.dropna()]
    if not числа:
        raise BaseStatsError(f"колонка {name}: нет ни одного значения")
    подходят = sum(1 for з in числа if (з <= threshold if keep_below else з >= threshold))
    квартили = statistics.quantiles(числа, n=4) if len(числа) > 1 else [числа[0]] * 3
    return Distribution(
        name=name,
        count=len(числа),
        minimum=min(числа),
        q1=квартили[0],
        median=statistics.median(числа),
        q3=квартили[2],
        maximum=max(числа),
        threshold=threshold,
        passing=подходят,
    )


def dataset_composition(klifs_dir: Path = KLIFS_DIR) -> DatasetComposition:
    """Считает состав выгрузки: сколько структур на каждом шаге отбора и чем представлены.

    Принимает каталог выгрузки, возвращает `DatasetComposition`. Нуклеотидные комплексы
    считаются среди структур **типа I с лигандом** — так задан знаменатель,
    и менять его нельзя, иначе доля перестанет сравниваться с записанной там.
    """
    всё = annotate_ligand_class(_read_table(klifs_dir / "structures_all.csv"))
    отобранные = _read_table(klifs_dir / "structures.csv")
    киназы = _read_table(klifs_dir / "kinases.csv")
    отпечатки = read_fingerprint_table(klifs_dir / "klifs_ifp.csv")

    тип_i = всё[всё["inhibitor_type"] == INHIBITOR_TYPE_I]
    с_лигандом = тип_i[тип_i[LIGAND_CLASS_COLUMN] != LIGAND_CLASS_NONE]
    нуклеотид = с_лигандом[с_лигандом[LIGAND_CLASS_COLUMN] == LIGAND_CLASS_NUCLEOTIDE]

    return DatasetComposition(
        downloaded=len(всё),
        selected=len(отобранные),
        kinases=len(киназы),
        fingerprints=len(отпечатки),
        with_ligand=len(с_лигандом),
        nucleotide=len(нуклеотид),
        resolution=_distribution(
            всё[RESOLUTION_COLUMN], "разрешение, Å", MAX_RESOLUTION, keep_below=True
        ),
        quality=_distribution(
            всё[QUALITY_COLUMN], "оценка качества", MIN_QUALITY_SCORE, keep_below=False
        ),
    )


def _bit_positions(bits: str) -> tuple[tuple[int, str], ...]:
    """Разбирает битовую строку эталона в пары «позиция кармана, тип взаимодействия».

    Раскладка: позиция = `i // 7`, тип = `i % 7`. Длина проверяется здесь,
    а не у вызывающего: строка не той длины разложится молча и даст правдоподобный вздор.
    """
    if len(bits) != KLIFS_IFP_LENGTH:
        raise BaseStatsError(f"битовая строка длины {len(bits)}, ожидалось {KLIFS_IFP_LENGTH}")
    пары: list[tuple[int, str]] = []
    for индекс, символ in enumerate(bits):
        if символ == "1":
            позиция, номер_типа = divmod(индекс, N_KLIFS_INTERACTION_TYPES)
            пары.append((позиция, KLIFS_INTERACTION_TYPES[номер_типа]))
    return tuple(пары)


def selection_values(klifs_dir: Path = KLIFS_DIR) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Значения разрешения и оценки качества по всей выгрузке — для гистограмм.

    Возвращает две последовательности: сами числа, а не их квартили. Нужны рисунку
    состава базы, которому квартилей мало, и объявлены рядом с `dataset_composition`
    затем, чтобы имя колонки жило в одном месте. Пустые ячейки отбрасываются — так же,
    как их отбрасывает `_distribution`, иначе доли на рисунке и в таблице разошлись бы.
    """
    всё = _read_table(klifs_dir / "structures_all.csv")
    разрешение = tuple(float(з) for з in всё[RESOLUTION_COLUMN].dropna())
    качество = tuple(float(з) for з in всё[QUALITY_COLUMN].dropna())
    if not разрешение or not качество:
        raise BaseStatsError(
            f"в {klifs_dir / 'structures_all.csv'} нет значений разрешения или оценки качества"
        )
    return разрешение, качество


def reference_bits(klifs_dir: Path = KLIFS_DIR) -> ReferenceBits:
    """Считает по всем эталонным отпечаткам базы, сколько бит попадает в скор.

    Принимает каталог выгрузки, возвращает `ReferenceBits`. Направленными считаются биты
    типов `SCORING_INTERACTION_TYPES` — тех же, по которым считается `ifp_score`
, поэтому их число и есть знаменатель скора структуры.
    """
    таблица = read_fingerprint_table(klifs_dir / "klifs_ifp.csv")
    if таблица.empty:
        raise BaseStatsError(f"{klifs_dir / 'klifs_ifp.csv'}: таблица эталонов пуста")

    направленных: list[int] = []
    позиций: list[int] = []
    направленных_позиций: list[int] = []
    по_типам = dict.fromkeys(KLIFS_INTERACTION_TYPES, 0)

    for строка in таблица["bits"]:
        пары = _bit_positions(str(строка))
        свои_позиции: set[int] = set()
        свои_направленные: set[int] = set()
        сколько = 0
        for позиция, тип in пары:
            по_типам[тип] += 1
            свои_позиции.add(позиция)
            if тип in SCORING_INTERACTION_TYPES:
                сколько += 1
                свои_направленные.add(позиция)
        направленных.append(сколько)
        позиций.append(len(свои_позиции))
        направленных_позиций.append(len(свои_направленные))

    return ReferenceBits(
        structures=len(направленных),
        directed_median=statistics.median(направленных),
        directed_mean=statistics.mean(направленных),
        directed_minimum=min(направленных),
        directed_maximum=max(направленных),
        few_directed=sum(1 for сколько in направленных if сколько <= FEW_DIRECTED_BITS),
        bits_by_type=tuple((тип, по_типам[тип]) for тип in KLIFS_INTERACTION_TYPES),
        positions_median=statistics.median(позиций),
        directed_positions_median=statistics.median(направленных_позиций),
    )
