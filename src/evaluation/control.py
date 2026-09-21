"""Контрольные метрики молекул: валидность, уникальность, разнообразие, SA score.

Метрики двух родов, и разница между ними важнее, чем кажется. `valid`, `connected`,
`qed`, `sa_score` и `n_heavy_atoms` считаются по одной молекуле и попадают в колонки
`metrics_per_molecule.csv`. `uniqueness` и `diversity` определены только
на наборе целиком, колонок в таблице метрик не имеют и иметь не могут: их место — строка таблицы
«было/стало» и `metrics_summary.csv`.

Все пять по-молекульных метрик заполняются только при `source=diffsbdd`: на наборе поз
 у набора положений это одна молекула в ста конформациях, где они вырождены по построению.
"""

from __future__ import annotations

import importlib.util
import random
import statistics
import types
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import QED, RDConfig, rdFingerprintGenerator

from kinase_ifp.config import ECFP_N_BITS, ECFP_RADIUS


class ControlError(RuntimeError):
    """Контрольную метрику посчитать нельзя: не тот набор или нет нужного модуля."""


def is_valid(mol: Chem.Mol | None) -> int:
    """1, если RDKit разобрал молекулу и `SanitizeMol` прошёл без исключения; иначе 0.

    `None` на входе — это запись SDF, которую разобрать не удалось; такие записи
    считаются невалидными, а не пропускаются (`docs/metrics.md`, 3.3: знаменатель
    включает невалидные молекулы).
    """
    if mol is None:
        return 0
    # Копия: SanitizeMol правит молекулу на месте, а вызывающий код считает по ней
    # отпечаток и ждёт ровно те свойства, что пришли из файла.
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return 0
    return 1


def is_connected(mol: Chem.Mol | None) -> int:
    """1, если молекула состоит из одного связного фрагмента; иначе 0.

    Невалидная молекула (`None`) связной не считается: `GetMolFrags` по ней не считается
    вовсе, и 0 здесь означает «проверить нечего», что для доли по набору равносильно.
    """
    if mol is None:
        return 0
    return int(len(Chem.GetMolFrags(mol)) == 1)


def _санитизированная(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Копия молекулы после `SanitizeMol` или `None`, если санитизация не прошла.

    Копия, а не сама молекула, по той же причине, что и в `is_valid`: вызывающий код
    считает по исходной молекуле отпечаток и ждёт ровно те свойства, что пришли из файла.
    """
    if mol is None:
        return None
    копия = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(копия)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError):
        return None
    return копия


@lru_cache(maxsize=1)
def _sascorer() -> types.ModuleType:
    """Модуль `sascorer` из RDKit Contrib.

    Загружается по пути, а не обычным импортом: `sascorer` лежит вне пакетов, доступных
    импорту, и канонический рецепт из документации RDKit — дописать `sys.path` между
    импортами. В этом проекте так не делают (см. шапку `evaluation/figures.py`), поэтому
    модуль поднимается явно через `importlib`.

    Кэш обязателен: при первом обращении `sascorer` распаковывает `fpscores.pkl.gz`,
    и без кэша это повторялось бы на каждой молекуле набора.
    """
    путь = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    if not путь.is_file():
        raise ControlError(
            f"В сборке RDKit нет {путь}: SA score считать нечем "
            "(`docs/metrics.md`, 3.4 требует именно sascorer из Contrib)"
        )
    spec = importlib.util.spec_from_file_location("sascorer", путь)
    if spec is None or spec.loader is None:
        raise ControlError(f"Не удалось загрузить модуль {путь}")
    модуль = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(модуль)
    return модуль


def qed(mol: Chem.Mol | None) -> float | None:
    """QED молекулы, 0…1, больше — лучше. `None`, если молекула невалидна.

    Пустая ячейка у невалидной молекулы — это «метрика не определена», а не забытый
    расчёт: `docs/metrics.md`, раздел 4 разрешает её прямо.
    """
    молекула = _санитизированная(mol)
    if молекула is None:
        return None
    return float(QED.qed(молекула))


def sa_score(mol: Chem.Mol | None) -> float | None:
    """SA score молекулы, 1…10, меньше — лучше. `None`, если молекула невалидна."""
    молекула = _санитизированная(mol)
    if молекула is None:
        return None
    return float(_sascorer().calculateScore(молекула))


def n_heavy_atoms(mol: Chem.Mol | None) -> int | None:
    """Число тяжёлых атомов молекулы. `None`, если молекула невалидна.

    Метрика обязательная, а не справочная: без неё нельзя отличить «потенциал улучшил
    связывание» от «потенциал раздул молекулы» (`docs/metrics.md`, 3.4).
    """
    молекула = _санитизированная(mol)
    if молекула is None:
        return None
    return int(молекула.GetNumHeavyAtoms())


def canonical_smiles(mol: Chem.Mol) -> str:
    """Канонический SMILES без явных водородов.

    `RemoveHs` обязателен: молекулы прогона приходят с явными водородами,
    и без него два одинаковых графа с разной расстановкой H дают разные строки,
    то есть считаются разными молекулами.
    """
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))


def uniqueness(mols: Sequence[Chem.Mol | None]) -> tuple[float, int, int]:
    """Доля различных канонических SMILES, а также числитель и знаменатель отдельно.

    Числитель — число различных SMILES **среди валидных** молекул, знаменатель — размер
    **всего** набора, включая невалидные. Так написано в `docs/metrics.md`, 3.3, и там же
    требуется называть знаменатель явно — поэтому возвращается тройка, а не одно число.
    Невалидные молекулы входят в знаменатель.
    """
    if not mols:
        raise ControlError("Набор пуст: uniqueness не определён")
    уникальные = {
        canonical_smiles(mol) for mol in mols if mol is not None and is_valid(mol) == 1
    }
    return len(уникальные) / len(mols), len(уникальные), len(mols)


def diversity(mols: Sequence[Chem.Mol | None]) -> float:
    """Средняя непохожесть молекул набора: 1 − среднее попарное Танимото по ECFP4.

    Считается только по валидным молекулам (`docs/metrics.md`, 3.5). Падение diversity
    при росте целевых метрик означает, что потенциал загнал генерацию в одну область
    химического пространства, — это провал, а не успех.

    Поднимает `ControlError`, если валидных молекул меньше двух: делитель `N(N−1)`
    обращается в ноль, и вернуть здесь 0.0 значило бы выдать «все молекулы одинаковы»
    там, где сравнивать нечего.
    """
    валидные = [молекула for mol in mols if (молекула := _санитизированная(mol)) is not None]
    if len(валидные) < 2:
        raise ControlError(
            f"Валидных молекул {len(валидные)}: diversity определена от двух и больше"
        )
    генератор = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_N_BITS)
    отпечатки = [генератор.GetFingerprint(молекула) for молекула in валидные]
    сумма = 0.0
    for индекс in range(1, len(отпечатки)):
        сумма += sum(DataStructs.BulkTanimotoSimilarity(отпечатки[индекс], отпечатки[:индекс]))
    пар = len(отпечатки) * (len(отпечатки) - 1) / 2
    return 1.0 - сумма / пар


# --- разнообразие отбора против разнообразия меньшего набора ---------------
#
# Утверждение «разнообразие падает при ужесточении отбора»
# нельзя защитить рядом из шести чисел: набор при этом становится меньше, а меньший
# набор разнообразнее или беднее сам по себе, независимо от того, чем он отобран.
# Сравнение идёт со случайными подмножествами того же размера из того же прогона.


@dataclass(frozen=True)
class DiversityAgainstRandom:
    """Разнообразие отобранного подмножества против случайных того же размера."""

    k: int
    observed: float
    random_median: float
    random_low: float
    random_high: float
    share_below: float

    @property
    def distinguishable(self) -> bool:
        """Наблюдаемое значение вышло за 95 % случайных подмножеств того же размера."""
        return not self.random_low <= self.observed <= self.random_high


def pairwise_similarity(mols: Sequence[Chem.Mol | None]) -> list[list[float]]:
    """Матрица попарного Танимото по ECFP4 для валидных молекул набора.

    Принимает молекулы, возвращает симметричную матрицу со сходством `[i][j]`.
    Нужна там, где разнообразие считается многократно по подмножествам одного набора:
    пересчитывать отпечатки на каждом розыгрыше незачем.

    Поднимает `ControlError`, если валидных молекул меньше двух.
    """
    валидные = [молекула for mol in mols if (молекула := _санитизированная(mol)) is not None]
    if len(валидные) < 2:
        raise ControlError(
            f"Валидных молекул {len(валидные)}: попарное сходство определено от двух и больше"
        )
    генератор = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_N_BITS)
    отпечатки = [генератор.GetFingerprint(молекула) for молекула in валидные]
    матрица = [[0.0] * len(отпечатки) for _ in отпечатки]
    for i in range(1, len(отпечатки)):
        строка = DataStructs.BulkTanimotoSimilarity(отпечатки[i], отпечатки[:i])
        for j, значение in enumerate(строка):
            матрица[i][j] = матрица[j][i] = значение
    return матрица


def subset_diversity(similarity: Sequence[Sequence[float]], indices: Sequence[int]) -> float:
    """Разнообразие подмножества по готовой матрице сходства: 1 − среднее попарное.

    Формула та же, что у `diversity`; отличается только вход — матрица вместо молекул.
    Поднимает `ControlError`, если в подмножестве меньше двух элементов.
    """
    номера = list(indices)
    if len(номера) < 2:
        raise ControlError(
            f"В подмножестве {len(номера)} элементов: diversity определена от двух и больше"
        )
    сумма = sum(
        similarity[номера[i]][номера[j]] for i in range(1, len(номера)) for j in range(i)
    )
    пар = len(номера) * (len(номера) - 1) / 2
    return 1.0 - сумма / пар


def diversity_vs_random(
    similarity: Sequence[Sequence[float]],
    k: int,
    resamples: int = 2000,
    seed: int = 0,
    rng: random.Random | None = None,
) -> DiversityAgainstRandom:
    """Сравнивает разнообразие первых `k` элементов со случайными подмножествами того же размера.

    Принимает матрицу сходства (порядок строк — порядок ранжирования), размер топа `k`
    и число розыгрышей; возвращает `DiversityAgainstRandom` с наблюдаемым значением
    и границами 95 % случайных подмножеств.

    `rng` передаётся тогда, когда несколько долей разыгрываются подряд и поток чисел
    обязан остаться одним: свой `random.Random(seed)` на каждый вызов дал бы другие
    подмножества и другие границы при том же `seed`.

    Поднимает `ControlError` при `k` меньше двух или больше размера набора.
    """
    n = len(similarity)
    if k < 2 or k > n:
        raise ControlError(f"Размер топа {k} вне диапазона 2..{n}")
    генератор = rng if rng is not None else random.Random(seed)
    наблюдаемое = subset_diversity(similarity, range(k))
    случайные = sorted(
        subset_diversity(similarity, генератор.sample(range(n), k)) for _ in range(resamples)
    )
    return DiversityAgainstRandom(
        k=k,
        observed=наблюдаемое,
        random_median=statistics.median(случайные),
        random_low=случайные[int(0.025 * len(случайные))],
        random_high=случайные[min(int(0.975 * len(случайные)), len(случайные) - 1)],
        share_below=sum(1 for з in случайные if з < наблюдаемое) / len(случайные),
    )
