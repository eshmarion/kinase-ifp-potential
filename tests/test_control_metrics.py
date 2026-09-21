"""Тесты контрольных метрик: валидность, уникальность, разнообразие, SA score."""

from __future__ import annotations

import pytest
from rdkit import Chem

from evaluation.control import (
    ControlError,
    canonical_smiles,
    diversity,
    diversity_vs_random,
    is_connected,
    is_valid,
    n_heavy_atoms,
    pairwise_similarity,
    qed,
    sa_score,
    subset_diversity,
    uniqueness,
)

# Фикстура: валидная молекула, разорванная на два фрагмента и невалидная.
# Задаётся кодом, а не файлом: три SMILES читаются глазами, а SDF пришлось бы открывать.
ВАЛИДНАЯ = "CCOc1ccccc1"
РАЗОРВАННАЯ = "CCO.CCN"
# Пятивалентный углерод: RDKit разбирает строку только с sanitize=False и роняет
# SanitizeMol на валентности — ровно то, что в прогоне приходит битой записью SDF.
НЕВАЛИДНАЯ = "C(C)(C)(C)(C)C"


@pytest.fixture
def валидная() -> Chem.Mol:
    return Chem.MolFromSmiles(ВАЛИДНАЯ)


@pytest.fixture
def разорванная() -> Chem.Mol:
    return Chem.MolFromSmiles(РАЗОРВАННАЯ)


@pytest.fixture
def невалидная() -> Chem.Mol:
    молекула = Chem.MolFromSmiles(НЕВАЛИДНАЯ, sanitize=False)
    assert молекула is not None, "фикстура обязана разбираться без санитизации"
    return молекула


def test_валидность_и_связность_на_трёх_молекулах(
    валидная: Chem.Mol, разорванная: Chem.Mol, невалидная: Chem.Mol
) -> None:
    """Валидная — 1/1, разорванная — валидна, но не связна, невалидная — 0."""
    assert (is_valid(валидная), is_connected(валидная)) == (1, 1)
    assert (is_valid(разорванная), is_connected(разорванная)) == (1, 0)
    assert is_valid(невалидная) == 0


def test_неразобранная_запись_sdf_считается_невалидной() -> None:
    """`None` — это запись, которую RDKit не разобрал; она входит в знаменатель.

    Пропустить её значило бы улучшить validity молчанием (`docs/metrics.md`, 3.3).
    """
    assert is_valid(None) == 0
    assert is_connected(None) == 0


def test_qed_в_своём_диапазоне_и_пуст_у_невалидной(
    валидная: Chem.Mol, невалидная: Chem.Mol
) -> None:
    """QED лежит в 0…1; у невалидной молекулы метрики нет, а не ноль."""
    значение = qed(валидная)
    assert значение is not None
    assert 0.0 <= значение <= 1.0
    assert qed(невалидная) is None
    assert qed(None) is None


def test_sa_score_в_диапазоне_один_десять(валидная: Chem.Mol, невалидная: Chem.Mol) -> None:
    """SA score лежит в 1…10 (`docs/metrics.md`, 3.4), у невалидной его нет.

    Заодно проверяется, что `sascorer` из RDKit Contrib вообще поднялся: без него
    функция подняла бы `ControlError`, а не вернула число.
    """
    значение = sa_score(валидная)
    assert значение is not None
    assert 1.0 <= значение <= 10.0
    assert sa_score(невалидная) is None


def test_число_тяжёлых_атомов_не_считает_водороды(валидная: Chem.Mol) -> None:
    """У фенетола 9 тяжёлых атомов; явные водороды на число не влияют."""
    assert n_heavy_atoms(валидная) == 9
    assert n_heavy_atoms(Chem.AddHs(валидная)) == 9
    assert n_heavy_atoms(невалидная := Chem.MolFromSmiles(НЕВАЛИДНАЯ, sanitize=False)) is None
    assert невалидная is not None


def test_канонизация_снимает_явные_водороды(валидная: Chem.Mol) -> None:
    """Один и тот же граф с явными H и без них даёт одну строку.

    Молекулы прогона приходят с явными водородами, и без `RemoveHs`
    они считались бы разными молекулами, завышая uniqueness.
    """
    assert canonical_smiles(Chem.AddHs(валидная)) == canonical_smiles(валидная)


def test_uniqueness_называет_знаменатель_явно(валидная: Chem.Mol) -> None:
    """Числитель — различные SMILES среди валидных, знаменатель — весь набор.

    Прочтение противоречивого места `docs/metrics.md`, 3.3 зафиксировано
    так: невалидные молекулы входят в знаменатель.
    """
    кофеин = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    доля, различных, всего = uniqueness([валидная, Chem.AddHs(валидная), кофеин, None])
    assert (различных, всего) == (2, 4)
    assert доля == pytest.approx(0.5)


def test_uniqueness_на_пустом_наборе_отказывается() -> None:
    """Пустой набор — не ноль уникальности, а отсутствие знаменателя."""
    with pytest.raises(ControlError):
        uniqueness([])


def test_diversity_ноль_на_одинаковых_и_растёт_на_разных(валидная: Chem.Mol) -> None:
    """Две копии одной молекулы дают 0, две разные — заметно больше нуля."""
    копия = Chem.MolFromSmiles(ВАЛИДНАЯ)
    assert diversity([валидная, копия]) == pytest.approx(0.0, abs=1e-9)

    кофеин = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    assert diversity([валидная, кофеин]) > 0.5


def test_diversity_меньше_чем_на_двух_молекулах_отказывается(валидная: Chem.Mol) -> None:
    """Делитель N(N−1) обращается в ноль — вернуть 0.0 значило бы соврать.

    0.0 читается как «все молекулы одинаковы», тогда как сравнивать здесь нечего.
    """
    with pytest.raises(ControlError):
        diversity([валидная])
    with pytest.raises(ControlError):
        diversity([валидная, None])


def test_diversity_считает_только_по_валидным(валидная: Chem.Mol) -> None:
    """Невалидные молекулы в попарные сравнения не входят (`docs/metrics.md`, 3.5)."""
    кофеин = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    битая = Chem.MolFromSmiles(НЕВАЛИДНАЯ, sanitize=False)
    assert diversity([валидная, кофеин, битая, None]) == pytest.approx(
        diversity([валидная, кофеин])
    )


# --- разнообразие подмножества против случайных ---------------


НАБОР_SMILES = (
    "CCOc1ccccc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "c1ccc2ccccc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CCN(CC)CC",
    "OCC1OC(O)C(O)C(O)C1O",
)


@pytest.fixture
def набор() -> list[Chem.Mol]:
    return [Chem.MolFromSmiles(строка) for строка in НАБОР_SMILES]


def test_матрица_сходства_симметрична_и_пуста_по_диагонали(набор: list[Chem.Mol]) -> None:
    матрица = pairwise_similarity(набор)

    assert len(матрица) == len(набор)
    for i in range(len(матрица)):
        assert матрица[i][i] == 0.0
        for j in range(len(матрица)):
            assert матрица[i][j] == матрица[j][i]


def test_разнообразие_по_матрице_совпадает_с_diversity(набор: list[Chem.Mol]) -> None:
    """Одна величина, два входа: молекулы и готовая матрица. Расхождение здесь означало бы,
    что печать команды и таблица «было/стало» считают разное."""
    матрица = pairwise_similarity(набор)

    assert subset_diversity(матрица, range(len(набор))) == pytest.approx(diversity(набор))
    assert subset_diversity(матрица, [0, 1]) == pytest.approx(diversity([набор[0], набор[1]]))


def test_разнообразие_подмножества_меньше_двух_отказывается(набор: list[Chem.Mol]) -> None:
    матрица = pairwise_similarity(набор)

    with pytest.raises(ControlError):
        subset_diversity(матрица, [0])


def test_сравнение_со_случайными_подмножествами_того_же_размера(набор: list[Chem.Mol]) -> None:
    """Направление изменения метрики набора без такого сравнения недоказуемо."""
    матрица = pairwise_similarity(набор)

    проверка = diversity_vs_random(матрица, k=3, resamples=200, seed=0)

    assert проверка.k == 3
    assert проверка.observed == pytest.approx(subset_diversity(матрица, range(3)))
    assert проверка.random_low <= проверка.random_median <= проверка.random_high
    assert 0.0 <= проверка.share_below <= 1.0


def test_полный_набор_неотличим_от_себя(набор: list[Chem.Mol]) -> None:
    """При k = n случайное подмножество — это весь набор, и границы вырождаются в точку."""
    матрица = pairwise_similarity(набор)

    проверка = diversity_vs_random(матрица, k=len(набор), resamples=50, seed=0)

    assert проверка.observed == pytest.approx(проверка.random_median)
    assert проверка.distinguishable is False


def test_зерно_повторяет_розыгрыш(набор: list[Chem.Mol]) -> None:
    матрица = pairwise_similarity(набор)

    первый = diversity_vs_random(матрица, k=3, resamples=100, seed=7)
    второй = diversity_vs_random(матрица, k=3, resamples=100, seed=7)

    assert первый == второй


def test_размер_топа_вне_диапазона_отвергается(набор: list[Chem.Mol]) -> None:
    матрица = pairwise_similarity(набор)

    with pytest.raises(ControlError, match="вне диапазона"):
        diversity_vs_random(матрица, k=1, resamples=10)
    with pytest.raises(ControlError, match="вне диапазона"):
        diversity_vs_random(матрица, k=len(набор) + 1, resamples=10)
