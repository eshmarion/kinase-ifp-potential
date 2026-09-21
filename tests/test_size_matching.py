"""Подбор пар по числу тяжёлых атомов для сравнения отбора без влияния размера."""

from __future__ import annotations

from evaluation.size_matching import SizedItem, match_by_size


def _молекулы(префикс: str, размеры: list[int], значение: float) -> list[SizedItem]:
    return [SizedItem(f"{префикс}{i}", размер, значение) for i, размер in enumerate(размеры)]


def test_пары_строятся_по_ближайшему_размеру() -> None:
    подбор = match_by_size(_молекулы("верх", [20, 30], 1.0), _молекулы("низ", [30, 20], 0.0))
    assert подбор.unmatched == 0
    assert {верх.size: низ.size for верх, низ in подбор.pairs} == {20: 20, 30: 30}
    assert подбор.differences == [1.0, 1.0]


def test_без_пары_молекула_выпадает_и_это_видно() -> None:
    """Крупной молекуле пары нет: ближайшая отбракованная меньше на девять атомов."""
    подбор = match_by_size(_молекулы("верх", [20, 40], 1.0), _молекулы("низ", [20, 31], 0.0))
    assert подбор.unmatched == 1
    assert [верх.size for верх, _ in подбор.pairs] == [20]


def test_отбракованная_молекула_не_используется_дважды() -> None:
    подбор = match_by_size(_молекулы("верх", [25, 25], 1.0), _молекулы("низ", [25], 0.0))
    assert подбор.unmatched == 1
    assert len(подбор.pairs) == 1


def test_подбор_не_зависит_от_порядка_строк() -> None:
    верх = _молекулы("верх", [21, 22, 23], 1.0)
    низ = _молекулы("низ", [22, 23, 21], 0.0)
    прямой = match_by_size(верх, низ)
    обратный = match_by_size(list(reversed(верх)), list(reversed(низ)))
    assert {(в.mol_id, н.mol_id) for в, н in прямой.pairs} == {
        (в.mol_id, н.mol_id) for в, н in обратный.pairs
    }
