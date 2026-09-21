"""Три варианта построения потенциала на одном и том же отпечатке.

Зачем модуль нужен. До сих пор целевая функция оптимизации была вшита в `optimize_pose`
одной строкой `tversky(эталон, поза, alpha=1, beta=0)`, и заменить её было нельзя,
не переписав оптимизатор. Измерение 18.09.2026 показало, что эта формула покупает
воспроизведение взаимодействий ценой энергии и физичности (`results/
pocket_mode_comparison.md`), и вопрос «что даст другая формула» стал содержательным.
Здесь целевая функция выделена в объект, а оптимизатор перестал зависеть от её устройства.

Три варианта — это три способа построения потенциала, у каждого литературный образец:

* **М0, дискретный** (`DiscretePotential`) — базовая формула, как есть. Класс
  известен с 2006 года: Mpamhanga, Chen, McLay, Willett, JCIM 46(2):686 кодируют позы
  докинга битовыми строками и оценивают сходством с эталонной модой связывания.
* **М1, с членом клэша** (`ClashPenaltyPotential`) — тот же скор минус шарнирный штраф
  за наложение на белок. Образец — `delta_SC` в BInD (Lee, Zhung, Seo, Kim, Adv Sci
  2025, 12(35)), где стерический клэш входит **слагаемым** целевой функции, а не
  фильтром шага, как у нас до сих пор.
* **М2, взвешенный** (`WeightedPotential`) — биты взвешены частотой в базе KLIFS.
  Образец — PADIF (Jasper, Humbeck, Brinkjost, Koch, J Cheminform 2018, 10:15).

**Правила расчёта отпечатка одинаковы во всех трёх.** Различается только то, как из
готового отпечатка получается число: раскладка (85, 7), пороги `KLIFS_RULE_*` и состав
типов не трогаются нигде в этом модуле. Это условие сравнимости: иначе три варианта
считали бы разные величины, и разница между ними ничего не говорила бы о формуле.

Все три возвращают величину «больше — лучше», и оптимизатор одинаково ведёт себя
с любой из них. У М0 и М2 значения лежат в [0, 1]; у М1 нижней границы нет, потому
что штраф не ограничен — поза, вжатая в белок, получает сколь угодно плохое значение,
и это ровно то поведение, которого от члена клэша ждут.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rdkit import Chem

from kinase_ifp.bit_weights import cached_bit_weights
from kinase_ifp.config import (
    CLASH_PENALTY_WEIGHT,
    CLASH_VDW_FRACTION,
    KLIFS_IFP_CSV,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.similarity import select_types, tversky, type_rows

# Имена вариантов для командной строки и для меток в именах файлов результатов.
POTENTIAL_VARIANTS: tuple[str, ...] = ("discrete", "clash", "weighted")


class Potential(Protocol):
    """Целевая функция оптимизации позы: «больше — лучше».

    Принимает молекулу и её уже посчитанный отпечаток. Отпечаток передаётся готовым,
    а не считается внутри, потому что оптимизатор считает его один раз на пробную позу
    и отдаёт всем потребителям: расчёт отпечатка — самая дорогая часть шага.
    """

    def __call__(self, mol: Chem.Mol, ifp: np.ndarray) -> float: ...

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class DiscretePotential:
    """М0: доля взаимодействий эталона, воспроизведённая позой.

    Формула не изменена ни в чём: `tversky(эталон, поза, alpha=1, beta=0)` по заданному
    набору типов. Лишние контакты не штрафуются намеренно (`beta=0`), и это остаётся
    выбором проекта, а не упрощением.
    """

    reference_ifp: np.ndarray
    scoring_types: Sequence[str] = SCORING_INTERACTION_TYPES

    @property
    def name(self) -> str:
        return "discrete"

    def __call__(self, mol: Chem.Mol, ifp: np.ndarray) -> float:
        эталон = select_types(self.reference_ifp, self.scoring_types)
        return tversky(эталон, select_types(ifp, self.scoring_types), alpha=1.0, beta=0.0)


@dataclass(frozen=True)
class WeightedPotential:
    """М2: то же, но каждый бит взвешен частотой своего взаимодействия в базе KLIFS.

    Формула — взвешенный Tversky с теми же `alpha=1, beta=0`:

        score = sum(w[i] for i in A & B) / sum(w[i] for i in A),

    где `A` — биты эталона, `B` — биты позы, `w` — веса из `bit_weights`. При всех
    весах, равных единице, это ровно формула М0 (проверяется тестом
    `test_веса_из_единиц_дают_базовый_скор`), то есть М2 — её обобщение, а не замена.

    Смысл: воспроизвести H-связь с каталитическим лизином, которая есть у тысяч
    киназных комплексов, ценнее, чем случайный контакт на краю кармана.
    """

    reference_ifp: np.ndarray
    weights: np.ndarray
    scoring_types: Sequence[str] = SCORING_INTERACTION_TYPES

    @property
    def name(self) -> str:
        return "weighted"

    def __call__(self, mol: Chem.Mol, ifp: np.ndarray) -> float:
        эталон = select_types(self.reference_ifp, self.scoring_types).astype(bool)
        поза = select_types(ifp, self.scoring_types).astype(bool)
        # Веса вещественные, поэтому режутся `type_rows`, а не `select_types`:
        # последняя проверяет, что значения массива — только 0 и 1.
        веса = np.asarray(self.weights)[type_rows(self.scoring_types)]
        знаменатель = float(веса[эталон].sum())
        if знаменатель == 0.0:
            # У эталона нет ни одного бита нужных типов: воспроизводить нечего.
            # Ноль здесь — то же соглашение, что у `tversky` при пустом знаменателе.
            return 0.0
        return float(веса[эталон & поза].sum()) / знаменатель


@dataclass(frozen=True)
class ClashPenaltyPotential:
    """М1: скор М0 минус шарнирный штраф за наложение на тяжёлые атомы кармана.

        value = score_M0 - weight * sum_ij max(0, r_min_ij - d_ij),
        r_min_ij = CLASH_VDW_FRACTION * (vdw_i + vdw_j)

    Устройство штрафа взято у `delta_SC` из BInD: величина шарнирная — ноль, пока
    пара не ближе порога, и растёт линейно за порогом. Именно этим она отличается
    от нашего прежнего правила, где клэш был фильтром шага: фильтр говорит «нельзя»,
    но не говорит «в какую сторону», а шарнирный штраф задаёт направление и потому
    способен вытаскивать позу из наложения, а не только запрещать в него входить.

    Порог зависит от пары элементов, а не единый для всех: у пары C-C сумма
    ван-дер-ваальсовых радиусов 3.40 A, у пары C-H — 2.90 A.

    Штраф считается по **тяжёлым** атомам обеих сторон. Водороды опущены не ради
    скорости: их положение у лиганда задано достройкой, а не измерением, и штрафовать
    за наложение достроенного водорода значило бы штрафовать за нашу же процедуру.
    """

    base: DiscretePotential
    pocket_atoms: np.ndarray
    pocket_radii: np.ndarray
    weight: float = CLASH_PENALTY_WEIGHT

    @property
    def name(self) -> str:
        return "clash"

    def clash_penalty(self, mol: Chem.Mol) -> float:
        """Суммарное нарушение ван-дер-ваальсовых границ в ангстремах; 0.0 — нет наложений."""
        conf = mol.GetConformer()
        координаты = []
        радиусы = []
        таблица = Chem.GetPeriodicTable()
        for атом in mol.GetAtoms():
            if атом.GetAtomicNum() == 1:
                continue
            точка = conf.GetAtomPosition(атом.GetIdx())
            координаты.append([точка.x, точка.y, точка.z])
            радиусы.append(таблица.GetRvdw(атом.GetAtomicNum()))
        лиганд = np.asarray(координаты, dtype=float)
        r_лиганда = np.asarray(радиусы, dtype=float)

        расстояния = np.linalg.norm(
            лиганд[:, None, :] - self.pocket_atoms[None, :, :], axis=2
        )
        пороги = CLASH_VDW_FRACTION * (r_лиганда[:, None] + self.pocket_radii[None, :])
        return float(np.maximum(0.0, пороги - расстояния).sum())

    def __call__(self, mol: Chem.Mol, ifp: np.ndarray) -> float:
        return self.base(mol, ifp) - self.weight * self.clash_penalty(mol)


def make_potential(
    variant: str,
    reference_ifp: np.ndarray,
    *,
    scoring_types: Sequence[str] = SCORING_INTERACTION_TYPES,
    pocket_atoms: np.ndarray | None = None,
    pocket_radii: np.ndarray | None = None,
    clash_weight: float = CLASH_PENALTY_WEIGHT,
    weights_csv: Path = KLIFS_IFP_CSV,
) -> Potential:
    """Собирает потенциал по имени варианта: `discrete`, `clash` или `weighted`.

    Отказывается собирать `clash` без координат и радиусов кармана: молчаливое
    построение дискретного потенциала вместо запрошенного дало бы таблицу с меткой
    «clash», посчитанную базовой формулой, и подменённый вариант был бы неотличим
    от настоящего.
    """
    if variant not in POTENTIAL_VARIANTS:
        raise ValueError(
            f"неизвестный вариант потенциала {variant!r}; допустимы {list(POTENTIAL_VARIANTS)}"
        )
    база = DiscretePotential(reference_ifp=reference_ifp, scoring_types=scoring_types)
    if variant == "discrete":
        return база
    if variant == "weighted":
        return WeightedPotential(
            reference_ifp=reference_ifp,
            weights=cached_bit_weights(weights_csv),
            scoring_types=scoring_types,
        )
    if pocket_atoms is None or pocket_radii is None:
        raise ValueError(
            "варианту 'clash' нужны координаты и радиусы тяжёлых атомов кармана: "
            "зови pose_optimization.pocket_heavy_atoms и pocket_vdw_radii"
        )
    return ClashPenaltyPotential(
        base=база,
        pocket_atoms=pocket_atoms,
        pocket_radii=pocket_radii,
        weight=clash_weight,
    )
