"""Мягкий дифференцируемый потенциал взаимодействий на torch.

Зачем сглаживание. Жёсткий скор кусочно-постоянен: бит взаимодействия ставится
по порогу, поэтому производная равна нулю всюду, где определена, и градиентом такую
величину не оптимизируешь (`docs/faq.md`, «почему потенциал должен быть гладким»).
Здесь каждое пороговое условие правил KLIFS заменено сигмоидой, а логическое «и» —
произведением: значение остаётся в диапазоне [0, 1] и совпадает с жёстким битом
вдали от порога, но у него есть ненулевой градиент в окрестности порога.

Правила — те же, что у жёсткого расчёта, прочитанные в исходнике FingerPrintLib
(`docs/klifs-rules-source.md`, раздел 2.3), и пороги берутся из тех же констант
`KLIFS_RULE_*`. Это важно: мягкий потенциал должен быть сглаженной **той же** мерой,
а не похожей — иначе сравнение «жёсткий против мягкого» сравнивало бы два разных
определения взаимодействия.

Ширина сглаживания `tau` — единственный свободный параметр. По расстоянию она задана
в ангстремах, по углу — в градусах, и значения по умолчанию выбраны так, чтобы переход
от 1 к 0 укладывался примерно в 0.5 Å и 20°: это меньше разброса, который дают сами
позы, и больше точности координат. Значение не подгонялось под результат — влияние
ширины меряется отдельно, как свип.

Флаги атомов здесь не вычисляются: сторона берётся готовой из `ligand_flags`
и `klifs_rules`, то есть химия у жёсткого и мягкого потенциала общая, а различается
только геометрия — ступенька против сигмоиды.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from kinase_ifp.config import (
    KLIFS_RULE_AROMATIC_DISTANCE,
    KLIFS_RULE_AROMATIC_FACE_ANGLE_DEG,
    KLIFS_RULE_HBOND_DISTANCE,
    KLIFS_RULE_HBOND_MAX_DEVIATION_DEG,
    KLIFS_RULE_HYDROPHOBE_DISTANCE,
    KLIFS_RULE_IONIC_DISTANCE,
)

# Ширина сглаживания по расстоянию, ангстремы: переход 1 -> 0 укладывается примерно
# в 0.5 Å вокруг порога. Меньше разброса поз, больше точности координат.
TAU_DISTANCE_A: Final[float] = 0.12

# Ширина сглаживания по углу, градусы: переход укладывается примерно в 20°.
TAU_ANGLE_DEG: Final[float] = 5.0

# Ниже этого значения расстояние считается вырожденным: два атома в одной точке.
# Нужно только чтобы не делить на ноль при нормировке векторов.
_EPS: Final[float] = 1e-8


class SoftPotentialError(RuntimeError):
    """Потенциал посчитать нельзя: несовместимые формы или пустая сторона."""


def soft_step_below(
    value: torch.Tensor, threshold: float, tau: float = TAU_DISTANCE_A
) -> torch.Tensor:
    """Гладкая замена условию `value < threshold`: сигмоида, падающая на пороге.

    Возвращает величину в (0, 1): около 1 при `value` заметно меньше порога, около 0
    при заметно большем. Производная максимальна ровно на пороге — именно она
    и сообщает атому, в какую сторону двигаться.
    """
    return torch.sigmoid((threshold - value) / tau)


def soft_contact(
    left: torch.Tensor, right: torch.Tensor, threshold: float, tau: float = TAU_DISTANCE_A
) -> torch.Tensor:
    """Мягкий признак «есть пара точек ближе порога» для двух наборов координат.

    `left` формы (N, 3), `right` формы (M, 3). Жёсткое правило (`_есть_пара`) — это
    «хотя бы одна пара», то есть максимум по парам. Максимум сам по себе
    дифференцируем, но градиент получает только одна пара; поэтому берётся мягкий
    максимум через `logsumexp` с той же шириной: сигнал получают все пары вблизи
    порога, и он не переключается скачком при смене ближайшей пары.
    """
    if left.numel() == 0 or right.numel() == 0:
        return torch.zeros((), dtype=torch.float64)
    расстояния = torch.cdist(left, right)
    признаки = soft_step_below(расстояния, threshold, tau)
    return _мягкий_максимум(признаки.flatten(), tau)


def _мягкий_максимум(значения: torch.Tensor, tau: float) -> torch.Tensor:
    """Мягкий максимум набора: logsumexp с масштабом `tau`, зажатый в [0, 1]."""
    масштаб = max(tau, _EPS)
    return torch.clamp(масштаб * torch.logsumexp(значения / масштаб, dim=0), max=1.0)


def soft_hbond(
    donor_heavy: torch.Tensor,
    donor_h: torch.Tensor,
    acceptors: torch.Tensor,
    *,
    tau_distance: float = TAU_DISTANCE_A,
    tau_angle: float = TAU_ANGLE_DEG,
) -> torch.Tensor:
    """Мягкий признак водородной связи по правилу KLIFS.

    Правило оригинала: расстояние от **тяжёлого** донора до акцептора меньше 3.5 Å
    и угол между направлением D–H и направлением D→A меньше 45°. Оба условия
    сглаживаются и перемножаются — произведение и есть мягкое «и».

    `donor_heavy` формы (N, 3), `donor_h` формы (N, 3) — водород, сидящий на том же
    доноре, `acceptors` формы (M, 3).
    """
    if donor_heavy.numel() == 0 or acceptors.numel() == 0:
        return torch.zeros((), dtype=torch.float64)
    if donor_heavy.shape != donor_h.shape:
        raise SoftPotentialError(
            f"донорам {tuple(donor_heavy.shape)} не соответствуют водороды "
            f"{tuple(donor_h.shape)}: правило требует пару «тяжёлый атом — водород»"
        )

    к_акцептору = acceptors[None, :, :] - donor_heavy[:, None, :]
    расстояния = torch.linalg.vector_norm(к_акцептору, dim=2)
    близко = soft_step_below(расстояния, KLIFS_RULE_HBOND_DISTANCE, tau_distance)

    направление_h = donor_h - donor_heavy
    направление_h = направление_h / (
        torch.linalg.vector_norm(направление_h, dim=1, keepdim=True) + _EPS
    )
    направление_a = к_акцептору / (расстояния[:, :, None] + _EPS)
    косинус = torch.clamp((направление_h[:, None, :] * направление_a).sum(dim=2), -1.0, 1.0)
    угол = torch.rad2deg(torch.arccos(косинус))
    по_углу = soft_step_below(угол, KLIFS_RULE_HBOND_MAX_DEVIATION_DEG, tau_angle)

    return _мягкий_максимум((близко * по_углу).flatten(), tau_distance)


@dataclass(frozen=True)
class SoftSide:
    """Одна сторона взаимодействия в виде тензоров: то же, что `klifs_rules.Groups`.

    Отдельная запись, а не переиспользование `Groups`: там кортежи кортежей питона,
    здесь нужны тензоры, по которым идёт градиент. Преобразование — `side_from_groups`
    в `bridge.py`, чтобы этот модуль не зависел ни от RDKit, ни от mol2.
    """

    hydrophobes: torch.Tensor
    ring_centres: torch.Tensor
    ring_normals: torch.Tensor
    donor_heavy: torch.Tensor
    donor_h: torch.Tensor
    acceptors: torch.Tensor
    cations: torch.Tensor
    anions: torch.Tensor


def soft_aromatic(
    left: SoftSide, right: SoftSide, *, face: bool, tau_distance: float = TAU_DISTANCE_A
) -> torch.Tensor:
    """Мягкие биты `F-F` (лицом к лицу) и `F-E` (ребром к плоскости).

    Правило: расстояние между ароматическими **атомами** меньше 4.0 Å и острый угол
    между нормалями либо меньше 30° (лицом), либо больше 30° (ребром). Второй случай
    в оригинале записан как отклонение от прямого угла меньше 60°, что на остром угле
    равносильно границе 30° с другой стороны.
    """
    if left.ring_centres.numel() == 0 or right.ring_centres.numel() == 0:
        return torch.zeros((), dtype=torch.float64)

    расстояния = torch.cdist(left.ring_centres, right.ring_centres)
    близко = soft_step_below(расстояния, KLIFS_RULE_AROMATIC_DISTANCE, tau_distance)

    косинус = torch.clamp(
        torch.abs(left.ring_normals @ right.ring_normals.T), -1.0, 1.0
    )
    угол = torch.rad2deg(torch.arccos(косинус))
    по_углу = (
        soft_step_below(угол, KLIFS_RULE_AROMATIC_FACE_ANGLE_DEG, TAU_ANGLE_DEG)
        if face
        else soft_step_below(-угол, -KLIFS_RULE_AROMATIC_FACE_ANGLE_DEG, TAU_ANGLE_DEG)
    )
    return _мягкий_максимум((близко * по_углу).flatten(), tau_distance)


def soft_bits(residue: SoftSide, ligand: SoftSide) -> torch.Tensor:
    """Семь мягких бит одной позиции кармана в порядке `KLIFS_INTERACTION_TYPES`.

    Возвращает тензор формы (7,) со значениями в [0, 1]. Порядок и смысл бит те же,
    что у `klifs_rules.bits_for_residue`: `a` — атом кармана, `b` — атом лиганда,
    и направленные биты не симметричны.
    """
    return torch.stack(
        [
            soft_contact(residue.hydrophobes, ligand.hydrophobes, KLIFS_RULE_HYDROPHOBE_DISTANCE),
            soft_aromatic(residue, ligand, face=True),
            soft_aromatic(residue, ligand, face=False),
            soft_hbond(residue.donor_heavy, residue.donor_h, ligand.acceptors),
            soft_hbond(ligand.donor_heavy, ligand.donor_h, residue.acceptors),
            soft_contact(residue.cations, ligand.anions, KLIFS_RULE_IONIC_DISTANCE),
            soft_contact(residue.anions, ligand.cations, KLIFS_RULE_IONIC_DISTANCE),
        ]
    )
