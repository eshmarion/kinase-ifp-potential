"""Мост между химией правил KLIFS и мягким потенциалом на torch.

Здесь и только здесь сходятся две стороны: `klifs_rules.Groups` (флаги атомов,
посчитанные по правилам оригинала из mol2 или из RDKit) и `terms.SoftSide` (те же
атомы тензорами, по которым идёт градиент). Разделение нужно, чтобы `terms.py`
не зависел ни от RDKit, ни от mol2 и оставался проверяемым на синтетических входах.

Потенциал определён так же, как жёсткий скор: доля ключевых взаимодействий эталона,
воспроизведённых молекулой, по типам `SCORING_INTERACTION_TYPES` (все, кроме
гидрофобных). Отличие одно — вместо логического «есть бит» стоит мягкое значение
в [0, 1]. Поэтому величины сравнимы напрямую: у позы, воспроизводящей все ключевые
взаимодействия с запасом, мягкий скор стремится к 1, как и жёсткий.

Энергия — минус скор: `guidance` двигает координаты против градиента энергии,
то есть в сторону роста воспроизведения взаимодействий.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kinase_ifp.config import (
    KLIFS_INTERACTION_TYPES,
    N_KLIFS_POSITIONS,
    SCORING_INTERACTION_TYPES,
)
from kinase_ifp.fingerprint_klifs import PocketRules
from kinase_ifp.klifs_rules import Groups
from soft_potential.terms import SoftPotentialError, SoftSide, soft_bits

_ТИП_ТЕНЗОРА = torch.float64


def _тензор(точки: tuple, требует_градиент: bool = False) -> torch.Tensor:
    if not точки:
        return torch.zeros((0, 3), dtype=_ТИП_ТЕНЗОРА, requires_grad=требует_градиент)
    массив = np.asarray(точки, dtype=float)
    return torch.tensor(массив, dtype=_ТИП_ТЕНЗОРА, requires_grad=требует_градиент)


def side_from_groups(groups: Groups) -> SoftSide:
    """Переводит флаги атомов правил KLIFS в тензоры мягкого потенциала.

    Доноры разворачиваются в пары «тяжёлый атом — водород»: правило требует
    направление D–H, а на одном доноре водородов может быть несколько.
    """
    тяжёлые: list[tuple[float, float, float]] = []
    водороды: list[tuple[float, float, float]] = []
    for донор in groups.donors:
        for водород in донор.hydrogens:
            тяжёлые.append(донор.heavy)
            водороды.append(водород)

    return SoftSide(
        hydrophobes=_тензор(groups.hydrophobes),
        ring_centres=_тензор(tuple(кольцо.centroid for кольцо in groups.rings)),
        ring_normals=_тензор(tuple(кольцо.normal for кольцо in groups.rings)),
        donor_heavy=_тензор(tuple(тяжёлые)),
        donor_h=_тензор(tuple(водороды)),
        acceptors=_тензор(groups.acceptors),
        cations=_тензор(groups.cations),
        anions=_тензор(groups.anions),
    )


@dataclass(frozen=True)
class SoftTarget:
    """Мишень для мягкого потенциала: карман тензорами плюс ключевые биты эталона.

    `reference_bits` — список пар «позиция кармана, номер типа» по тем битам эталона,
    которые входят в скор. Их число и есть знаменатель: скор считается как доля
    воспроизведённых, ровно как в `scoring.ifp_score`.
    """

    residues: dict[str, SoftSide]
    position_to_residue: dict[int, str]
    reference_bits: tuple[tuple[int, int], ...]


def soft_target(pocket: PocketRules, reference_ifp: np.ndarray) -> SoftTarget:
    """Готовит мишень: переводит карман в тензоры и отбирает ключевые биты эталона.

    Принимает разобранный карман и эталонный отпечаток формы (7, 85); возвращает
    `SoftTarget`. Поднимает `SoftPotentialError`, если у эталона нет ни одного бита
    по типам скора: воспроизводить нечего, и потенциал был бы тождественным нулём.
    """
    номера_типов = [KLIFS_INTERACTION_TYPES.index(тип) for тип in SCORING_INTERACTION_TYPES]
    биты = tuple(
        (позиция + 1, тип)
        for тип in номера_типов
        for позиция in range(N_KLIFS_POSITIONS)
        if reference_ifp[тип, позиция]
    )
    if not биты:
        raise SoftPotentialError(
            "у эталона нет бит по типам скора: мягкий потенциал был бы нулём всюду"
        )
    return SoftTarget(
        residues={имя: side_from_groups(группы) for имя, группы in pocket.residues.items()},
        position_to_residue=dict(pocket.position_to_residue),
        reference_bits=биты,
    )


def soft_score(ligand: SoftSide, target: SoftTarget) -> torch.Tensor:
    """Мягкий скор позы: доля ключевых взаимодействий эталона, воспроизведённых мягко.

    Возвращает скаляр-тензор в [0, 1], по которому можно взять градиент. Считаются
    только те позиции, где у эталона есть ключевой бит: остальные в скор не входят
    вовсе, и тратить на них расчёт незачем.
    """
    нужные = {позиция for позиция, _ in target.reference_bits}
    биты_позиции: dict[int, torch.Tensor] = {}
    for позиция in нужные:
        остаток = target.position_to_residue.get(позиция)
        сторона = target.residues.get(остаток) if остаток else None
        if сторона is None:
            continue
        биты_позиции[позиция] = soft_bits(сторона, ligand)

    вклады = [
        биты_позиции[позиция][тип]
        for позиция, тип in target.reference_bits
        if позиция in биты_позиции
    ]
    if not вклады:
        return torch.zeros((), dtype=_ТИП_ТЕНЗОРА)
    return torch.stack(вклады).sum() / float(len(target.reference_bits))


def soft_energy(ligand: SoftSide, target: SoftTarget) -> torch.Tensor:
    """Энергия мягкого потенциала: минус мягкий скор.

    Именно эту величину дифференцируют: `guidance` сдвигает координаты против
    градиента энергии, то есть в сторону воспроизведения взаимодействий эталона.
    """
    return -soft_score(ligand, target)
