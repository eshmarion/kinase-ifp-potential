"""Расчёт позиционного отпечатка взаимодействий белок–лиганд.

Отпечаток — массив (7 типов, 85 позиций KLIFS). Позиционность здесь главное:
свёртка контактов в «сколько всего водородных связей» фингерпринтом не является,
и ровно на этом сломался черновик (`source_files/мяу.py:2496-2528`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import prolif
from rdkit import Chem

from kinase_ifp.config import (
    KLIFS_BITS_SHAPE,
    KLIFS_IFP_LENGTH,
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    KLIFS_TO_PROLIF,
    KLIFS_TO_PROLIF_IMPLICIT_H,
    PROLIF_PARAMETERS,
)
from kinase_ifp.pocket import IMPLICIT_PROTONATION, load_protein
from kinase_ifp.protonate import UNNORMALIZED_ACID, has_explicit_hydrogens

if TYPE_CHECKING:
    from kinase_ifp.pocket import Pocket

# Значения поля `protonation` формата пакета мишени и соответствующие им наборы типов ProLIF.
# Ключи — те же строки, что перечисляет klifs.PROTONATION_MODES.
_MAPPING_BY_PROTONATION: dict[str, dict[str, str]] = {
    "explicit": KLIFS_TO_PROLIF,
    IMPLICIT_PROTONATION: KLIFS_TO_PROLIF_IMPLICIT_H,
}


class IfpError(RuntimeError):
    """Отпечаток для этой молекулы посчитать нельзя.

    Отдельный тип, а не голый ValueError: производителям SDF нужно
    отличать «эта молекула не годится, записать её в failures.csv» от программной ошибки.
    """


def compute_ifp(pocket: Pocket, ligand: Chem.Mol) -> np.ndarray:
    """Считает отпечаток взаимодействий лиганда с карманом.

    Возвращает массив формы `KLIFS_IFP_SHAPE` = (7, 85), dtype `uint8`: строка — тип
    взаимодействия в порядке `KLIFS_INTERACTION_TYPES`, столбец — каноническая позиция
    KLIFS 1..85.

    Лиганд обязан прийти подготовленным: с приведёнными зарядами и явными водородами.
    Без водородов ProLIF не отличает донор водородной связи от акцептора. Подготовка
    здесь не выполняется намеренно — расчёт, молча достраивающий входные данные,
    скрывает ровно тот дефект, из-за которого черновик считал неверный отпечаток.
    Готовит производитель SDF общей функцией `protonate.prepare_ligand`.

    Поднимает `IfpError`, если у лиганда нет водородов или конформации, если режим
    протонирования в пакете мишени неизвестен или если ProLIF нашёл взаимодействие
    с остатком, которого нет в карте позиций кармана.
    """
    if ligand.GetNumConformers() == 0:
        raise IfpError("У лиганда нет конформации: взаимодействия считать не по чему")
    if not has_explicit_hydrogens(ligand):
        raise IfpError(
            "У лиганда нет явных водородов: ProLIF не отличит донор водородной связи "
            "от акцептора. Готовьте молекулу protonate.prepare_ligand на стороне "
            "производителя SDF (формат папки прогона)"
        )

    # Нейтральная кислотная группа при pH 7.4 физически невозможна: если она здесь,
    # молекулу готовили в обход `prepare_ligand`. Пропустить это молча значило бы
    # потерять солевые мостики у одного источника молекул и сохранить у другого —
    # то самое расхождение, ради которого нормализация и заводилась (В-12).
    if ligand.HasSubstructMatch(UNNORMALIZED_ACID):
        raise IfpError(
            "У лиганда есть непротонированная кислотная группа: при pH 7.4 её быть "
            "не может. Готовьте молекулу protonate.prepare_ligand — она приводит "
            "заряды и только потом достраивает водороды (формат папки прогона)"
        )

    mapping = _MAPPING_BY_PROTONATION.get(pocket.protonation)
    if mapping is None:
        raise IfpError(
            f"Неизвестный режим протонирования {pocket.protonation!r} в пакете мишени "
            f"{pocket.pdb_id}; допустимы {sorted(_MAPPING_BY_PROTONATION)}"
        )

    protein = load_protein(pocket)
    interactions = list(mapping.values())
    fingerprint = prolif.Fingerprint(
        interactions=interactions,
        parameters={n: PROLIF_PARAMETERS[n] for n in interactions if n in PROLIF_PARAMETERS},
        implicit_hydrogens=pocket.protonation == IMPLICIT_PROTONATION,
    )

    # residues задаётся явным списком остатков кармана: считать по всему белку — ещё
    # один дефект черновика, а отбор по радиусу зависел бы от размера молекулы.
    # metadata=True обязателен: без него ProLIF возвращает массив булей без имён типов,
    # и разложить взаимодействия по типам KLIFS нечем.
    result = fingerprint.generate(
        prolif.Molecule.from_rdkit(ligand),
        protein,
        residues=list(pocket.residue_to_position),
        metadata=True,
    )

    ifp = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    for (_, protein_residue), found in result.items():
        position = pocket.residue_to_position.get(str(protein_residue))
        if position is None:
            raise IfpError(
                f"ProLIF нашёл взаимодействие с остатком {protein_residue}, которого нет "
                f"в карте позиций кармана {pocket.pdb_id}: карман и белок разъехались"
            )
        for klifs_type, prolif_name in mapping.items():
            if prolif_name in found:
                ifp[KLIFS_INTERACTION_TYPES.index(klifs_type), position - 1] = 1
    return ifp


def ifp_to_bits(ifp: np.ndarray) -> str:
    """Переводит отпечаток (7, 85) в 595-символьную строку в раскладке KLIFS."""
    if ifp.shape != KLIFS_IFP_SHAPE:
        raise IfpError(f"Ожидалась форма {KLIFS_IFP_SHAPE}, получена {ifp.shape}")
    # Транспонирование — не косметика: внутри проект держит (7 типов, 85 позиций),
    # а строка KLIFS уложена как (85 позиций, 7 типов). См. KLIFS_BITS_SHAPE.
    return "".join(str(int(bit)) for bit in ifp.T.reshape(-1))


def bits_to_ifp(bits: str) -> np.ndarray:
    """Разбирает 595-символьную строку KLIFS в массив (7, 85)."""
    if len(bits) != KLIFS_IFP_LENGTH:
        raise IfpError(
            f"Ожидалась строка длиной {KLIFS_IFP_LENGTH} символов, получена {len(bits)}"
        )
    if set(bits) - {"0", "1"}:
        raise IfpError("В строке отпечатка есть символы, кроме '0' и '1'")
    return np.array([int(c) for c in bits], dtype=np.uint8).reshape(KLIFS_BITS_SHAPE).T.copy()
