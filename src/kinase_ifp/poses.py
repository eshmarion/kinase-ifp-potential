"""Набор поз из кристаллического лиганда.

Из одного кристаллического лиганда делается набор: нативная поза плюс искусственно
испорченные («декои»). По нему меряются ROC-AUC, top-1 и монотонность скора по RMSD
 — то есть проверяется, отличает ли потенциал верную позу от неверной.

Своей подготовки лиганда здесь нет: формат папки прогона требует, чтобы локальные позы
и молекулы из DiffSBDD проходили одну и ту же точку входа `protonate.prepare_ligand` —
заряды при pH 7.4, а затем явные водороды. Иначе сравнение условий превратится
в сравнение двух протоколов подготовки.

Запись файлов прогона — не здесь, а в `experiments.run_io`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolAlign
from rdkit.Geometry import Point3D

from kinase_ifp.config import (
    POSE_CONFORMER_SHARE,
    POSE_DIRECTION_ATTEMPTS,
    POSE_MAX_CENTROID_SHIFT_A,
    POSE_RMSD_RANGE_A,
    POSE_RMSD_TOLERANCE_A,
)
from kinase_ifp.molecule_io import read_mol
from kinase_ifp.protonate import prepare_ligand


class PoseGenerationError(RuntimeError):
    """Позу не удалось построить."""


def heavy_atom_centroid(mol: Chem.Mol) -> np.ndarray:
    """Геометрический центр тяжёлых атомов позы."""
    conf = mol.GetConformer()
    points = [
        conf.GetAtomPosition(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1
    ]
    centroid: np.ndarray = np.array([[p.x, p.y, p.z] for p in points], dtype=float).mean(axis=0)
    return centroid


def heavy_atom_rmsd(probe: Chem.Mol, reference: Chem.Mol) -> float:
    """RMSD между позами по тяжёлым атомам, без выравнивания.

    Принимает две конформации одной молекулы, возвращает отклонение в ангстремах.

    Считает `rdMolAlign.CalcRMS`, а не `GetBestRMS`: второй сначала совмещает молекулы
    и лишь потом меряет отклонение, то есть на наборе поз обнулил бы ровно ту величину,
    которую мы измеряем — смещение относительно кристалла. `CalcRMS` меряет «на месте»
    и при этом учитывает симметрию молекулы (поворот фенила не считается смещением).

    Водороды отбрасываются: граница «поза воспроизведена верно» в 2 Å определена
    в докинге по тяжёлым атомам.
    """
    return float(rdMolAlign.CalcRMS(Chem.RemoveHs(probe), Chem.RemoveHs(reference)))


def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    """Равномерно распределённое направление на сфере."""
    vector = rng.normal(size=3)
    return vector / np.linalg.norm(vector)


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Матрица поворота вокруг оси на угол (формула Родрига)."""
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation: np.ndarray = (
        np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    )
    return rotation


def apply_rigid(
    mol: Chem.Mol, rotation: np.ndarray, translation: np.ndarray, pivot: np.ndarray
) -> Chem.Mol:
    """Копия молекулы, повёрнутая вокруг точки и сдвинутая как жёсткое тело."""
    moved = Chem.Mol(mol)
    conf = moved.GetConformer()
    for i in range(moved.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        position = rotation @ (np.array([p.x, p.y, p.z]) - pivot) + pivot + translation
        conf.SetAtomPosition(i, Point3D(*(float(c) for c in position)))
    return moved


# Шагов двоичного поиска амплитуды. 40 шагов сужают отрезок [0, 1] далеко за пределы
# точности координат, то есть попадание в допуск ограничено формой молекулы, а не поиском.
_BISECTION_STEPS: Final[int] = 40


@dataclass(frozen=True)
class _Direction:
    """Направление возмущения: ось поворота, направление сдвига и точка поворота.

    Отдельный объект, а не замыкание внутри цикла: замыкание над переменной цикла
    ловится линтером (B023) и в самом деле опасно, если позу отложить на потом.
    """

    axis: np.ndarray
    shift: np.ndarray
    pivot: np.ndarray

    def pose(self, mol: Chem.Mol, amplitude: float) -> Chem.Mol:
        """Поза при заданной амплитуде: 0 — исходная, 1 — разворот на 180° и предельный сдвиг."""
        return apply_rigid(
            mol,
            rotation_matrix(self.axis, amplitude * np.pi),
            amplitude * POSE_MAX_CENTROID_SHIFT_A * self.shift,
            self.pivot,
        )


def perturb_to_target_rmsd(
    mol: Chem.Mol,
    reference: Chem.Mol,
    target_rmsd: float,
    rng: np.random.Generator,
) -> Chem.Mol:
    """Двигает позу как жёсткое тело до заданного RMSD относительно эталона.

    Принимает молекулу с конформацией, эталон для измерения, целевой RMSD в ангстремах
    и генератор случайных чисел; возвращает новую позу. Исходная молекула не меняется.

    Поворот идёт вокруг центра тяжёлых атомов, поэтому сдвиг центра равен длине вектора
    переноса и ограничен `POSE_MAX_CENTROID_SHIFT_A`: декой обязан остаться в кармане
    и контактировать с белком неправильно, а не отсутствовать в нём вовсе.

    Амплитуда подбирается двоичным поиском, а не считается по формуле: связь «угол
    поворота — RMSD» зависит от оси и формы молекулы. Поиск корректен, потому что при
    фиксированном направлении RMSD растёт по амплитуде монотонно — вклад поворота
    и вклад переноса складываются независимо (перекрёстный член обращается в ноль,
    так как поворот идёт вокруг самого центра).

    Поднимает `PoseGenerationError`, если целевой RMSD недостижим: у вытянутой молекулы
    поворот вокруг длинной оси смещает атомы слабо, и цель в 5 Å по такой оси не берётся
    даже полным разворотом. Тихо вернуть позу с другим RMSD нельзя — лестница
    искажений перестала бы быть лестницей.
    """
    if target_rmsd <= 0.0:
        return Chem.Mol(mol)

    pivot = heavy_atom_centroid(mol)

    for _ in range(POSE_DIRECTION_ATTEMPTS):
        direction = _Direction(
            axis=_random_unit_vector(rng), shift=_random_unit_vector(rng), pivot=pivot
        )

        if heavy_atom_rmsd(direction.pose(mol, 1.0), reference) < target_rmsd:
            continue

        low, high = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            middle = 0.5 * (low + high)
            if heavy_atom_rmsd(direction.pose(mol, middle), reference) < target_rmsd:
                low = middle
            else:
                high = middle

        result = direction.pose(mol, high)
        if abs(heavy_atom_rmsd(result, reference) - target_rmsd) <= POSE_RMSD_TOLERANCE_A:
            return result

    raise PoseGenerationError(
        f"целевой RMSD {target_rmsd:.2f} A недостижим за {POSE_DIRECTION_ATTEMPTS} "
        f"попыток при пределе сдвига центра {POSE_MAX_CENTROID_SHIFT_A} A"
    )


POSE_KIND_NATIVE: Final[str] = "native"
POSE_KIND_RIGID: Final[str] = "rigid"
POSE_KIND_CONFORMER: Final[str] = "conformer"

# Сколько конформеров держать в запасе. Нижние ступени лестницы берут только те, что
# сами по себе близки к кристаллу: у лиганда 6tgu (4 вращаемые связи) внутренний RMSD
# конформеров ETKDG расходится от 0.32 до 2.58 A, и ближе 0.5 A оказываются единицы.
_CONFORMER_POOL: Final[int] = 100


@dataclass(frozen=True)
class Pose:
    """Одна поза набора: молекула, её отклонение от кристалла и способ построения."""

    mol: Chem.Mol
    rmsd_to_ref: float
    kind: str


@dataclass(frozen=True)
class PoseFailure:
    """Поза, которую построить не удалось. Молча из набора не исчезает ничто."""

    index: int
    stage: str
    reason: str


@dataclass(frozen=True)
class PoseSet:
    """Результат генерации: построенные позы и причины по каждой непостроенной."""

    poses: tuple[Pose, ...]
    failures: tuple[PoseFailure, ...]


def rmsd_ladder(n: int) -> tuple[float, ...]:
    """Целевые RMSD для `n` поз: нативная (0) плюс равномерная лестница искажений.

    Равномерная, а не два кластера: коэффициенту Спирмена нужна непрерывная
    шкала — по набору из «почти нативных» и «явных декоев» он лишь повторяет ROC-AUC.
    """
    if n < 2:
        raise ValueError(f"в наборе должно быть не меньше двух поз, запрошено {n}")
    low, high = POSE_RMSD_RANGE_A
    return (0.0, *(float(v) for v in np.linspace(low, high, n - 1)))


def _aligned_conformers(reference: Chem.Mol, seed: int) -> list[tuple[float, Chem.Mol]]:
    """Конформеры RDKit, совмещённые с кристаллической позой, с их внутренним RMSD.

    После совмещения остаточный RMSD — это отличие торсий, а не размещения: именно
    оно задаёт нижнюю границу, ниже которой такой конформер на лестницу не поставить.
    """
    pool = Chem.Mol(reference)
    pool.RemoveAllConformers()
    pool.AddConformer(reference.GetConformer(), assignId=True)
    # rdDistGeom, а не AllChem: там она и определена, а реэкспорт через AllChem
    # не виден стабам rdkit (mypy: «Module has no attribute»).
    conformer_ids = rdDistGeom.EmbedMultipleConfs(
        pool, numConfs=_CONFORMER_POOL, randomSeed=seed
    )

    conformers = []
    for conformer_id in conformer_ids:
        candidate = Chem.Mol(pool, confId=conformer_id)
        rdMolAlign.AlignMol(candidate, reference)
        conformers.append((heavy_atom_rmsd(candidate, reference), candidate))
    return conformers


def generate_poses(ligand_path: Path, n: int, seed: int) -> PoseSet:
    """Строит набор поз из кристаллического лиганда мишени.

    Принимает путь к SDF лиганда из пакета мишени, размер набора и сид; возвращает
    построенные позы вместе с причинами по каждой непостроенной.

    Первая поза — нативная, остальные равномерно разложены по RMSD. Половина искажённых
    поз строится на других конформерах RDKit, половина — жёсткими сдвигами и поворотами
    кристаллического конформера: отпечаток обязан отличать и неверное размещение,
    и неверную внутреннюю геометрию.

    Подготовка идёт общей функцией `protonate.prepare_ligand` (заряды, затем водороды) —
    по формату папки прогона своей подготовки у производителя SDF быть не должно, иначе локальные
    позы и молекулы DiffSBDD станут несравнимы. Готовится один раз, эталон: все позы
    набора — конформации этой же молекулы, и повторять подготовку по каждой незачем.
    """
    # `read_mol` вместо `MolFromMolFile`: тот берёт путь строкой и передаёт его
    # в C++ через ANSI, а на Windows кириллица в пути даёт «Bad input file».
    raw = read_mol(ligand_path)
    if raw is None:
        raise PoseGenerationError(f"RDKit не разобрал лиганд {ligand_path}")
    reference = prepare_ligand(raw)

    rng = np.random.default_rng(seed)
    conformers = _aligned_conformers(reference, seed)

    poses: list[Pose] = [Pose(mol=Chem.Mol(reference), rmsd_to_ref=0.0, kind=POSE_KIND_NATIVE)]
    failures: list[PoseFailure] = []

    for index, target in enumerate(rmsd_ladder(n)[1:], start=1):
        # Чередование, а не «первая половина такая, вторая сякая»: иначе способ
        # построения совпал бы с величиной искажения, и нельзя было бы отличить
        # чувствительность к размещению от чувствительности к торсиям.
        подходящие = [mol for internal, mol in conformers if internal < target]
        # Конформер годится ступени, только если его собственное отличие торсий меньше
        # целевого RMSD: ниже своего внутреннего отклонения такую позу не опустить.
        # На нижних ступенях подходящих не оказывается (у 6tgu ближе 0.5 A — единицы
        # из ста), и ступень строится жёстким возмущением. Терять её нельзя: величину
        # искажения задаёт лестница, а не способ построения. Что именно вышло, видно
        # в поле `kind`, поэтому подмена не молчаливая.
        берём_конформер = index % 2 == 1 and POSE_CONFORMER_SHARE > 0.0 and bool(подходящие)

        if берём_конформер:
            начальная = подходящие[int(rng.integers(len(подходящие)))]
            kind = POSE_KIND_CONFORMER
        else:
            начальная = reference
            kind = POSE_KIND_RIGID

        try:
            mol = perturb_to_target_rmsd(начальная, reference, float(target), rng)
        except PoseGenerationError as ошибка:
            failures.append(
                PoseFailure(
                    index=index,
                    stage="perturb",
                    reason=f"rmsd_target_unreachable: {ошибка}",
                )
            )
            continue

        poses.append(Pose(mol=mol, rmsd_to_ref=heavy_atom_rmsd(mol, reference), kind=kind))

    return PoseSet(poses=tuple(poses), failures=tuple(failures))
