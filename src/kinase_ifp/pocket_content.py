"""Состав кармана исходного кристалла и соответствие условиям работы DiffSBDD.

Отвечает на вопрос, который по пакету KLIFS не задать: **что ещё занимало карман
в исходной структуре, кроме целевого лиганда.** KLIFS отдаёт белок уже очищенным —
в `protein.pdb` мишени 6tgu 5497 записей `ATOM` и ни одной `HETATM`, — поэтому ионы,
вода, криопротектор и вторая копия лиганда из него не видны вовсе. Источник здесь —
исходный файл PDB с RCSB.

**Что считается соответствием условиям DiffSBDD.** Не наше соображение, а четыре
условия, прочитанные в коде модели (`process_bindingmoad.py`, `utils.py`, коммит
5d0d38d):

1. карман модели состоит **только** из стандартных аминокислот в радиусе
   `DIFFSBDD_POCKET_CUTOFF_A` — `is_aa(resname, standard=True)`. Ион, кофактор
   и вторая копия лиганда в карман не попадают: модель их не видит ни при обучении,
   ни при генерации;
2. лиганд не связан с белком ковалентно — модель порождает свободную молекулу;
3. `QED >= DIFFSBDD_QED_THRESHOLD` — тот же порог, что при отборе обучающей выборки;
4. лиганд — одна связная молекула, а не несколько копий в одном поле.

**Вода намеренно не учитывается.** Карман модели её не содержит по построению, наш
расчёт IFP тоже (в ProLIF 2.2.1 водных мостиков нет среди 21 типа), а 95 % структур
PDB сняты при криогенной температуре, где сетка воды отражает условия съёмки. Считать
воду значило бы отсеять почти всё: в измерении на 61 структуре вода есть у 54.

Сеть нужна только двум функциям — `fetch_pdb` и `fetch_ligand_qed`; вся классификация
работает с уже прочитанным текстом и проверяется тестом без сети.
"""

from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from kinase_ifp.config import (
    COVALENT_BOND_MAX_A,
    CRYSTALLIZATION_ADDITIVE_CODES,
    DIFFSBDD_DATASET_CROSSDOCKED,
    DIFFSBDD_DATASET_MOAD,
    DIFFSBDD_DATASETS,
    DIFFSBDD_POCKET_CUTOFF_A,
    DIFFSBDD_QED_THRESHOLD,
    METAL_ION_CODES,
    NUCLEOTIDE_LIGAND_CODES,
    POCKET_CONTENT_ADDITIVE,
    POCKET_CONTENT_COFACTOR,
    POCKET_CONTENT_COVALENT,
    POCKET_CONTENT_LOW_QED,
    POCKET_CONTENT_METAL,
    POCKET_CONTENT_MULTI_LIGAND,
    POCKET_CONTENT_READY,
    WATER_CODES,
)

RCSB_STRUCTURE_URL: Final[str] = "https://files.rcsb.org/download/{pdb_id}.pdb"
RCSB_LIGAND_URL: Final[str] = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"

#: Сколько раз пробовать загрузку, прежде чем признать её несостоявшейся.
RCSB_ATTEMPTS: Final[int] = 2


class PocketContentError(RuntimeError):
    """Состав кармана установить нельзя — с указанием, чего именно не хватило."""


@dataclass(frozen=True)
class Atom:
    """Атом записи `ATOM` или `HETATM`: только то, что нужно для расстояний и кодов."""

    record: str
    resname: str
    chain: str
    resseq: str
    x: float
    y: float
    z: float
    element: str

    def distance_to(self, other: Atom) -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


@dataclass(frozen=True)
class Neighbour:
    """Гетерогруппа рядом с лигандом: код, цепь, номер и сколько атомов попало."""

    resname: str
    chain: str
    resseq: str
    atoms: int


@dataclass
class PocketVerdict:
    """Вердикт по одному комплексу."""

    pdb_id: str
    expo_id: str
    pocket_class: str
    matches_diffsbdd: bool
    reasons: tuple[str, ...]
    ligand_atoms: int
    ligand_copies: int
    radius: float
    qed: float | None
    water_atoms: int
    dataset: str = DIFFSBDD_DATASET_CROSSDOCKED
    neighbours: tuple[Neighbour, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """Одна строка для отчёта: класс, вердикт и причины."""
        знак = "да" if self.matches_diffsbdd else "нет"
        причины = "; ".join(self.reasons) if self.reasons else "—"
        return f"{self.pdb_id} {self.expo_id}: {self.pocket_class}, годен: {знак} ({причины})"


# ---------------------------------------------------------------------------
# Разбор файла PDB
# ---------------------------------------------------------------------------


def parse_atoms(text: str) -> list[Atom]:
    """Разбирает записи `ATOM` и `HETATM` первой модели файла PDB.

    Берётся именно первая модель: в файлах ЯМР их десятки, и считать соседей
    по всем сразу значило бы умножить каждую гетерогруппу на число моделей.
    Альтернативные положения (`altLoc`) берутся все — отбор выбирает
    модель A на уровне KLIFS, а здесь нужен полный список того, что в кармане есть.
    """
    atoms: list[Atom] = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            raise PocketContentError(f"Короткая запись PDB, координаты не читаются: {line!r}")
        atoms.append(
            Atom(
                record=line[:6].strip(),
                resname=line[17:20].strip(),
                chain=line[21],
                resseq=line[22:27].strip(),
                x=float(line[30:38]),
                y=float(line[38:46]),
                z=float(line[46:54]),
                element=line[76:78].strip().upper() if len(line) >= 78 else "",
            )
        )
    if not atoms:
        raise PocketContentError("В файле нет ни одной записи ATOM или HETATM")
    return atoms


def _residue_key(atom: Atom) -> tuple[str, str, str]:
    return (atom.resname, atom.chain, atom.resseq)


def ligand_copies(atoms: list[Atom], expo_id: str) -> list[list[Atom]]:
    """Все копии целевого лиганда в структуре, каждая — отдельным списком атомов.

    Копий бывает несколько: одна на цепь в кристалле с двумя молекулами в асимметричной
    единице. Это не «второй лиганд в кармане» — копии стоят в разных карманах, и такую
    структуру отбраковывать незачем. Вторая копия становится помехой, только если она
    попала в радиус первой, и проверяется это уже в `classify_pocket`.
    """
    код = expo_id.strip().upper()
    группы: dict[tuple[str, str, str], list[Atom]] = {}
    for atom in atoms:
        if atom.record == "HETATM" and atom.resname.upper() == код:
            группы.setdefault(_residue_key(atom), []).append(atom)
    return list(группы.values())


def neighbours_within(
    ligand: list[Atom], atoms: list[Atom], radius: float
) -> list[Neighbour]:
    """Гетерогруппы (кроме самого лиганда), у которых есть атом ближе `radius`.

    Считается по атомам, а не по центрам: остаток крупнее лиганда дотянулся бы
    до кармана краем и не попал бы в список при расчёте по центроидам.
    """
    свой = {_residue_key(atom) for atom in ligand}
    попавшие: dict[tuple[str, str, str], int] = {}
    for atom in atoms:
        if atom.record != "HETATM":
            continue
        ключ = _residue_key(atom)
        if ключ in свой:
            continue
        if any(atom.distance_to(точка) <= radius for точка in ligand):
            попавшие[ключ] = попавшие.get(ключ, 0) + 1
    return [
        Neighbour(resname=имя, chain=цепь, resseq=номер, atoms=число)
        for (имя, цепь, номер), число in sorted(попавшие.items())
    ]


def is_covalent(
    ligand: list[Atom], atoms: list[Atom], max_distance: float = COVALENT_BOND_MAX_A
) -> tuple[bool, float | None]:
    """Связан ли лиганд с белком ковалентно: есть ли пара тяжёлых атомов ближе предела.

    Возвращает признак и найденное минимальное расстояние. Водороды исключены с обеих
    сторон: связь X–H короче любого порога и дала бы ложное срабатывание на каждой
    структуре с явными водородами.
    """
    белок = [
        atom
        for atom in atoms
        if atom.record == "ATOM" and atom.element not in ("H", "D") and atom.element
    ]
    тяжёлые = [atom for atom in ligand if atom.element not in ("H", "D")]
    минимум: float | None = None
    for atom in тяжёлые:
        for сосед in белок:
            расстояние = atom.distance_to(сосед)
            if минимум is None or расстояние < минимум:
                минимум = расстояние
    return (минимум is not None and минимум <= max_distance), минимум


# ---------------------------------------------------------------------------
# Классификация
# ---------------------------------------------------------------------------


def classify_pocket(
    text: str,
    pdb_id: str,
    expo_id: str,
    *,
    qed: float | None = None,
    radius: float = DIFFSBDD_POCKET_CUTOFF_A,
    qed_threshold: float = DIFFSBDD_QED_THRESHOLD,
    dataset: str = DIFFSBDD_DATASET_CROSSDOCKED,
) -> PocketVerdict:
    """Классифицирует комплекс по четырём условиям DiffSBDD.

    Принимает текст исходного файла PDB, код структуры и код лиганда; `qed` считается
    отдельно (`fetch_ligand_qed`), потому что требует сети, а всё остальное — нет.
    `qed=None` означает «не измерен»: условие 3 тогда не проверяется, и это видно
    в причинах, а не подменяется молчаливым «прошёл».

    Возвращает `PocketVerdict`: класс из закрытого списка `POCKET_CONTENT_CLASSES`,
    признак соответствия и **все** причины несоответствия, а не первую найденную —
    структура, забракованная и по иону, и по QED, должна показывать оба.
    """
    atoms = parse_atoms(text)
    копии = ligand_copies(atoms, expo_id)
    if not копии:
        raise PocketContentError(
            f"Лиганд {expo_id} не найден среди записей HETATM структуры {pdb_id}"
        )

    целевой = max(копии, key=len)
    соседи = neighbours_within(целевой, atoms, radius)
    ковалентный, ближайший = is_covalent(целевой, atoms)

    вода = sum(с.atoms for с in соседи if с.resname.upper() in WATER_CODES)
    металлы = [с for с in соседи if с.resname.upper() in METAL_ION_CODES]
    добавки = [с for с in соседи if с.resname.upper() in CRYSTALLIZATION_ADDITIVE_CODES]
    кофакторы = [с for с in соседи if с.resname.upper() in NUCLEOTIDE_LIGAND_CODES]
    учтено = WATER_CODES | METAL_ION_CODES | CRYSTALLIZATION_ADDITIVE_CODES
    учтено = учтено | NUCLEOTIDE_LIGAND_CODES
    прочие = [с for с in соседи if с.resname.upper() not in учтено]

    причины: list[str] = []
    классы: list[str] = []
    if прочие:
        имена = ", ".join(sorted({с.resname for с in прочие}))
        причины.append(f"в карман попала посторонняя молекула: {имена}")
        классы.append(POCKET_CONTENT_MULTI_LIGAND)
    if кофакторы:
        имена = ", ".join(sorted({с.resname for с in кофакторы}))
        причины.append(f"кофактор или нуклеотид в кармане: {имена}")
        классы.append(POCKET_CONTENT_COFACTOR)
    if металлы:
        имена = ", ".join(sorted({с.resname for с in металлы}))
        причины.append(f"ион металла в кармане: {имена} (модель его не видит)")
        классы.append(POCKET_CONTENT_METAL)
    if добавки:
        имена = ", ".join(sorted({с.resname for с in добавки}))
        причины.append(f"добавка кристаллизации в кармане: {имена}")
        классы.append(POCKET_CONTENT_ADDITIVE)
    if ковалентный:
        расстояние = f"{ближайший:.2f}" if ближайший is not None else "?"
        причины.append(f"лиганд связан с белком ковалентно (ближайший контакт {расстояние} Å)")
        классы.append(POCKET_CONTENT_COVALENT)
    if dataset not in DIFFSBDD_DATASETS:
        raise PocketContentError(
            f"Неизвестный набор условий {dataset!r}; допустимы {DIFFSBDD_DATASETS}"
        )
    # Порог QED стоит только в ветке Binding MOAD. Наш чекпойнт из ветки CrossDocked,
    # где такого фильтра нет: применять его значило бы браковать комплексы по условию,
    # которого модель не знает.
    if dataset == DIFFSBDD_DATASET_MOAD:
        if qed is not None and qed < qed_threshold:
            причины.append(f"QED {qed:.3f} ниже порога обучающей выборки {qed_threshold}")
            классы.append(POCKET_CONTENT_LOW_QED)
        if qed is None:
            причины.append("QED не измерен: условие отбора обучающей выборки не проверено")

    соответствует = not классы
    # Порядок важности при нескольких причинах: посторонняя молекула тяжелее иона,
    # ион тяжелее добавки. Все причины при этом остаются в `reasons`.
    класс = classes_priority(классы) if классы else POCKET_CONTENT_READY

    return PocketVerdict(
        pdb_id=pdb_id,
        expo_id=expo_id,
        pocket_class=класс,
        matches_diffsbdd=соответствует,
        reasons=tuple(причины),
        ligand_atoms=len(целевой),
        ligand_copies=len(копии),
        radius=radius,
        qed=qed,
        water_atoms=вода,
        neighbours=tuple(соседи),
        dataset=dataset,
    )


_CLASS_PRIORITY: Final[tuple[str, ...]] = (
    POCKET_CONTENT_MULTI_LIGAND,
    POCKET_CONTENT_COFACTOR,
    POCKET_CONTENT_COVALENT,
    POCKET_CONTENT_METAL,
    POCKET_CONTENT_ADDITIVE,
    POCKET_CONTENT_LOW_QED,
)


def classes_priority(found: list[str]) -> str:
    """Главный класс из нескольких найденных — по закреплённому порядку важности."""
    for класс in _CLASS_PRIORITY:
        if класс in found:
            return класс
    return found[0]


# ---------------------------------------------------------------------------
# Загрузка (единственные функции, которым нужна сеть)
# ---------------------------------------------------------------------------


def fetch_pdb(pdb_id: str, cache_dir: Path, *, timeout: float = 60.0) -> str:
    """Скачивает исходный файл PDB с RCSB, складывая копию в кэш.

    Кэш обязателен, а не удобен: выборка в несколько десятков структур скачивается
    минуты, и повторный прогон не должен зависеть от того, отвечает ли RCSB сейчас.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    файл = cache_dir / f"{pdb_id.lower()}.pdb"
    if файл.exists():
        return файл.read_text(encoding="utf-8")
    адрес = RCSB_STRUCTURE_URL.format(pdb_id=pdb_id.upper())
    # Одна повторная попытка: на выборке из 60 структур RCSB обрывает передачу
    # примерно раз, и без повтора в отчёте появляется «не разобрано» там, где
    # дело не в структуре, а в сети.
    последняя: Exception | None = None
    for _ in range(RCSB_ATTEMPTS):
        try:
            ответ = urllib.request.urlopen(адрес, timeout=timeout).read()
            текст: str = bytes(ответ).decode("utf-8", "replace")
        except Exception as ошибка:  # сеть, 404, обрыв — причина сохраняется, а не глотается
            последняя = ошибка
            continue
        файл.write_text(текст, encoding="utf-8")
        return текст
    raise PocketContentError(
        f"Структура {pdb_id} не получена с RCSB: {последняя}"
    ) from последняя


def fetch_ligand_qed(expo_id: str, cache_dir: Path, *, timeout: float = 60.0) -> float:
    """QED лиганда по идеальной геометрии химического компонента RCSB.

    Считать QED прямо по координатам из PDB нельзя: в записях `HETATM` нет порядков
    связей, и RDKit получил бы молекулу без ароматичности и зарядов. Файл
    `<код>_ideal.sdf` несёт связи явно — это тот же компонент, что стоит в структуре.
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    cache_dir.mkdir(parents=True, exist_ok=True)
    файл = cache_dir / f"{expo_id.lower()}_ideal.sdf"
    if файл.exists():
        блок = файл.read_text(encoding="utf-8")
    else:
        адрес = RCSB_LIGAND_URL.format(code=expo_id.upper())
        try:
            сырьё = urllib.request.urlopen(адрес, timeout=timeout).read()
            блок = bytes(сырьё).decode("utf-8", "replace")
        except Exception as ошибка:
            raise PocketContentError(
                f"Компонент {expo_id} не получен с RCSB: {ошибка}"
            ) from ошибка
        файл.write_text(блок, encoding="utf-8")

    молекула = Chem.MolFromMolBlock(блок)
    if молекула is None:
        raise PocketContentError(f"RDKit не разобрал компонент {expo_id}: QED не считается")
    значение: float = float(QED.qed(молекула))
    return значение


def classify_structure(
    pdb_id: str,
    expo_id: str,
    cache_dir: Path,
    *,
    radius: float = DIFFSBDD_POCKET_CUTOFF_A,
    with_qed: bool = True,
    dataset: str = DIFFSBDD_DATASET_CROSSDOCKED,
) -> PocketVerdict:
    """Скачивает структуру и компонент и возвращает вердикт по комплексу."""
    текст = fetch_pdb(pdb_id, cache_dir)
    qed: float | None = None
    if with_qed:
        try:
            qed = fetch_ligand_qed(expo_id, cache_dir)
        except PocketContentError:
            qed = None  # причина попадёт в `reasons` вердикта как «QED не измерен»
    return classify_pocket(текст, pdb_id, expo_id, qed=qed, radius=radius, dataset=dataset)
