"""Внешняя оценка позы докингом: `vina_score` и `ligand_efficiency`.

Зачем модуль нужен. Все наши числа до сих пор считались одной и той же мерой: потенциал
поднимает `ifp_score`, и результатом предъявляется, что `ifp_score` поднялся. Круг
размыкается только независимой мерой, и такой мерой служит докинг. Здесь она
и живёт.

**Поза оценивается как есть, без поиска и без минимизации** (`docs/metrics.md` 3.8).
Это не упрощение ради скорости, а единственный осмысленный режим для нашей задачи:
мы проверяем ту позу, которую поставил потенциал, а полный докинг нашёл бы свою
и ответил бы на другой вопрос. Так же поступает DiffInt (Sako et al., JCIM 2025),
прямо оговаривая, что докинг-скоры с поиском позы непригодны для оценки моделей,
которые позу порождают.

Рецептор готовится заново при каждом запуске во временном каталоге, а не хранится
рядом с пакетом мишени. Причина не в экономии места: паспорт прогона несёт
`target.package_sha256`, и `score_run` отказывается считать при расхождении,
поэтому новый файл внутри `data/targets/<pdb_id>/` остановил бы расчёт везде.
Подготовка занимает несколько секунд и делается один раз на мишень.

Заряды Гастайгера, которые проставляет Meeko, оценочная функция `vina` не использует
вовсе — электростатика в ней не выделена отдельным членом. Они пишутся в PDBQT
потому, что этого требует формат, а не потому, что влияют на число.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from kinase_ifp.config import (
    VINA_BOX_SIZE_A,
    VINA_RECEPTOR_PADDING_A,
    VINA_SCORING_FUNCTION,
)
from kinase_ifp.molecule_io import read_mol

_ПОДГОТОВЩИК_РЕЦЕПТОРА = "mk_prepare_receptor.py"


class DockingError(RuntimeError):
    """Позу оценить нельзя: рецептор не готовится, лиганд не переводится или вне бокса."""


@dataclass(frozen=True)
class DockingScore:
    """Оценка одной позы. `total` — это и есть `vina_score` формата выхода этапа.

    `inter` — межмолекулярная часть, `intra` — внутримолекулярная. Для нашей задачи
    важна первая: конформацию потенциал не меняет, поэтому вся разница «до и после»
    обязана сидеть в межмолекулярном члене, и расхождение этих двух величин — признак
    того, что с позой что-то сделали помимо твёрдотельного сдвига.
    """

    total: float
    inter: float
    intra: float
    n_heavy_atoms: int

    @property
    def ligand_efficiency(self) -> float:
        """`−vina_score / n_heavy_atoms` — обязательна вместе со скором (`docs/metrics.md` 3.8).

        Без неё метрика вознаграждает просто крупные молекулы: энергия Vina растёт
        по модулю с числом атомов почти линейно.
        """
        return -self.total / self.n_heavy_atoms


def prepare_receptor(target_json: Path, workdir: Path) -> Path:
    """Готовит PDBQT рецептора из белка пакета мишени; возвращает путь к файлу.

    Берётся `protein.pdb` — целый белок, а не карман: карман KLIFS собран из 85
    несмежных остатков с оборванными связями, и перцепция типов атомов на срезах
    дала бы мусор. На стоимость расчёта это не влияет, потому что энергия считается
    только внутри бокса.
    """
    белок = target_json.parent / "protein.pdb"
    лиганд = target_json.parent / "ligand.sdf"
    if not белок.is_file():
        raise DockingError(f"нет protein.pdb в пакете {target_json.parent}")
    workdir.mkdir(parents=True, exist_ok=True)
    основа = workdir / "receptor"
    итог = subprocess.run(  # noqa: S603
        [
            _ПОДГОТОВЩИК_РЕЦЕПТОРА,
            "--read_pdb",
            str(белок),
            "-o",
            str(основа),
            "-p",
            "--box_enveloping",
            str(лиганд),
            "--padding",
            str(VINA_RECEPTOR_PADDING_A),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    рецептор = основа.with_suffix(".pdbqt")
    if итог.returncode != 0 or not рецептор.is_file():
        raise DockingError(
            f"{_ПОДГОТОВЩИК_РЕЦЕПТОРА} не собрал рецептор для {target_json.parent.name}: "
            f"код {итог.returncode}, вывод: {итог.stdout[-400:]} {итог.stderr[-400:]}"
        )
    return рецептор


def ligand_pdbqt(mol: Chem.Mol) -> str:
    """Переводит молекулу RDKit в PDBQT-строку, сохраняя её координаты.

    Принимает молекулу с явными водородами и конформером. Поднимает `DockingError`,
    если Meeko отказался: молчаливый пропуск молекулы исказил бы выборку — в таблице
    оказались бы только те позы, которые перевелись, и сравнение «до и после» поехало
    бы на разных наборах.
    """
    # Импорт внутри функции: meeko тянет за собой заметный кусок зависимостей,
    # а модуль импортируется и там, где докинг не считается.
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    if mol.GetNumConformers() == 0:
        raise DockingError("у молекулы нет конформера: оценивать нечего")
    водородов = sum(1 for атом in mol.GetAtoms() if атом.GetAtomicNum() == 1)
    неявных = sum(атом.GetTotalNumHs() for атом in mol.GetAtoms())
    if водородов == 0 and неявных > 0:
        # Meeko падает здесь ValueError с английским текстом; своя проверка называет
        # и причину, и лекарство, а тип ошибки остаётся одним на весь модуль.
        raise DockingError(
            f"водороды неявные ({неявных} шт.): типы атомов PDBQT без них не назначить, "
            "зови protonate.prepare_ligand до оценки"
        )
    фрагментов = len(Chem.GetMolFrags(mol))
    if фрагментов > 1:
        # Диффузионная модель порождает и несвязные молекулы: это известный дефект
        # генерации, который уже учитывается колонкой `connected` в таблице результатов.
        # Докинг к такому набору кусков неприменим по смыслу - «лиганда» здесь
        # нет, - поэтому отказ явный, а не ValueError из недр Meeko.
        raise DockingError(f"молекула несвязная: {фрагментов} фрагмента, это не один лиганд")
    наборы = MoleculePreparation().prepare(mol)
    if not наборы:
        raise DockingError("Meeko не разобрал молекулу")
    строка, успех, причина = PDBQTWriterLegacy.write_string(наборы[0])
    if not успех:
        raise DockingError(f"Meeko не записал PDBQT: {причина}")
    return str(строка)


class PoseScorer:
    """Оценщик поз в одном кармане: карты Vina считаются один раз на мишень.

    Карты — самая дорогая часть расчёта, а от молекулы они не зависят, поэтому
    оценка сотни поз стоит примерно столько же, сколько оценка одной.
    """

    def __init__(self, target_json: Path, workdir: Path, box_size: float = VINA_BOX_SIZE_A):
        from vina import Vina

        эталон = read_mol(target_json.parent / "ligand.sdf")
        if эталон is None:
            raise DockingError(f"не читается ligand.sdf пакета {target_json.parent}")
        центр = эталон.GetConformer().GetPositions().mean(axis=0)
        self._vina = Vina(sf_name=VINA_SCORING_FUNCTION, verbosity=0)
        self._vina.set_receptor(str(prepare_receptor(target_json, workdir)))
        self._vina.compute_vina_maps(
            center=[float(к) for к in центр], box_size=[box_size] * 3
        )
        self._центр = центр
        self._ребро = box_size

    def score(self, mol: Chem.Mol) -> DockingScore:
        """Оценивает позу как есть; поднимает `DockingError`, если поза вне бокса."""
        self._загрузить(mol)
        return self._собрать(self._vina.score(), mol)

    def minimize(self, mol: Chem.Mol) -> DockingScore:
        """Энергия после локальной минимизации позы — величина «Vina Min» литературы.

        Нужна не вместо `score`, а рядом с ней. Оценка позы как есть отвечает на вопрос
        «годится ли эта поза», и у молекул генеративных моделей она уходит далеко
        в плюс: небольшое наложение атомов даёт штраф в десятки килокалорий, потому
        что отталкивание у Vina растёт неограниченно. Именно поэтому статьи поля
        приводят минимизированную величину, и без неё наши числа не с чем сравнивать.
        Минимизация двигает только лиганд и меняет его конформацию, поэтому подменять
        ею `vina_score` формата выхода этапа нельзя.
        """
        self._загрузить(mol)
        return self._собрать(self._vina.optimize(), mol)

    def _загрузить(self, mol: Chem.Mol) -> None:
        отклонение = np.abs(mol.GetConformer().GetPositions() - self._центр).max()
        if отклонение > self._ребро / 2:
            raise DockingError(
                f"поза выходит за бокс: отклонение {отклонение:.1f} A при половине "
                f"ребра {self._ребро / 2:.1f} A"
            )
        self._vina.set_ligand_from_string(ligand_pdbqt(mol))

    @staticmethod
    def _собрать(значения: object, mol: Chem.Mol) -> DockingScore:
        ряд = list(значения)  # type: ignore[call-overload]
        return DockingScore(
            total=float(ряд[0]),
            inter=float(ряд[1]),
            intra=float(ряд[2]),
            n_heavy_atoms=mol.GetNumHeavyAtoms(),
        )
