"""Сверка нашего отпечатка с эталонным отпечатком KLIFS.

Совпадения 1:1 не ждём: эталон считает сторонняя программа (FingerPrintLib) по своим
правилам. Задача сверки другая — поймать грубую поломку, которая выглядит как рабочий
результат: перепутанную раскладку 595 бит, инвертированную направленность
донор/акцептор и отпечаток, в котором остались одни ван-дер-ваальсовы касания.
Все три уже случались: раскладка была неверной, а свёрнутый до
гидрофобных контактов отпечаток — главный дефект черновика.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from kinase_ifp.config import KLIFS_INTERACTION_TYPES, SCORING_INTERACTION_TYPES
from kinase_ifp.fingerprint import compute_ifp
from kinase_ifp.klifs import PROTONATION_MODES, build_target_package
from kinase_ifp.molecule_io import read_mol
from kinase_ifp.pocket import IMPLICIT_PROTONATION, load_pocket
from kinase_ifp.protonate import prepare_ligand
from kinase_ifp.protonation import ON_MISSING_H_IMPLICIT, annotate_protonation
from kinase_ifp.similarity import select_types, tanimoto

# Пары «наш тип ↔ тип эталона», по которым видно инверсию направленности: в ProLIF имя
# описывает роль лиганда, в KLIFS — роль белка, и перепутать их легко (см. комментарий
# к KLIFS_TO_PROLIF в config.py).
_DIRECTIONAL_PAIR: tuple[str, str] = ("DON", "ACC")


class CalibrationError(RuntimeError):
    """Сверку невозможно выполнить: нет эталона или пакет мишени неполон."""


@dataclass(frozen=True)
class TypeComparison:
    """Расхождение по одному типу взаимодействий."""

    interaction_type: str
    ours: int
    reference: int
    shared: int

    @property
    def missed(self) -> int:
        """Биты, которые есть у KLIFS и отсутствуют у нас."""
        return self.reference - self.shared

    @property
    def extra(self) -> int:
        """Биты, которые есть у нас и отсутствуют у KLIFS."""
        return self.ours - self.shared


@dataclass(frozen=True)
class CalibrationResult:
    """Результат сверки в одном режиме протонирования."""

    protonation: str
    tanimoto_all: float
    tanimoto_scoring: float
    by_type: tuple[TypeComparison, ...]

    @property
    def ours(self) -> int:
        return sum(t.ours for t in self.by_type)

    @property
    def reference(self) -> int:
        return sum(t.reference for t in self.by_type)

    @property
    def shared(self) -> int:
        return sum(t.shared for t in self.by_type)

    @property
    def extra(self) -> int:
        return sum(t.extra for t in self.by_type)

    @property
    def scoring_defined(self) -> bool:
        """Есть ли у эталона хотя бы один бит по типам скора.

        Если нет — воспроизводить нечего, и `tanimoto_scoring` не «ноль», а величина,
        которой не существует: при пустом объединении бит `tanimoto` возвращает 0.0
, и без этого флага «нечего воспроизводить» неотличимо
        от «не воспроизвели ничего». Реальный случай — 3bhy (DAPK3), у которой все
        14 бит эталона гидрофобные (`docs/metrics.md`, 3.9); по всей базе таких
        структур 84 из 7462.
        """
        return any(
            t.reference > 0 for t in self.by_type if t.interaction_type in SCORING_INTERACTION_TYPES
        )


def compare_with_reference(
    ours: np.ndarray, reference: np.ndarray, protonation: str
) -> CalibrationResult:
    """Сравнивает посчитанный отпечаток с эталонным по каждому из семи типов."""
    a, b = ours.astype(bool), reference.astype(bool)
    by_type = tuple(
        TypeComparison(
            interaction_type=name,
            ours=int(a[i].sum()),
            reference=int(b[i].sum()),
            shared=int((a[i] & b[i]).sum()),
        )
        for i, name in enumerate(KLIFS_INTERACTION_TYPES)
    )
    return CalibrationResult(
        protonation=protonation,
        tanimoto_all=tanimoto(ours, reference),
        # Отдельно по типам скора: именно они идут в `ifp_score`, и расхождение по ним
        # значит для работы больше, чем расхождение по гидрофобным контактам.
        tanimoto_scoring=tanimoto(
            select_types(ours, SCORING_INTERACTION_TYPES),
            select_types(reference, SCORING_INTERACTION_TYPES),
        ),
        by_type=by_type,
    )


def gross_failures(result: CalibrationResult, ours: np.ndarray, reference: np.ndarray) -> list[str]:
    """Перечисляет грубые поломки, которые обязана поймать дымовая сверка.

    Пустой список означает, что ни одна из трёх не подтвердилась. Это не «отпечаток
    верен» — это «отпечаток не сломан очевидным образом»; величина расхождения
    оценивается отдельно, на выборке структур.
    """
    поломки: list[str] = []

    if result.ours == 0:
        поломки.append("отпечаток пуст: не найдено ни одного взаимодействия")

    специфичные = sum(
        t.ours for t in result.by_type if t.interaction_type in SCORING_INTERACTION_TYPES
    )
    if result.ours > 0 and специфичные == 0:
        поломки.append(
            "в отпечатке остались одни гидрофобные контакты — так выглядит расчёт "
            "без водородов (дефект черновика)"
        )

    # Инверсия направленности: наших бит одного типа нет там, где эталон их ставит,
    # зато они ровно на позициях парного типа.
    a, b = ours.astype(bool), reference.astype(bool)
    прямых = 0
    перекрёстных = 0
    for первый, второй in (_DIRECTIONAL_PAIR, _DIRECTIONAL_PAIR[::-1]):
        i = KLIFS_INTERACTION_TYPES.index(первый)
        j = KLIFS_INTERACTION_TYPES.index(второй)
        прямых += int((a[i] & b[i]).sum())
        перекрёстных += int((a[i] & b[j]).sum())
    if перекрёстных > прямых:
        поломки.append(
            f"направленность донор/акцептор выглядит инвертированной: совпадений "
            f"крест-накрест {перекрёстных} против {прямых} прямых"
        )

    return поломки


def available_modes(target_json: Path) -> list[str]:
    """Какие режимы протонирования можно посчитать для этого пакета мишени."""
    package = json.loads(target_json.read_text(encoding="utf-8"))
    base = target_json.parent

    режимы = [str(package["protonation"])]
    второй = IMPLICIT_PROTONATION if режимы[0] != IMPLICIT_PROTONATION else "explicit"
    файл = (
        package.get("protein_noh_pdb")
        if второй == IMPLICIT_PROTONATION
        else package["pocket_pdb"]
    )
    if файл and (base / файл).is_file():
        режимы.append(второй)
    return [m for m in режимы if m in PROTONATION_MODES]


def calibrate_target(target_json: Path, modes: list[str] | None = None) -> list[CalibrationResult]:
    """Считает сверку для кристаллического лиганда мишени во всех доступных режимах.

    Режим, отличный от записанного в пакете, считается на временной копии каталога:
    менять `target.json` ради измерения нельзя — режим задаётся явно, а не побочным
    эффектом расчёта.
    """
    режимы = modes if modes is not None else available_modes(target_json)
    базовый = load_pocket(target_json)
    if базовый.reference_ifp is None:
        raise CalibrationError(
            f"{target_json}: у мишени нет эталонного отпечатка KLIFS, сверять не с чем"
        )

    результаты: list[CalibrationResult] = []
    # Один временный каталог на весь цикл: `Pocket` хранит пути, а не содержимое,
    # и удалять копию раньше, чем посчитан отпечаток, нельзя.
    #
    # ignore_cleanup_errors: на Windows MDAnalysis держит прочитанный PDB открытым, и
    # удаление каталога падает с WinError 32. Ронять из-за этого посчитанную сверку
    # нельзя — временные файлы всё равно уберёт система.
    with TemporaryDirectory(ignore_cleanup_errors=True) as времянка:
        for номер, режим in enumerate(режимы):
            if режим == базовый.protonation:
                карман = базовый
            else:
                карман = load_pocket(_copy_in_mode(target_json, Path(времянка) / str(номер), режим))
            эталон = карман.reference_ifp
            if эталон is None:
                raise CalibrationError(
                    f"{target_json}: у копии пакета в режиме {режим} пропал эталонный отпечаток"
                )
            # `read_mol` вместо `MolFromMolFile`: путь уходит в C++ через ANSI,
            # а пакет здесь лежит во временной копии, корень которой — `%TEMP%`
            # внутри домашнего каталога пользователя, и тот может быть любым.
            сырой = read_mol(карман.ligand_path)
            if сырой is None:
                raise CalibrationError(
                    f"{карман.ligand_path}: RDKit не разобрал лиганд мишени"
                )
            лиганд = prepare_ligand(сырой)
            отпечаток = compute_ifp(карман, лиганд)
            результаты.append(compare_with_reference(отпечаток, эталон, режим))
    return результаты


def _copy_in_mode(target_json: Path, куда: Path, protonation: str) -> Path:
    """Копирует пакет мишени, подменив режим протонирования. Возвращает путь к копии.

    Менять исходный `target.json` ради измерения нельзя: выбранный режим фиксируется
    явно, а не побочным эффектом расчёта.
    """
    shutil.copytree(target_json.parent, куда)
    копия = куда / target_json.name
    package = json.loads(копия.read_text(encoding="utf-8"))
    package["protonation"] = protonation
    копия.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
    return копия


@dataclass(frozen=True)
class StructureCalibration:
    """Сверка одной структуры выборки в одном режиме протонирования."""

    pdb_id: str
    kinase: str
    group: str
    klifs_structure_id: int
    result: CalibrationResult

    @property
    def protonation(self) -> str:
        return self.result.protonation


@dataclass(frozen=True)
class SampleSummary:
    """Сводка сверки по выборке структур в одном режиме протонирования.

    Медиана и IQR по типам скора считаются только по структурам, у которых эти типы
    определены (`CalibrationResult.scoring_defined`); сколько структур исключено —
    в `n_scoring_undefined`. Число обязано доехать до текста: молча уменьшенный
    знаменатель сдвинул бы медиану, и заметить это было бы нечем.

    Разбивка `by_type` суммирована по тем же структурам, что и медианы. Разные выборки
    для медианы и для разбивки — ровно то, из-за чего числа от 31.08 в текст
    не пошли.
    """

    protonation: str
    n_structures: int
    n_scoring_undefined: int
    tanimoto_all: tuple[float, float, float]  # медиана, Q1, Q3
    tanimoto_scoring: tuple[float, float, float]
    by_type: tuple[TypeComparison, ...]


def _median_iqr(values: Sequence[float]) -> tuple[float, float, float]:
    """Медиана и границы межквартильного размаха. Пустая выборка даёт NaN, а не ноль."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    array = np.asarray(values, dtype=float)
    return (
        float(np.median(array)),
        float(np.percentile(array, 25)),
        float(np.percentile(array, 75)),
    )


def summarize_calibration(rows: Sequence[StructureCalibration]) -> SampleSummary:
    """Сводит сверку по выборке структур: медианы, IQR и суммарная разбивка по типам.

    Все строки обязаны быть посчитаны в одном режиме протонирования: отпечатки разных
    режимов несравнимы, и смешанная сводка была бы средним по двум
    протоколам расчёта.

    Квартили — на numpy, без scipy: пакет не объявлен в `pyproject.toml`
.
    """
    if not rows:
        raise CalibrationError("Выборка пуста: сводить нечего")

    режимы = {row.protonation for row in rows}
    if len(режимы) > 1:
        raise CalibrationError(
            f"В сводке смешаны режимы протонирования {sorted(режимы)}: отпечатки разных "
            f"режимов несравнимы"
        )

    with_scoring = [row for row in rows if row.result.scoring_defined]
    by_type = tuple(
        TypeComparison(
            interaction_type=name,
            ours=sum(row.result.by_type[i].ours for row in rows),
            reference=sum(row.result.by_type[i].reference for row in rows),
            shared=sum(row.result.by_type[i].shared for row in rows),
        )
        for i, name in enumerate(KLIFS_INTERACTION_TYPES)
    )
    return SampleSummary(
        protonation=rows[0].protonation,
        n_structures=len(rows),
        n_scoring_undefined=len(rows) - len(with_scoring),
        tanimoto_all=_median_iqr([row.result.tanimoto_all for row in rows]),
        tanimoto_scoring=_median_iqr([row.result.tanimoto_scoring for row in with_scoring]),
        by_type=by_type,
    )


def calibrate_sample(
    session: Any,
    structures: pd.DataFrame,
    fingerprints: Mapping[int, str],
    workdir: Path,
) -> list[StructureCalibration]:
    """Считает сверку для выборки структур: собирает пакеты мишеней и сверяет с эталоном.

    Принимает сессию KLIFS, таблицу структур (по строке на структуру, как отдаёт
    `klifs.select_best_per_kinase`), эталонные отпечатки по `structure.klifs_id`
    и каталог, в котором собираются пакеты. Возвращает по строке на каждую пару
    «структура × доступный режим протонирования».

    Пакеты собираются в переданный каталог, а не в `data/targets/`: двенадцать пакетов —
    около 17 МБ, и класть их в репозиторий ради одного измерения не нужно. Числа
    сохраняются отдельно (`scripts/calibrate_ifp.py --sample`), поэтому проверить их
    можно, не пересобирая структуры.

    Структура без водородов не теряется, а считается в режиме `implicit-prolif` —
    то же правило, что у `select_target.py`; режим при этом виден
    в результате, и смешать его с `explicit` в одной сводке не даст
    `summarize_calibration`.
    """
    _require_sample_columns(structures)

    rows: list[StructureCalibration] = []
    for _, structure in structures.iterrows():
        klifs_id = int(structure["structure.klifs_id"])
        target_json = build_target_package(
            session, structure, workdir, fingerprints.get(klifs_id)
        )
        annotate_protonation(target_json, on_missing_hydrogens=ON_MISSING_H_IMPLICIT)
        for result in calibrate_target(target_json):
            rows.append(
                StructureCalibration(
                    pdb_id=str(structure["structure.pdb_id"]),
                    kinase=str(structure["kinase.klifs_name"]),
                    group=_группа(structure),
                    klifs_structure_id=klifs_id,
                    result=result,
                )
            )
    return rows


def _группа(structure: pd.Series) -> str:
    """Группа киназы строкой; пустое значение остаётся пустым, а не превращается в "nan".

    В таблице структур KLIFS колонка `kinase.group` не заполнена,
    и `str(nan)` дал бы в отчёте слово «nan» вместо честного прочерка.
    """
    значение = structure.get("kinase.group", "")
    if значение is None or pd.isna(значение):
        return ""
    return str(значение).strip()


def _require_sample_columns(structures: pd.DataFrame) -> None:
    """Проверяет, что в таблице выборки есть всё, что нужно сверке."""
    if structures.empty:
        raise CalibrationError("Таблица структур для сверки пуста")
    missing = [
        column
        for column in ("structure.klifs_id", "structure.pdb_id", "kinase.klifs_name")
        if column not in structures.columns
    ]
    if missing:
        raise CalibrationError(f"В таблице структур для сверки нет колонок {missing}")


#: Порядок колонок по-структурного файла сверки. Объявлен здесь, а не в скрипте:
#: писатель и читатель обязаны видеть один и тот же состав, иначе они разойдутся
#: при первой же добавленной колонке, и старые числа перестанут читаться молча.
CALIBRATION_CSV_COLUMNS: tuple[str, ...] = (
    "pdb_id",
    "kinase",
    "group",
    "klifs_structure_id",
    "protonation",
    "tanimoto_all",
    "tanimoto_scoring",
    "scoring_defined",
    *(
        f"{тип}_{поле}"
        for тип in KLIFS_INTERACTION_TYPES
        for поле in ("ours", "reference", "shared")
    ),
)

#: CSV пишется с переводом строки '\n': тот же порядок, что у прогонов
#: (`experiments/layout.CSV_EOL`) и что у правила `data/** text eol=lf`
#: в `.gitattributes`. Модуль `csv` по умолчанию ставит CRLF, и файл, записанный кодом,
#: отличался бы от того же файла, сохранённого git, во всех строках сразу.
CALIBRATION_CSV_EOL = "\n"


def write_structure_calibration_csv(rows: Sequence[StructureCalibration], path: Path) -> None:
    """По-структурные числа выборки: чтобы их можно было проверить без пересборки пакетов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as поток:
        писатель = csv.writer(поток, lineterminator=CALIBRATION_CSV_EOL)
        писатель.writerow(CALIBRATION_CSV_COLUMNS)
        for row in rows:
            запись: list[object] = [
                row.pdb_id,
                row.kinase,
                row.group,
                row.klifs_structure_id,
                row.protonation,
                f"{row.result.tanimoto_all:.6f}",
                f"{row.result.tanimoto_scoring:.6f}" if row.result.scoring_defined else "",
                int(row.result.scoring_defined),
            ]
            for t in row.result.by_type:
                запись += [t.ours, t.reference, t.shared]
            писатель.writerow(запись)


def read_structure_calibration_csv(path: Path) -> list[StructureCalibration]:
    """Читает по-структурные числа обратно — так сравнивают прежнюю сверку с новой.

    Нужна затем, чтобы прежние числа приходили в отчёт из файла, а не переписывались
    руками из текста: иначе разница форматирования читалась бы как разница выборок.

    `scoring_defined` пересчитывается из разбивки по типам, а колонка файла служит
    сверкой: расхождение роняет чтение. Оно означает, что файл испорчен или записан
    другой версией кода, и считать по нему «было» нельзя.
    """
    if not path.is_file():
        raise CalibrationError(f"Нет файла с прежними числами сверки: {path}")

    таблица = pd.read_csv(path)
    нет_колонок = [имя for имя in CALIBRATION_CSV_COLUMNS if имя not in таблица.columns]
    if нет_колонок:
        raise CalibrationError(f"В {path} не хватает колонок: {нет_колонок}")

    строки: list[StructureCalibration] = []
    for _, запись in таблица.iterrows():
        by_type = tuple(
            TypeComparison(
                interaction_type=тип,
                ours=int(запись[f"{тип}_ours"]),
                reference=int(запись[f"{тип}_reference"]),
                shared=int(запись[f"{тип}_shared"]),
            )
            for тип in KLIFS_INTERACTION_TYPES
        )
        скор = запись["tanimoto_scoring"]
        результат = CalibrationResult(
            protonation=str(запись["protonation"]),
            tanimoto_all=float(запись["tanimoto_all"]),
            tanimoto_scoring=float("nan") if pd.isna(скор) else float(скор),
            by_type=by_type,
        )
        if результат.scoring_defined != bool(int(запись["scoring_defined"])):
            raise CalibrationError(
                f"{path}, структура {запись['pdb_id']}: колонка scoring_defined не сходится "
                f"с разбивкой по типам. Файл записан другой версией кода"
            )
        строки.append(
            StructureCalibration(
                pdb_id=str(запись["pdb_id"]),
                kinase=str(запись["kinase"]),
                group="" if pd.isna(запись["group"]) else str(запись["group"]),
                klifs_structure_id=int(запись["klifs_structure_id"]),
                result=результат,
            )
        )
    return строки
