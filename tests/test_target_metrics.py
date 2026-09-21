"""Тесты целевых метрик."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from evaluation.target import (
    hinge_hbond,
    ifp_tanimoto,
    ifp_tversky,
    key_hbond_positions,
)
from experiments.layout import SOURCE_LOCAL_POSES
from experiments.local_poses import build_local_pose_run
from experiments.metrics import (
    METRICS_COLUMNS,
    MetricsError,
    compute_run_metrics,
    read_metrics,
    unfilled_columns,
)
from experiments.registry import describe_run
from experiments.runs import record_condition
from kinase_ifp.config import HINGE_POSITIONS, KLIFS_IFP_SHAPE, KLIFS_INTERACTION_TYPES
from kinase_ifp.fingerprint import compute_ifp
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.pocket import Pocket


@pytest.fixture(scope="module")
def прогон(tmp_path_factory: pytest.TempPathFactory, target_json: Path) -> Path:
    """Небольшой набор поз: нативная плюс искажённые копии."""
    runs = tmp_path_factory.mktemp("runs")
    return build_local_pose_run(target_json, n=6, seed=1, runs_dir=runs, day="2026-08-24")


def test_кристаллическая_поза_совпадает_с_эталоном_сама_с_собой(
    pocket: Pocket, crystal_ligand: Chem.Mol
) -> None:
    """Танимото отпечатка с самим собой равен 1.0.

    Проверяется именно наш отпечаток, а не эталон KLIFS: с эталоном совпадения 1:1
    не ждём, его считает сторонняя программа.
    """
    отпечаток = compute_ifp(pocket, crystal_ligand)

    assert ifp_tanimoto(отпечаток, отпечаток) == pytest.approx(1.0)
    assert ifp_tversky(отпечаток, отпечаток) == pytest.approx(1.0)


def test_tversky_несимметричен_и_считает_долю_эталона() -> None:
    """Порядок аргументов проверяется, а не подразумевается: ошибка в нём не видна из таблицы.

    Фикстура намеренно несимметрична по числу бит: у эталона три
    взаимодействия, у молекулы два. При равном числе бит Тверски давал бы 0.5 в обе
    стороны, и перестановка аргументов теста не роняла — проверено подменой.
    """
    эталон = np.zeros(KLIFS_IFP_SHAPE, dtype=int)
    молекула = np.zeros(KLIFS_IFP_SHAPE, dtype=int)
    # Эталон: три взаимодействия, молекула воспроизвела одно из них и добавила своё.
    эталон[KLIFS_INTERACTION_TYPES.index("DON"), 16] = 1
    эталон[KLIFS_INTERACTION_TYPES.index("ACC"), 45] = 1
    эталон[KLIFS_INTERACTION_TYPES.index("ACC"), 59] = 1
    молекула[KLIFS_INTERACTION_TYPES.index("DON"), 16] = 1
    молекула[KLIFS_INTERACTION_TYPES.index("HYD"), 2] = 1

    # Доля взаимодействий эталона: одно из трёх. Лишний контакт молекулы не штрафуется.
    assert ifp_tversky(молекула, эталон) == pytest.approx(1 / 3)
    # Перестановка аргументов даёт долю битов молекулы — другое число и другой вопрос.
    assert ifp_tversky(эталон, молекула) == pytest.approx(0.5)
    # Танимото симметричен и штрафует: общее одно, объединение четыре.
    assert ifp_tanimoto(молекула, эталон) == pytest.approx(0.25)
    assert ifp_tanimoto(эталон, молекула) == pytest.approx(0.25)


def test_ключевые_позиции_эталона_6tgu(pocket: Pocket) -> None:
    """У кристаллического лиганда 6tgu H-связи не с шарниром, а с позициями 17 и 81.

    Ровно этот факт заставил считать `hinge_hbond` по позициям эталона,
    а не по `HINGE_POSITIONS`: по шарниру метрика давала бы 0 на верхней планке сравнения.
    """
    assert pocket.reference_ifp is not None
    позиции = key_hbond_positions(pocket.reference_ifp)

    assert позиции == (17, 81)
    assert not set(позиции) & set(HINGE_POSITIONS)


def test_hinge_hbond_считает_по_переданным_позициям(pocket: Pocket) -> None:
    """Позиции — аргумент: метрики берут позиции эталона, разметка — шарнир."""
    assert pocket.reference_ifp is not None

    assert hinge_hbond(pocket.reference_ifp, key_hbond_positions(pocket.reference_ifp)) == 1
    assert hinge_hbond(pocket.reference_ifp, HINGE_POSITIONS) == 0


def test_hinge_hbond_отвергает_позиции_вне_диапазона(pocket: Pocket) -> None:
    """Ноль вместо ошибки был бы неотличим от честного «связи нет»."""
    assert pocket.reference_ifp is not None

    with pytest.raises(ValueError, match="вне диапазона"):
        hinge_hbond(pocket.reference_ifp, [0])
    with pytest.raises(ValueError, match="пуст"):
        hinge_hbond(pocket.reference_ifp, [])


def test_metrics_csv_по_заголовку(прогон: Path, target_json: Path) -> None:
    """Заголовок таблицы метрик задан целиком; неприменимые колонки присутствуют и пусты."""
    compute_run_metrics(прогон, target_json)

    строки = read_metrics(прогон)
    assert строки
    assert tuple(строки[0]) == METRICS_COLUMNS
    # Набор поз: сверх PoseBusters и докинга пустыми обязаны быть и контрольные
    # свойства — на одной молекуле в ста конформациях они вырождены.
    пустые = unfilled_columns(SOURCE_LOCAL_POSES)
    assert "qed" in пустые
    for строка in строки:
        assert all(строка[колонка] == "" for колонка in пустые)
        assert строка["run_id"] == прогон.name
        assert строка["valid"] == "1"
        assert строка["connected"] == "1"


def test_танимото_нативной_позы_к_эталону_klifs(прогон: Path, target_json: Path) -> None:
    """Нативное положение даёт с эталоном KLIFS 0.588 — то же число, что измерила сверка.

    Единицы здесь не ждём и ждать не можем: эталон считает сторонняя программа
    FingerPrintLib с другими правилами, и всё расхождение сидит в гидрофобных
    контактах, которые она ставит щедрее ProLIF (`docs/calibration.md`).

    **Нативная поза при этом не обязана быть лучшей по этой метрике** — и не является:
    на наборе поз её обгоняют смещённые копии, случайно закрывающие лишние биты `HYD`.
    Это свойство метрики, а не дефект: ранжировать позы — работа скора `ifp_score`,
    который `HYD` не учитывает вовсе и нативную позу первой ставит.
    Проверка держит оба утверждения разом, чтобы разница между двумя числами про одну
    молекулу не стёрлась при следующей правке.
    """
    compute_run_metrics(прогон, target_json)

    строки = read_metrics(прогон)
    with open_sdf(прогон / "molecules.sdf") as поставщик:
        rmsd = {
            mol.GetProp("mol_id"): float(mol.GetProp("rmsd_to_ref"))
            for mol in поставщик
            if mol is not None
        }
    нативная = next(строка for строка in строки if rmsd[строка["mol_id"]] == 0.0)

    assert float(нативная["ifp_tanimoto"]) == pytest.approx(0.588, abs=0.001)
    # Метрика обязана различать позы: константа по набору ничего не измеряет.
    assert len({строка["ifp_tanimoto"] for строка in строки}) > 1


def test_чужой_пакет_мишени_отвергается(прогон: Path, tmp_path: Path, target_json: Path) -> None:
    """Отпечатки разных режимов протонирования несравнимы: считать нельзя."""
    чужой = tmp_path / "6tgu"
    shutil.copytree(target_json.parent, чужой)
    пакет = json.loads((чужой / "target.json").read_text(encoding="utf-8"))
    пакет["protonation"] = "implicit-prolif"
    (чужой / "target.json").write_text(
        json.dumps(пакет, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(MetricsError, match="другом пакете мишени"):
        compute_run_metrics(прогон, чужой / "target.json")


def test_условие_доезжает_из_паспорта_в_метрики_и_реестр(
    прогон: Path, target_json: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Сквозная проверка колонки `condition`: паспорт → таблица метрик → реестр.

    Сравнение условий читает эту колонку в реестре, а заполняется она за три шага
    в трёх модулях. Проверять их порознь недостаточно: 24.08 колонка была пуста
    именно потому, что паспорт её не нёс, хотя каждый шаг в отдельности работал.

    Прогон копируется: фикстура общая на модуль, и метка в её паспорте протекла бы
    в соседние тесты.
    """
    # Каталог берётся у `tmp_path_factory` с латинским именем, а не у `tmp_path`:
    # последний называется по имени теста, имена тестов здесь русские, и путь
    # с кириллицей RDKit на Windows не открывает — «Bad input file».
    runs_dir = tmp_path_factory.mktemp("runs")
    копия = runs_dir / прогон.name
    shutil.copytree(прогон, копия)

    record_condition(копия, "baseline")
    compute_run_metrics(копия, target_json)

    assert {строка["condition"] for строка in read_metrics(копия)} == {"baseline"}
    assert describe_run(копия, runs_dir)["condition"] == "baseline"
