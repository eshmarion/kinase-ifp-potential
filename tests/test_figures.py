"""Тесты рисунков раздела «Результаты».

Фикстура — игрушечный прогон из четырёх поз с `rmsd_to_ref` 0.0, 0.8, 1.5 и 3.0:
по одному значению в каждый бин и два в первый. Каталог назван `figures`, а не
`runs`: шаблон `runs/` в `.gitignore` срабатывает на любом уровне вложенности,
и фикстура просто не попала бы в репозиторий.

Пиксели не сравниваются. Проверяется поведение: что группировка выбрана по данным,
что файл появился и не пуст, и что отсутствие контрольных метрик даёт исключение
с названной причиной, а не пустую картинку.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from evaluation.figures import (
    AXIS_LABEL_PT,
    BLOCK_TEXT_PT,
    FIGURE_LIBRARIES,
    GROUPING_CONDITION,
    GROUPING_RMSD,
    GROUPING_SELECTION,
    MIN_FONT_SIZE_PT,
    SELECTION_TOP_SUFFIX,
    TICK_LABEL_PT,
    FiguresError,
    collect_figure_data,
    figure_base_composition,
    figure_control,
    figure_ifp_map,
    figure_pipeline,
    figure_pose_scores,
    figure_refinement,
    figure_selection_size,
    figure_similarity,
    full_caption,
    groupings,
    runs_with_metrics,
    write_environment,
    write_figure_facts,
    write_panel_sizes,
    write_readme,
)
from experiments.layout import FIGURES_ENVIRONMENT_JSON, METRICS_CSV, RUN_JSON
from experiments.run_io import read_rmsd
from kinase_ifp.config import FIGURES_DIR, RMSD_BIN_LABELS

ФИКСТУРА = Path(__file__).parent / "fixtures" / "figures" / "poses"


@pytest.fixture()
def данные() -> pd.DataFrame:
    return collect_figure_data([ФИКСТУРА])


def test_rmsd_читается_из_sdf() -> None:
    значения = read_rmsd(ФИКСТУРА)

    assert значения == {
        "fix-run-0001": 0.0,
        "fix-run-0002": 0.8,
        "fix-run-0003": 1.5,
        "fix-run-0004": 3.0,
    }


def test_метрики_и_rmsd_склеиваются(данные: pd.DataFrame) -> None:
    assert len(данные) == 4
    assert данные["source"].unique().tolist() == ["local-poses"]
    # rmsd_to_ref в формате таблицы метрик отсутствует и приходит только из SDF.
    assert данные["rmsd_to_ref"].notna().all()


def test_прогон_без_паспорта_не_рисуется(tmp_path: Path) -> None:
    (tmp_path / "metrics_per_molecule.csv").write_text("mol_id\n", encoding="utf-8")

    with pytest.raises(Exception, match="run.json"):
        collect_figure_data([tmp_path])


def test_при_одном_условии_остаётся_разбивка_по_бинам_rmsd(данные: pd.DataFrame) -> None:
    разбивки = groupings(данные)

    assert [разбивка.name for разбивка in разбивки] == [GROUPING_RMSD]
    assert разбивки[0].order == RMSD_BIN_LABELS
    assert разбивки[0].labels.tolist() == [
        RMSD_BIN_LABELS[0],
        RMSD_BIN_LABELS[0],
        RMSD_BIN_LABELS[1],
        RMSD_BIN_LABELS[2],
    ]


def test_условия_не_вытесняют_разбивку_по_rmsd(данные: pd.DataFrame) -> None:
    # Картинка по RMSD нужна в тексте и тогда, когда появились
    # условия, — она доказывает, что метрика вообще различает позы.
    данные = данные.copy()
    данные["condition"] = ["baseline", "baseline", "rerank_top20", "rerank_top20"]

    разбивки = groupings(данные)

    assert [разбивка.name for разбивка in разбивки] == [GROUPING_CONDITION, GROUPING_RMSD]
    assert разбивки[0].order == ("baseline", "rerank_top20")


def test_при_двух_разбивках_рисунок_получает_две_панели(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    данные = данные.copy()
    данные["condition"] = ["baseline", "baseline", "rerank_top20", "rerank_top20"]
    файл = tmp_path / "fig1.png"

    entry = figure_similarity(данные, файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert GROUPING_CONDITION in entry.caption and GROUPING_RMSD in entry.caption


def test_без_условий_и_без_rmsd_группировать_нечем(данные: pd.DataFrame) -> None:
    данные = данные.copy()
    данные["rmsd_to_ref"] = None

    with pytest.raises(FiguresError, match="группировать нечем"):
        groupings(данные)


def test_рисунок_сходства_создаётся(данные: pd.DataFrame, tmp_path: Path) -> None:
    файл = tmp_path / "fig1.png"

    entry = figure_similarity(данные, файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ("fix-run",)
    # Подпись обязана называть разбивку: рисунок читают отдельно от кода.
    assert GROUPING_RMSD in entry.caption


def test_контрольные_метрики_без_данных_дают_ошибку(данные: pd.DataFrame, tmp_path: Path) -> None:
    файл = tmp_path / "fig2.png"

    with pytest.raises(FiguresError, match="source=diffsbdd"):
        figure_control(данные, файл)

    assert not файл.exists(), "пустой рисунок создаваться не должен"


def test_контрольные_метрики_рисуются_когда_колонки_заполнены(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    данные = данные.copy()
    данные["qed"] = [0.71, 0.65, 0.58, 0.44]
    данные["sa_score"] = [2.1, 2.4, 3.0, 3.6]
    данные["n_heavy_atoms"] = [24, 22, 26, 19]
    файл = tmp_path / "fig2.png"

    entry = figure_control(данные, файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ("fix-run",)


def test_схема_пайплайна_строится_без_данных(tmp_path: Path) -> None:
    файл = tmp_path / "fig3.png"

    entry = figure_pipeline(файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ()


def test_состав_базы_строится_без_прогонов(tmp_path: Path) -> None:
    """Рисунок 4 собирается из выгрузки KLIFS и пакета мишени, прогоны ему не нужны."""
    файл = tmp_path / "fig4.png"

    entry = figure_base_composition(файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ()
    # Поголовья панелей у него нет намеренно: `panels.csv` считает молекулы прогонов,
    # а здесь по осям структуры базы. Числа подписи подтверждает `klifs_base.csv`.
    assert entry.panels == ()


def _таблица_уточнения(путь: Path, пары: list[tuple[float, float]]) -> None:
    """Кладёт таблицу оптимизации поз с нужными колонками — как её пишет optimize_poses."""
    строки = ["mol_id,score_before,score_after,rmsd_shift"]
    строки += [f"m{номер},{до},{после},0.5" for номер, (до, после) in enumerate(пары)]
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_text(chr(10).join(строки) + chr(10), encoding="utf-8")


def _результаты_с_уточнением(корень: Path) -> Path:
    """Каталог выхода этапа с одной парой «опыт и контроль»."""
    результаты = корень / "results"
    _таблица_уточнения(
        результаты / "pose_optimization-2026-08-23-6tgu-s0-n100-prep-all7-clash.csv",
        [(0.4, 0.6), (0.5, 0.5), (0.3, 0.7)],
    )
    _таблица_уточнения(
        результаты / "pose_optimization-2026-08-23-6tgu-s0-n100-prep-all7-control-clash.csv",
        [(0.4, 0.4), (0.5, 0.4), (0.3, 0.3)],
    )
    return результаты


def test_уточнение_позы_рисуется_из_выхода_этапа(tmp_path: Path) -> None:
    """Рисунок 5 берёт таблицы `results/`, а не папки прогонов.

    Уточнение считается после прогона, и его результат живёт в выходе этапа: в папке
    прогона таких колонок нет вовсе.
    """
    результаты = _результаты_с_уточнением(tmp_path)
    файл = tmp_path / "fig5.png"

    entry = figure_refinement(файл, results_dir=результаты)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ("2026-08-23-6tgu-s0-n100-prep",)
    # Панели обязаны быть объявлены: `run_ids` непуст, значит поголовье молекул
    # попадает в `panels.csv` и становится источником чисел подписи.
    assert entry.panels == (("Потенциал работы", 3), ("Отрицательный контроль", 3))
    assert "вырос у 2 молекул из 3" in entry.caption


def test_уточнение_без_таблиц_отказывается(tmp_path: Path) -> None:
    """Пустое поле точек утверждало бы, что уточнения не было вовсе."""
    пусто = tmp_path / "results"
    пусто.mkdir()

    with pytest.raises(FiguresError, match="нет таблиц уточнения позы"):
        figure_refinement(tmp_path / "fig5.png", results_dir=пусто)


def test_уточнение_без_контроля_отказывается(tmp_path: Path) -> None:
    """Опыт без контроля — половина рисунка и всё его содержание разом.

    Правая панель существует затем, чтобы прирост нельзя было объяснить самим
    шевелением позы; нарисовать левую без правой значит предъявить утверждение
    без того, с чем оно сравнивается.
    """
    результаты = tmp_path / "results"
    _таблица_уточнения(
        результаты / "pose_optimization-2026-08-23-6tgu-s0-n100-prep-all7-clash.csv",
        [(0.4, 0.6)],
    )

    with pytest.raises(FiguresError, match="нет контроля"):
        figure_refinement(tmp_path / "fig5.png", results_dir=результаты)


def test_отбор_и_размер_рисуются_из_помолекулярной_выгрузки(tmp_path: Path) -> None:
    """Рисунок 6 берёт ту же выгрузку и то же правило топа, что проверка отбора."""
    результаты = tmp_path / 'results'
    результаты.mkdir()
    строки = ['run_id,mol_id,score_all7,score_scoring6,tanimoto,key_hbond,n_heavy']
    # Десять молекул: топ — доля 0.2, то есть две с наибольшим скором.
    for номер in range(10):
        скор = 0.9 - номер * 0.05
        строки.append(
            f'run-a,m{номер},{скор:.2f},{скор:.2f},{0.8 - номер * 0.03:.2f},1,{20 + номер}'
        )
    (результаты / 'klifs_rules_rescore_molecules.csv').write_text(
        '\n'.join(строки) + '\n', encoding='utf-8'
    )
    файл = tmp_path / 'fig6.png'

    entry = figure_selection_size(файл, results_dir=результаты)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ('run-a',)
    # Поголовье панели — все молекулы, а не сумма групп: группы не пересекаются,
    # но число в подписи должно совпадать с числом строк выгрузки.
    assert entry.panels == (('Танимото к эталону', 10), ('Число тяжёлых атомов', 10))
    assert 'отобрано 2' in entry.caption


def test_отбор_без_выгрузки_отказывается(tmp_path: Path) -> None:
    """Без помолекулярной выгрузки сравнивать отобранное с отбракованным нечем."""
    пусто = tmp_path / 'results'
    пусто.mkdir()

    with pytest.raises(FiguresError, match='нет помолекулярной выгрузки'):
        figure_selection_size(tmp_path / 'fig6.png', results_dir=пусто)

def _точки_поз(корень: Path) -> Path:
    """Каталог выхода этапа с точками двух мишеней и нативной позой у каждой."""
    результаты = корень / 'results'
    результаты.mkdir(exist_ok=True)
    строки = ['run_id,pdb_id,mol_id,score_all7,rmsd_to_ref,pose_kind']
    for мишень in ('3war', '6tgu'):
        строки.append(f'{мишень}-r0,{мишень},native,1.0,0.0,native')
        for номер in range(4):
            скор = 0.8 - номер * 0.15
            строки.append(
                f'{мишень}-r0,{мишень},m{номер},{скор:.2f},{0.5 + номер:.2f},conformer'
            )
    (результаты / 'klifs_rules_rescore_points.csv').write_text(
        '\n'.join(строки) + '\n', encoding='utf-8'
    )
    return результаты


def test_различение_поз_рисуется_по_точкам(tmp_path: Path) -> None:
    """Рисунок 7 строится из помолекулярных точек: в сводках лежат одни медианы."""
    результаты = _точки_поз(tmp_path)
    файл = tmp_path / 'fig7.png'

    entry = figure_pose_scores(файл, results_dir=результаты)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ('3war-r0', '6tgu-r0')
    assert entry.panels == (('3war', 5), ('6tgu', 5))


def test_различение_поз_без_точек_отказывается(tmp_path: Path) -> None:
    """Медианы по наборам облако точек не заменяют, и рисовать из них нечего."""
    пусто = tmp_path / 'results'
    пусто.mkdir()

    with pytest.raises(FiguresError, match='нет точек наборов поз'):
        figure_pose_scores(tmp_path / 'fig7.png', results_dir=пусто)

def test_карта_отпечатка_строится_из_пакета_мишени(tmp_path: Path) -> None:
    """Рисунок 8 считает биты на лету: готовых отпечатков на диске нет."""
    файл = tmp_path / 'fig8.png'

    entry = figure_ifp_map(файл)

    assert файл.is_file() and файл.stat().st_size > 0
    assert entry.run_ids == ()
    # Карта показывает мишень и эталон, а не прогоны, поэтому поголовья панелей
    # у неё нет — как у состава базы и схемы конвейера.
    assert entry.panels == ()
    assert 'бит' in entry.caption


def test_карта_отпечатка_без_пакета_отказывается(tmp_path: Path) -> None:
    """Пустая сетка сказала бы, что взаимодействий нет вовсе."""
    пусто = tmp_path / 'targets'
    пусто.mkdir()

    with pytest.raises(FiguresError, match='нет пакета мишени'):
        figure_ifp_map(tmp_path / 'fig8.png', targets_dir=пусто)

def test_опись_фактов_несёт_числа_подписи(tmp_path: Path) -> None:
    """Число подписи сверяется наравне с числом абзаца: подпись уходит в текст.

    19.09 сверка нашла в подписях собранных рисунков поголовье точек облака
    и число молекул с выросшим скором — машинного источника у них не было нигде,
    потому что panels.csv описывает только объёмы панелей.
    """
    результаты = _результаты_с_уточнением(tmp_path)
    рисунок = figure_refinement(tmp_path / 'fig.png', results_dir=результаты)

    файл = write_figure_facts([рисунок], tmp_path / 'figures')

    строки = файл.read_text(encoding='utf-8').splitlines()
    assert строки[0] == 'figure,fact,value'
    assert '1,скор вырос в опыте,2' in строки
    # Номер в описи — порядок сборки, тот же, что в подписи и в panels.csv.
    assert all(строка.startswith('1,') for строка in строки[1:])


def test_рисунок_без_чисел_в_подписи_описи_не_даёт(tmp_path: Path) -> None:
    """У схемы конвейера подпись чисел не называет, и строк в описи у неё нет."""
    схема = figure_pipeline(tmp_path / 'fig3.png')

    файл = write_figure_facts([схема], tmp_path / 'figures')

    assert схема.facts == ()
    assert файл.read_text(encoding='utf-8').splitlines() == ['figure,fact,value']

def test_состав_базы_без_выгрузки_отказывается(tmp_path: Path) -> None:
    """Пустая гистограмма хуже отказа: рисунок сказал бы, что база пуста."""
    пусто = tmp_path / "klifs"
    пусто.mkdir()

    with pytest.raises(FiguresError, match="состав базы не измерить"):
        figure_base_composition(tmp_path / "fig4.png", klifs_dir=пусто)


def test_readme_называет_подпись_и_прогоны(данные: pd.DataFrame, tmp_path: Path) -> None:
    entries = [
        figure_similarity(данные, tmp_path / "fig1.png"),
        figure_pipeline(tmp_path / "fig3.png"),
    ]

    путь = write_readme(entries, tmp_path)
    текст = путь.read_text(encoding="utf-8")

    assert "fig1.png" in текст and "fig3.png" in текст
    assert "fix-run" in текст
    # У схемы прогонов нет, и это должно быть сказано словами, а не пустотой.
    assert "данные не используются" in текст


def test_подпись_оформлена_по_положению(данные: pd.DataFrame, tmp_path: Path) -> None:
    """Положение ФББ, Приложение 3, п. 1.8: «Рисунок N – Название с заглавной буквы»."""
    entry = figure_similarity(данные, tmp_path / "fig1.png")

    подпись = full_caption(entry, 1)

    assert подпись.startswith("Рисунок 1 – ")
    название = подпись[len("Рисунок 1 – ")]
    assert название.isupper(), f"название начинается со строчной: {подпись}"
    # Точка перед пояснением ставится ровно одна: title хранится без неё.
    assert " – . " not in подпись and ".." not in подпись


def test_нумерация_идёт_по_собранным_рисункам(данные: pd.DataFrame, tmp_path: Path) -> None:
    """Несобранный рисунок 2 не должен оставлять дыру в нумерации текста.

    Схема лежит в файле `fig3_*`, но если контрольные метрики не построены
    (нет прогона source=diffsbdd), в тексте она обязана быть «Рисунком 2».
    """
    entries = [
        figure_similarity(данные, tmp_path / "fig1.png"),
        figure_pipeline(tmp_path / "fig3.png"),
    ]

    текст = write_readme(entries, tmp_path).read_text(encoding="utf-8")

    assert "Рисунок 2 – Схема расчёта отпечатка взаимодействий" in текст
    assert "Рисунок 3 –" not in текст, "пропущен номер 2 — дефект оформления"
    assert "(Рисунок 1)" in текст and "(Рисунок 2)" in текст


def test_нумерация_при_трёх_рисунках_ставит_схему_третьей(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    """Когда контрольные метрики построены, схема становится Рисунком 3.

    Парный к тесту выше: тот охраняет случай без рисунка 2, этот — случай с ним.
    Номер зависит от того, что собрано, а не от имени файла, и ссылки «(Рисунок N)»
    в тексте обязаны совпадать с подписями.
    """
    с_контролем = данные.assign(
        source="diffsbdd",
        selected=[0, 1, 0, 1],
        qed=[0.4, 0.6, 0.5, 0.7],
        sa_score=[3.0, 4.0, 3.5, 4.5],
        n_heavy_atoms=[20, 22, 21, 23],
    )
    entries = [
        figure_similarity(данные, tmp_path / "fig1.png"),
        figure_control(с_контролем, tmp_path / "fig2.png"),
        figure_pipeline(tmp_path / "fig3.png"),
    ]

    текст = write_readme(entries, tmp_path).read_text(encoding="utf-8")

    assert "Рисунок 3 – Схема расчёта отпечатка взаимодействий" in текст
    assert "(Рисунок 3)" in текст


def test_группа_весь_набор_включает_отобранные(данные: pd.DataFrame) -> None:
    """«diffsbdd» и «diffsbdd, топ» вложены, как соседние колонки таблицы «было/стало».

    Если бы группы не пересекались, «весь набор» означал бы «всё, кроме топа»,
    и медиана на рисунке разошлась бы с медианой в таблице при тех же данных.
    """
    с_отбором = данные.assign(source="diffsbdd", selected=[0, 1, 0, 1])

    разбивка = next(g for g in groupings(с_отбором) if g.name == GROUPING_SELECTION)

    assert разбивка.mask("diffsbdd").sum() == 4
    assert разбивка.mask("diffsbdd" + SELECTION_TOP_SUFFIX).sum() == 2


def test_объём_панели_не_считает_отобранные_дважды(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    """Подпись называет поголовье молекул панели, а «весь набор» и «топ» вложены.

    Сумма размеров ящиков дала бы 6 при четырёх молекулах. На прогонах 6tgu та же
    ошибка давала 234 вместо 195, и сверка чисел не находила это число в данных:
    поголовья такого размера не существует.
    """
    с_отбором = данные.assign(source="diffsbdd", selected=[0, 1, 0, 1])

    entry = figure_similarity(с_отбором, tmp_path / "fig1.png")

    assert f"{GROUPING_SELECTION} — 4" in entry.caption
    assert f"{GROUPING_SELECTION} — 6" not in entry.caption


def test_объёмы_панелей_ложатся_в_машинный_файл(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    """`panels.csv` — источник поголовья для сверки чисел, а не украшение README.

    Строк ровно столько, сколько панелей, и перевод строки — LF:
    файл лежит в git рядом с прогонами, и CRLF развёл бы его с тем же файлом,
    сохранённым другой машиной.
    """
    с_отбором = данные.assign(source="diffsbdd", selected=[0, 1, 0, 1])
    entry = figure_similarity(с_отбором, tmp_path / "fig1.png")

    файл = write_panel_sizes([entry], tmp_path)
    сырое = файл.read_bytes().decode("utf-8")

    assert chr(13) + chr(10) not in сырое
    строки = сырое.splitlines()
    assert строки[0] == "figure,panel,molecules"
    assert len(строки) == 1 + len(entry.panels)
    assert f"1,{GROUPING_SELECTION},4" in строки


def test_поголовье_панелей_рисунка_2_попадает_в_машинный_файл(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    """Рисунок контрольных метрик обязан описать свои панели, а не только рисунок 1.

    До 16.09 `panels.csv` состоял из трёх строк, все с `figure=1`:
    `figure_control` создавал запись без панелей, и сверка по рисунку 2
    не проверяла ничего — поголовье из подписи подтверждать было нечем.
    """
    данные = данные.copy()
    данные["qed"] = [0.71, 0.65, 0.58, 0.44]
    данные["sa_score"] = [2.1, 2.4, 3.0, 3.6]
    данные["n_heavy_atoms"] = [24, 22, 26, 19]

    entry = figure_control(данные, tmp_path / "fig2.png")
    строки = write_panel_sizes([entry], tmp_path).read_text(encoding="utf-8").splitlines()

    assert entry.panels, "рисунок построен, а панелей не объявил"
    assert len(строки) == 1 + len(entry.panels)
    for имя, сколько in entry.panels:
        assert f"1,{имя},{сколько}" in строки
        assert f"{имя} — {сколько}" in entry.caption


def test_рисунок_без_данных_строк_поголовья_не_даёт(
    данные: pd.DataFrame, tmp_path: Path
) -> None:
    """Строка есть у каждого рисунка, построенного по прогонам, и только у него.

    Схема пайплайна данных не использует, панелей у неё нет по существу, и пустая
    строка в описи означала бы, что поголовье забыли, а не что его не бывает.
    """
    с_метриками = данные.assign(
        source="diffsbdd", selected=[0, 1, 0, 1], qed=[0.71, 0.65, 0.58, 0.44],
        sa_score=[2.1, 2.4, 3.0, 3.6], n_heavy_atoms=[24, 22, 26, 19],
    )
    рисунки = [
        figure_similarity(с_метриками, tmp_path / "fig1.png"),
        figure_control(с_метриками, tmp_path / "fig2.png"),
        figure_pipeline(tmp_path / "fig3.png"),
    ]

    строки = write_panel_sizes(рисунки, tmp_path).read_text(encoding="utf-8").splitlines()[1:]
    номера = {строка.split(",")[0] for строка in строки}

    assert номера == {"1", "2"}
    assert all(рисунок.panels for рисунок in рисунки if рисунок.run_ids)


def test_источники_не_смешиваются_в_один_ящик(данные: pd.DataFrame) -> None:
    """`local-poses` и `diffsbdd` — разные утверждения, и групп у них разные."""
    смесь = pd.concat(
        [
            данные.assign(source="local-poses", selected=[0, 1, 0, 0]),
            данные.assign(source="diffsbdd", selected=[1, 0, 0, 0]),
        ],
        ignore_index=True,
    )

    разбивка = next(g for g in groupings(смесь) if g.name == GROUPING_SELECTION)

    assert разбивка.order == (
        "local-poses",
        "local-poses" + SELECTION_TOP_SUFFIX,
        "diffsbdd",
        "diffsbdd" + SELECTION_TOP_SUFFIX,
    )
    assert разбивка.mask("local-poses").sum() == 4
    assert разбивка.mask("diffsbdd" + SELECTION_TOP_SUFFIX).sum() == 1


def test_контрольные_метрики_пропускают_группы_без_значений(данные: pd.DataFrame) -> None:
    """Рисунок 2 не строится по группе, где контрольные колонки пусты.

    Так выглядит `local-poses` рядом с `diffsbdd`: колонки qed там пусты
    по формату таблицы метрик, и ящик из пустоты был бы нарисован молча.
    """
    смесь = pd.concat(
        [
            данные.assign(source="local-poses", selected=[0, 1, 0, 0], qed=float("nan")),
            данные.assign(source="diffsbdd", selected=[0, 1, 0, 1], qed=[0.4, 0.6, 0.5, 0.7]),
        ],
        ignore_index=True,
    ).assign(sa_score=3.0, n_heavy_atoms=20)

    запись = figure_control(смесь, Path(tempfile.mkdtemp()) / "fig2.png")

    assert "diffsbdd" in запись.caption or запись.path.is_file()


def test_кегли_подписей_не_меньше_шести_пунктов() -> None:
    """Положение ФББ, п. 1.9: шрифт на рисунках не мельче 6 пт."""
    for кегль in (AXIS_LABEL_PT, TICK_LABEL_PT, BLOCK_TEXT_PT):
        assert кегль >= MIN_FONT_SIZE_PT


def test_названия_нет_внутри_картинки(данные: pd.DataFrame, tmp_path: Path) -> None:
    """Название живёт в подписи под рисунком, а не заголовком на самом рисунке.

    Иначе оно дублируется и расходится с подписью при первой же правке — а по
    положению (п. 1.8) обязательна именно подпись под рисунком.
    """
    entry = figure_similarity(данные, tmp_path / "fig1.png")

    assert entry.title and entry.title not in entry.caption


def test_прогоны_без_метрик_в_список_не_попадают(tmp_path: Path) -> None:
    (tmp_path / "пусто").mkdir()
    (tmp_path / "index.csv").write_text(
        "run_id,date,pdb_id,source,condition,guidance_scale,seed,n_returned,"
        "platform,status,path\nx,2026-08-24,6tgu,local-poses,,0.0,0,4,local-cpu,generated,пусто\n",
        encoding="utf-8",
    )

    assert runs_with_metrics(tmp_path) == []


def test_запись_окружения_называет_чем_нарисовано(tmp_path: Path) -> None:
    """Версия рисовалки в паспорт прогона не пишется — значит пишется сюда.

    Проверяется не «файл появился», а что в нём стоит версия **установленного**
    `matplotlib`: запись, разошедшаяся с тем, чем рисовали, хуже её отсутствия —
    ровно по этой причине паспорт прогона не приписывает версии задним числом.
    """
    import json
    from importlib import metadata

    путь = write_environment(tmp_path / "рисунки")

    assert путь.name == FIGURES_ENVIRONMENT_JSON
    записано = json.loads(путь.read_text(encoding="utf-8"))
    assert записано["matplotlib"] == metadata.version("matplotlib")
    assert записано["python"].count(".") == 2


def test_README_указывает_на_запись_окружения(данные: pd.DataFrame, tmp_path: Path) -> None:
    """Иначе файл лежит рядом и о нём никто не знает."""
    entry = figure_similarity(данные, tmp_path / "fig1.png")

    текст = write_readme([entry], tmp_path).read_text(encoding="utf-8")

    assert FIGURES_ENVIRONMENT_JSON in текст


def test_рисунки_строятся_по_одной_мишени(tmp_path: Path) -> None:
    """Панели группируются по условию, поэтому две киназы склеились бы в один ящик.

    17.09 это измерено на живых данных: набор поз получал 195 молекул вместо 100
    и медиану 0.412 вместо 0.458 — две разные киназы под одной подписью `local-poses`.
    Фильтр по мишени стоит здесь, а не в реестре: какие киназы показывает работа —
    вопрос состава результатов, и решают его авторы разделов.
    """
    прогоны = (("2026-08-24-6tgu-s0-n100", "6tgu"), ("2026-09-16-3war-s0-n100", "3war"))
    for run_id, pdb_id in прогоны:
        папка = tmp_path / run_id
        папка.mkdir()
        (папка / RUN_JSON).write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "source": "local-poses",
                    "created_utc": "2026-08-24T10:00:00Z",
                    "target": {"pdb_id": pdb_id},
                    "sampling": {"n_requested": 100, "n_returned": 100, "seed": 0},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (папка / METRICS_CSV).write_text("mol_id\n", encoding="utf-8")
    шапка = (
        "run_id,date,pdb_id,source,condition,guidance_scale,seed,n_returned,"
        "platform,status,path"
    )
    строки_реестра = (
        "2026-08-24-6tgu-s0-n100,2026-08-24,6tgu,local-poses,,0.0,0,100,local-cpu,"
        "scored,2026-08-24-6tgu-s0-n100",
        "2026-09-16-3war-s0-n100,2026-09-16,3war,local-poses,,0.0,0,100,local-cpu,"
        "scored,2026-09-16-3war-s0-n100",
    )
    (tmp_path / "index.csv").write_text(
        "\n".join((шапка, *строки_реестра)) + "\n", encoding="utf-8"
    )

    по_умолчанию = [папка.name for папка in runs_with_metrics(tmp_path)]
    assert по_умолчанию == ["2026-08-24-6tgu-s0-n100"]

    все_мишени = [папка.name for папка in runs_with_metrics(tmp_path, pdb_id=None)]
    assert все_мишени == ["2026-08-24-6tgu-s0-n100", "2026-09-16-3war-s0-n100"]


def test_каталог_рисунков_не_отстал_от_кода() -> None:
    """Выход этапа на диске собран **текущим** кодом, а не прошлой его версией.

    Единственный тест этого файла, который смотрит в настоящий каталог, а не во
    временный. Остальные зовут функции на `tmp_path` и поэтому проходят даже тогда,
    когда рядом с рисунками курсовой не лежит ничего из того, что эти функции пишут:
    16.09 запись окружения появилась в коде в 20:54, а рисунки собирались в 14:55,
    и полгода такого расхождения не заметил бы никто — тот же класс, что «реестр
    отстаёт от диска» и «правило завели, на соседнюю полку
    оно не распространилось).

    Сверяется состав записи, а не значения версий. Требовать совпадения версии
    с установленной значило бы красить `pytest` у всех сразу после обновления образа,
    в том числе у того, кто рисунков не трогал; пересборка после смены версии — повод
    для отдельного решения, а не для падающего теста.
    """
    import json

    путь = FIGURES_DIR / FIGURES_ENVIRONMENT_JSON
    assert путь.is_file(), (
        f"{путь} нет: рисунки в репозитории собраны кодом, который эту запись ещё "
        "не делал; пересоберите: python scripts/make_figures.py"
    )

    записано = json.loads(путь.read_text(encoding="utf-8"))
    assert set(записано) == {"python", *FIGURE_LIBRARIES}, (
        "состав записи разошёлся с FIGURE_LIBRARIES; "
        "пересоберите: python scripts/make_figures.py"
    )

    readme = (FIGURES_DIR / "README.md").read_text(encoding="utf-8")
    assert FIGURES_ENVIRONMENT_JSON in readme, (
        "README рядом с рисунками не знает про запись окружения — значит собран "
        "прошлой версией кода; пересоберите: python scripts/make_figures.py"
    )
