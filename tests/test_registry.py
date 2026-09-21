"""Тесты реестра прогонов `runs/index.csv`.

Реестр производный: он обязан восстанавливаться по содержимому `runs/` и не обязан
помнить ничего сверх того, что лежит на диске. Поэтому проверяется именно соответствие
строки тому, что в папке, — включая случай испорченной папки, которая должна попасть
в реестр со статусом `failed`, а не исчезнуть из него.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.registry import (
    INDEX_COLUMNS,
    INDEX_CSV,
    MOLECULES_IN_GIT,
    MOLECULES_LOCAL_ONLY,
    STATUS_FAILED,
    STATUS_GENERATED,
    STATUS_SCORED,
    RegistryError,
    describe_run,
    main_line_runs,
    read_index,
    rebuild_index,
    refresh_run,
    register_run,
    scored_runs,
)
from experiments.runs import MOLECULES_SDF, RUN_JSON, record_main_line
from kinase_ifp.config import MAIN_TARGET_PDB_ID

ПАСПОРТ = {
    "run_id": "2026-08-23-6tgu-s0-n5",
    "created_utc": "2026-08-23T18:00:00Z",
    "target": {"pdb_id": "6tgu", "klifs_structure_id": 12448},
    "source": "diffsbdd",
    "sampling": {"n_requested": 5, "n_returned": 4, "seed": 0, "guidance_scale": 0.0},
    "platform": "modal-gpu",
}


def создать_прогон(
    runs_dir: Path,
    run_id: str = "2026-08-23-6tgu-s0-n5",
    *,
    паспорт: dict[str, object] | None = None,
    ranking: bool = False,
    условия: tuple[str, ...] = (),
) -> Path:
    """Собирает папку прогона: молекулы, паспорт и, если нужно, ранжирование."""
    папка = runs_dir / run_id
    папка.mkdir(parents=True)
    данные = dict(паспорт if паспорт is not None else ПАСПОРТ)
    данные["run_id"] = run_id
    (папка / RUN_JSON).write_text(json.dumps(данные), encoding="utf-8")
    (папка / MOLECULES_SDF).write_text("", encoding="utf-8")
    if ranking:
        (папка / "ranking.csv").write_text("mol_id,ifp_score\n", encoding="utf-8")
    if условия:
        строки = "\n".join(f"m{номер},{имя}" for номер, имя in enumerate(условия))
        (папка / "metrics_per_molecule.csv").write_text(
            f"mol_id,condition\n{строки}\n", encoding="utf-8"
        )
    return папка


def test_строка_собирается_из_паспорта(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)

    строка = describe_run(папка, tmp_path)

    assert строка["run_id"] == "2026-08-23-6tgu-s0-n5"
    assert строка["source"] == "diffsbdd"
    assert строка["pdb_id"] == "6tgu"
    assert строка["date"] == "2026-08-23"
    assert строка["n_returned"] == 4
    assert строка["platform"] == "modal-gpu"
    assert строка["status"] == STATUS_GENERATED


def test_путь_записан_относительно_runs(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)

    assert describe_run(папка, tmp_path)["path"] == "2026-08-23-6tgu-s0-n5"


def test_ranking_переводит_прогон_в_scored(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path, ranking=True)

    assert describe_run(папка, tmp_path)["status"] == STATUS_SCORED


def test_условия_берутся_из_метрик(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path, условия=("baseline", "rerank-top20", "baseline"))

    assert describe_run(папка, tmp_path)["condition"] == "baseline;rerank-top20"


def test_без_метрик_условие_пусто(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)

    assert describe_run(папка, tmp_path)["condition"] == ""


def test_папка_без_молекул_остаётся_прогоном(tmp_path: Path) -> None:
    """Молекулы могут остаться у автора, и это не делает эксперимент несостоявшимся.

    До 18.09 такая папка получала `failed` и пустые поля, то есть выпадала из реестра,
    а следом из таблицы «было/стало» и из рисунков — молча, потому что никто не падал.
    Решение №31 разводит два случая: нет паспорта — прогона нет; нет молекул — есть
    прогон, по которому нельзя пересчитать скор и метрики.
    """
    папка = создать_прогон(tmp_path)
    (папка / MOLECULES_SDF).unlink()

    строка = describe_run(папка, tmp_path)

    assert строка["status"] != STATUS_FAILED
    assert строка["molecules"] == MOLECULES_LOCAL_ONLY
    assert строка["source"], "поля паспорта обнулены, хотя паспорт на месте"


def test_молекулы_рядом_отмечены_в_реестре(tmp_path: Path) -> None:
    """Обратный случай: по прогону можно пересчитать всё, и реестр это говорит."""
    папка = создать_прогон(tmp_path)

    assert describe_run(папка, tmp_path)["molecules"] == MOLECULES_IN_GIT


def test_папка_без_паспорта_помечается_failed(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)
    (папка / RUN_JSON).unlink()

    assert describe_run(папка, tmp_path)["status"] == STATUS_FAILED


def test_паспорт_без_source_помечается_failed(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path, паспорт={"run_id": "x", "created_utc": "2026-08-23"})

    assert describe_run(папка, tmp_path)["status"] == STATUS_FAILED


def test_реестр_собирает_все_папки(tmp_path: Path) -> None:
    создать_прогон(tmp_path, "2026-08-23-6tgu-s0-n5")
    создать_прогон(tmp_path, "2026-08-23-6tgu-s0-n100", ranking=True)

    строки = rebuild_index(tmp_path)

    assert [строка["run_id"] for строка in строки] == [
        "2026-08-23-6tgu-s0-n100",
        "2026-08-23-6tgu-s0-n5",
    ]


def test_заголовок_реестра_задан_одним_местом(tmp_path: Path) -> None:
    создать_прогон(tmp_path)
    rebuild_index(tmp_path)

    with (tmp_path / INDEX_CSV).open(encoding="utf-8", newline="") as файл:
        assert next(csv.reader(файл)) == list(INDEX_COLUMNS)


def test_реестр_перечитывается(tmp_path: Path) -> None:
    создать_прогон(tmp_path)
    rebuild_index(tmp_path)

    строки = read_index(tmp_path / INDEX_CSV)

    assert len(строки) == 1
    assert строки[0]["source"] == "diffsbdd"


def test_пересборка_не_ломается_на_испорченной_папке(tmp_path: Path) -> None:
    создать_прогон(tmp_path, "2026-08-23-6tgu-s0-n5")
    (tmp_path / "мусор").mkdir()

    статусы = {строка["run_id"]: строка["status"] for строка in rebuild_index(tmp_path)}

    assert статусы == {"2026-08-23-6tgu-s0-n5": STATUS_GENERATED, "мусор": STATUS_FAILED}


def test_регистрация_добавляет_строку(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)

    register_run(папка, tmp_path)

    assert [строка["run_id"] for строка in read_index(tmp_path / INDEX_CSV)] == [папка.name]


def test_повторная_регистрация_отклоняется(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)
    register_run(папка, tmp_path)

    with pytest.raises(RegistryError, match="уже в реестре"):
        register_run(папка, tmp_path)


def test_отсутствующий_реестр_читается_пустым(tmp_path: Path) -> None:
    assert read_index(tmp_path / INDEX_CSV) == []


def test_пересборка_без_каталога_падает(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="нет каталога"):
        rebuild_index(tmp_path / "нет-такого")


def test_в_отчёт_идут_только_прогоны_со_скором(tmp_path: Path) -> None:
    """Сборка отчёта берёт из реестра `scored`: у `generated` нет ни метрик, ни ranking.csv.

    Колонка такого прогона в таблице «было/стало» вышла бы пустой, а пустая колонка
    в тексте курсовой неотличима от посчитанного нуля.
    """
    создать_прогон(tmp_path, "2026-08-24-6tgu-s0-n5", ranking=True, условия=("baseline",))
    создать_прогон(tmp_path, "2026-08-24-6tgu-s0-n100")
    rebuild_index(tmp_path)

    папки = scored_runs(tmp_path)

    assert [папка.name for папка in папки] == ["2026-08-24-6tgu-s0-n5"]
    assert папки[0].is_dir()


def test_без_реестра_отчёту_нечего_собирать(tmp_path: Path) -> None:
    """Нет реестра — пустой список, а не исключение: решает вызывающий (`build_report.py`)."""
    assert scored_runs(tmp_path) == []


def test_обновление_меняет_статус_не_плодя_строку(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)
    register_run(папка, tmp_path)
    (папка / "ranking.csv").write_text("mol_id,ifp_score", encoding="utf-8")

    строка = refresh_run(папка, tmp_path)

    assert строка["status"] == STATUS_SCORED
    строки = read_index(tmp_path / INDEX_CSV)
    assert len(строки) == 1
    assert строки[0]["status"] == STATUS_SCORED


def test_обновление_не_трогает_соседние_прогоны(tmp_path: Path) -> None:
    первый = создать_прогон(tmp_path, "2026-08-23-6tgu-s0-n5")
    второй = создать_прогон(tmp_path, "2026-08-23-6tgu-s0-n100")
    register_run(первый, tmp_path)
    register_run(второй, tmp_path)
    (второй / "ranking.csv").write_text("mol_id,ifp_score", encoding="utf-8")

    refresh_run(второй, tmp_path)

    статусы = {строка["run_id"]: строка["status"] for строка in read_index(tmp_path / INDEX_CSV)}
    assert статусы == {
        "2026-08-23-6tgu-s0-n5": STATUS_GENERATED,
        "2026-08-23-6tgu-s0-n100": STATUS_SCORED,
    }


def test_обновление_незаведённого_прогона_отклоняется(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path)

    with pytest.raises(RegistryError, match="нет в реестре"):
        refresh_run(папка, tmp_path)


def создать_считанный(
    runs_dir: Path,
    run_id: str,
    *,
    source: str,
    created: str,
    n_requested: int,
    pocket_mode: str | None = None,
    pdb_id: str | None = None,
) -> Path:
    """Прогон со статусом `scored`: паспорт, молекулы и ranking.csv."""
    паспорт = {
        **ПАСПОРТ,
        "run_id": run_id,
        "source": source,
        "created_utc": created,
        **(
            {"target": {**ПАСПОРТ.get("target", {}), "pdb_id": pdb_id}}
            if pdb_id
            else {}
        ),
        "sampling": {
            "n_requested": n_requested,
            "n_returned": n_requested,
            "seed": 0,
            **({"pocket_mode": pocket_mode} if pocket_mode else {}),
        },
    }
    папка = создать_прогон(runs_dir, run_id, паспорт=паспорт, ranking=True)
    return папка


def test_основная_линия_берёт_полный_прогон_а_не_свежую_пробу(tmp_path: Path) -> None:
    """Случай 24.08: самой свежей была проба из 20 молекул по вопросу В-13.

    Попав в таблицу основной колонкой, она представляла бы работу худшим
    из измеренных вариантов — и с межквартильным размахом по двадцати молекулам.
    """
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n20-pocketids2",
        source="diffsbdd", created="2026-08-24T09:42:17Z", n_requested=20,
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-08-24-6tgu-s0-n100-seed1"]


def test_метка_основного_перевешивает_полноту(tmp_path: Path) -> None:
    """Случай 18.09: прогон втрое полнее занял место того, по которому написан текст.

    Метка разводит два вопроса: какой прогон полнее — вопрос к данным,
    каким прогоном представлена работа — решение авторов.
    """
    помеченный = создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-09-18-6tgu-s0-n300",
        source="diffsbdd", created="2026-09-18T14:52:00Z", n_requested=300,
    )
    record_main_line(помеченный)
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-08-24-6tgu-s0-n100-seed1"]


def test_без_метки_правило_прежнее(tmp_path: Path) -> None:
    """Механизм ничего не меняет, пока метку никто не поставил."""
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-09-18-6tgu-s0-n300",
        source="diffsbdd", created="2026-09-18T14:52:00Z", n_requested=300,
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-09-18-6tgu-s0-n300"]


def test_снятая_метка_возвращает_прежнее_правило(tmp_path: Path) -> None:
    """Метку можно снять: она говорит о представлении работы, а не о том, чем считали."""
    помеченный = создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-09-18-6tgu-s0-n300",
        source="diffsbdd", created="2026-09-18T14:52:00Z", n_requested=300,
    )
    record_main_line(помеченный)
    record_main_line(помеченный, main_line=False)
    rebuild_index(tmp_path)

    assert [папка.name for папка in main_line_runs(tmp_path)] == ["2026-09-18-6tgu-s0-n300"]


def test_две_метки_на_одном_источнике_останавливают_отбор(tmp_path: Path) -> None:
    """Молчаливый выбор одного из двух помеченных был бы тем же дефектом заново."""
    первый = создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    второй = создать_считанный(
        tmp_path, "2026-09-18-6tgu-s0-n300",
        source="diffsbdd", created="2026-09-18T14:52:00Z", n_requested=300,
    )
    record_main_line(первый)
    record_main_line(второй)
    rebuild_index(tmp_path)

    with pytest.raises(RegistryError, match="два прогона одного источника"):
        main_line_runs(tmp_path)


def test_нечитаемый_паспорт_считанного_прогона_останавливает_отбор(tmp_path: Path) -> None:
    """Ветка заведена потому, что прежде она не исполнялась ни разу.

    Реестр производный, и расхождение с диском возможно: строка числит прогон
    посчитанным, а паспорт на диске уже не читается. Молча пропустить такой прогон
    значило бы собрать таблицу работы по неполному составу и не сказать об этом.
    """
    папка = создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    rebuild_index(tmp_path)
    # Реестр уже собран, поэтому строка остаётся, а паспорт перестаёт быть полным.
    (папка / "run.json").write_text('{"run_id": "", "source": ""}', encoding="utf-8")

    with pytest.raises(RegistryError, match="паспорт не читается"):
        main_line_runs(tmp_path)


def test_среди_равных_по_полноте_берётся_свежий(tmp_path: Path) -> None:
    создать_считанный(
        tmp_path, "2026-08-23-6tgu-s0-n100-prep",
        source="diffsbdd", created="2026-08-23T23:53:10Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-08-24-6tgu-s0-n100-seed1"]


def test_контрольный_способ_кармана_не_представляет_работу(tmp_path: Path) -> None:
    """Прогон на `pocket-ids` поставлен ради сравнения с таблицей, а не для неё.

    16.09 защита полнотой перестала работать: контроль сделан на тех же 100 молекулах,
    что и основной прогон, и по свежести вытеснил его — работа была бы представлена
    собственным отрицательным контролем. Полнота тут не помогает по построению:
    контроль ставят на том же объёме, иначе он не контроль.
    """
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-09-16-6tgu-s0-n100",
        source="diffsbdd", created="2026-09-16T14:15:00Z", n_requested=100,
        pocket_mode="pocket-ids",
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-08-24-6tgu-s0-n100-seed1"]


def test_источники_не_смешиваются(tmp_path: Path) -> None:
    """По одному прогону на источник: колонки «было/стало» сравнивают их между собой."""
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100",
        source="local-poses", created="2026-08-24T09:30:03Z", n_requested=100,
    )
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]

    assert отобранные == ["2026-08-24-6tgu-s0-n100", "2026-08-24-6tgu-s0-n100-seed1"]


def test_непосчитанные_прогоны_в_основную_линию_не_идут(tmp_path: Path) -> None:
    создать_прогон(tmp_path, "2026-08-24-6tgu-s0-n100")
    rebuild_index(tmp_path)

    assert main_line_runs(tmp_path) == []


def test_пустой_реестр_даёт_пустую_основную_линию(tmp_path: Path) -> None:
    assert main_line_runs(tmp_path) == []


def test_вторая_мишень_не_вытесняет_первую_из_таблицы(tmp_path: Path) -> None:
    """Прогон на другой киназе — отдельная колонка, а не замена основной.

    До 17.09 правило группировало прогоны по источнику и о мишени не знало: мишень
    в проекте была одна. Первый же набор поз на второй киназе вытеснил набор поз
    основной мишени — тот же источник, та же полнота, свежее дата, — и таблица
    показала бы числа чужой киназы под заголовком, который текст относит к 6tgu.
    Заметить подмену было бы нечем: имя мишени в таблицу не выводится.
    """
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100",
        source="local-poses", created="2026-08-24T10:00:00Z", n_requested=100,
        pdb_id="6tgu",
    )
    создать_считанный(
        tmp_path, "2026-09-17-6fnk-s0-n100",
        source="local-poses", created="2026-09-17T05:00:00Z", n_requested=100,
        pdb_id="6fnk",
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]
    assert отобранные == ["2026-08-24-6tgu-s0-n100", "2026-09-17-6fnk-s0-n100"]


def test_внутри_одной_мишени_правило_прежнее(tmp_path: Path) -> None:
    """Мишень вошла в ключ группировки, а не в ключ сравнения: проба по-прежнему отсеивается."""
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100-seed1",
        source="diffsbdd", created="2026-08-24T08:33:46Z", n_requested=100,
        pdb_id="6tgu",
    )
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n20-proba",
        source="diffsbdd", created="2026-08-24T09:42:17Z", n_requested=20,
        pdb_id="6tgu",
    )
    rebuild_index(tmp_path)

    отобранные = [папка.name for папка in main_line_runs(tmp_path)]
    assert отобранные == ["2026-08-24-6tgu-s0-n100-seed1"]


def test_фильтр_по_мишени_оставляет_в_таблице_только_её(tmp_path: Path) -> None:
    """`pdb_id` сужает отбор до одной киназы — на случай, если вторую решат не показывать.

    По умолчанию фильтр выключен: прогоны разных мишеней дают отдельные колонки.
    Включать его — решение о составе результатов работы, и принимают его авторы
    разделов, а не реестр. Тест держит сам механизм, чтобы к моменту решения
    он был проверен, а не написан наспех.
    """
    создать_считанный(
        tmp_path, "2026-08-24-6tgu-s0-n100",
        source="local-poses", created="2026-08-24T10:00:00Z", n_requested=100,
        pdb_id=MAIN_TARGET_PDB_ID,
    )
    создать_считанный(
        tmp_path, "2026-09-17-6fnk-s0-n100",
        source="local-poses", created="2026-09-17T05:00:00Z", n_requested=100,
        pdb_id="6fnk",
    )
    rebuild_index(tmp_path)

    без_фильтра = [папка.name for папка in main_line_runs(tmp_path)]
    assert без_фильтра == ["2026-08-24-6tgu-s0-n100", "2026-09-17-6fnk-s0-n100"]

    с_фильтром = [папка.name for папка in main_line_runs(tmp_path, pdb_id=MAIN_TARGET_PDB_ID)]
    assert с_фильтром == ["2026-08-24-6tgu-s0-n100"]
