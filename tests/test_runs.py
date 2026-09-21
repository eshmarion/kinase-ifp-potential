"""Тесты состава папки прогона и защиты от перезаписи.

Модалью здесь не пахнет намеренно: выгрузка из тома — это копирование файлов, а всё,
что можно сломать, ломается на составе папки и на повторном запуске. Проверяется именно
это, без облака.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runs import (
    MOLECULES_SDF,
    REQUIRED_RUN_FILES,
    RUN_JSON,
    RunError,
    check_pulled,
    compare_target_stamp,
    ensure_absent,
    library_versions,
    missing_files,
    read_passport,
    record_condition,
    record_environment,
    stamp_target,
    target_for_run,
    target_stamp,
)

ПАСПОРТ = {
    "run_id": "2026-08-23-6tgu-s0-n5",
    "created_utc": "2026-08-23T18:00:00Z",
    "source": "diffsbdd",
    "sampling": {"n_requested": 5, "n_returned": 4},
}


def создать_прогон(каталог: Path, *, паспорт: dict[str, object] | None = None) -> Path:
    """Собирает минимальную папку прогона по форматам папки прогона и паспорта прогона."""
    каталог.mkdir(parents=True, exist_ok=True)
    (каталог / RUN_JSON).write_text(
        json.dumps(паспорт if паспорт is not None else ПАСПОРТ), encoding="utf-8"
    )
    (каталог / MOLECULES_SDF).write_text("", encoding="utf-8")
    return каталог


def test_полный_прогон_ничего_не_недостаёт(tmp_path: Path) -> None:
    assert missing_files(создать_прогон(tmp_path / "run")) == []


def test_прогон_без_молекул_неполон(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "run")
    (прогон / MOLECULES_SDF).unlink()

    assert missing_files(прогон) == [MOLECULES_SDF]


def test_пустая_папка_недосчитывается_всех_файлов(tmp_path: Path) -> None:
    tmp_path.joinpath("run").mkdir()

    assert missing_files(tmp_path / "run") == list(REQUIRED_RUN_FILES)


def test_паспорт_читается(tmp_path: Path) -> None:
    паспорт = read_passport(создать_прогон(tmp_path / "run"))

    assert паспорт["run_id"] == "2026-08-23-6tgu-s0-n5"
    assert паспорт["source"] == "diffsbdd"


def test_паспорт_без_source_отвергается(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "run", паспорт={"run_id": "x", "created_utc": "y"})

    with pytest.raises(RunError, match="source"):
        read_passport(прогон)


def test_прогон_без_паспорта_отвергается(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()

    with pytest.raises(RunError, match=RUN_JSON):
        read_passport(tmp_path / "run")


def test_повторная_выгрузка_отклоняется(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "run")

    with pytest.raises(RunError, match="уже выгружен"):
        ensure_absent(прогон)


def test_новый_прогон_выгрузке_не_мешает(tmp_path: Path) -> None:
    ensure_absent(tmp_path / "нет-такого")


def test_пустая_папка_выгрузке_не_мешает(tmp_path: Path) -> None:
    """Неудачная выгрузка оставляла пустой каталог, и повтор был закрыт навсегда.

    Удалять в `runs/` нельзя, поэтому единственным выходом был разбор
    руками. Граница приведена к той же, что у `run_io.create_run_dir`
    и у проверки имени в сэмплере: запрет защищает молекулы, а не имя каталога
.
    """
    пустая = tmp_path / "2026-09-18-6tgu-s0-n100"
    пустая.mkdir()

    ensure_absent(пустая)


def test_папка_с_одним_файлом_уже_считается_занятой(tmp_path: Path) -> None:
    """Послабление касается только пустых папок: любой файл внутри снова даёт отказ."""
    начатая = tmp_path / "2026-09-18-6tgu-s0-n100"
    начатая.mkdir()
    (начатая / "failures.csv").write_text("mol_id,stage,reason", encoding="utf-8")

    with pytest.raises(RunError, match="уже выгружен"):
        ensure_absent(начатая)


def test_неполная_выгрузка_видна_сразу(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "run")
    (прогон / MOLECULES_SDF).unlink()

    with pytest.raises(RunError, match=MOLECULES_SDF):
        check_pulled(прогон)


def test_полная_выгрузка_возвращает_паспорт(tmp_path: Path) -> None:
    assert check_pulled(создать_прогон(tmp_path / "run"))["source"] == "diffsbdd"


def test_мишень_берётся_из_паспорта(tmp_path: Path) -> None:
    папка = создать_прогон(
        tmp_path / "прогон",
        паспорт={**ПАСПОРТ, "target": {"pdb_id": "6tgu", "klifs_structure_id": 12448}},
    )

    путь = target_for_run(папка, tmp_path / "targets")

    assert путь == tmp_path / "targets" / "6tgu" / "target.json"


def test_имя_папки_на_мишень_не_влияет(tmp_path: Path) -> None:
    """Суффикс в `run_id` не должен уводить расчёт на другую мишень (вопрос В-15)."""
    папка = создать_прогон(
        tmp_path / "2026-08-24-6tgu-s0-n100-seed1",
        паспорт={**ПАСПОРТ, "target": {"pdb_id": "6tgu"}},
    )

    assert target_for_run(папка, tmp_path / "targets").parent.name == "6tgu"


def test_прогон_без_мишени_в_паспорте_отвергается(tmp_path: Path) -> None:
    папка = создать_прогон(tmp_path / "прогон")

    with pytest.raises(RunError, match="target.pdb_id"):
        target_for_run(папка, tmp_path / "targets")


def собрать_пакет(каталог: Path, *, protonation: str = "explicit") -> Path:
    """Минимальный пакет мишени: все файлы, по которым считается хэш."""
    каталог.mkdir(parents=True, exist_ok=True)
    (каталог / "target.json").write_text(
        json.dumps(
            {"pdb_id": "6tgu", "klifs_structure_id": 12448, "protonation": protonation},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for имя in ("protein.pdb", "protein_noh.pdb", "pocket.pdb", "ligand.sdf"):
        (каталог / имя).write_text(f"содержимое {имя}", encoding="utf-8")
    return каталог / "target.json"


def test_отсутствующий_пакет_мишени_отвергается(tmp_path: Path) -> None:
    """Ветка заведена потому, что прежде она не исполнялась ни разу.

    Отказ здесь важнее, чем выглядит: без него штамп прогона считался бы по пустому
    месту, а сообщение «пакет не найден» — единственное, что отличает «мишени нет»
    от «мишень другая».
    """
    with pytest.raises(RunError, match="Пакет мишени не найден"):
        target_stamp(tmp_path / "нет-такой-мишени" / "target.json")


def test_неполный_пакет_отвергается_с_названным_файлом(tmp_path: Path) -> None:
    """Неполный хэш хуже отсутствующего — он выглядит как совпадение."""
    путь = собрать_пакет(tmp_path / "6tgu")
    (путь.parent / "pocket.pdb").unlink()

    with pytest.raises(RunError, match="pocket.pdb"):
        target_stamp(путь)


def test_штамп_несёт_режим_и_хэш(tmp_path: Path) -> None:
    штамп = target_stamp(собрать_пакет(tmp_path / "6tgu"))

    assert штамп["pdb_id"] == "6tgu"
    assert штамп["protonation"] == "explicit"
    assert len(штамп["package_sha256"]) == 64


def test_штамп_повторяется(tmp_path: Path) -> None:
    путь = собрать_пакет(tmp_path / "6tgu")

    assert target_stamp(путь) == target_stamp(путь)


def test_смена_режима_меняет_хэш(tmp_path: Path) -> None:
    """Ровно тот случай, что стоил дня 24.08: пакеты разных режимов несравнимы."""
    первый = target_stamp(собрать_пакет(tmp_path / "a", protonation="explicit"))
    второй = target_stamp(собрать_пакет(tmp_path / "b", protonation="implicit-prolif"))

    assert первый["package_sha256"] != второй["package_sha256"]


def test_правка_структуры_меняет_хэш(tmp_path: Path) -> None:
    """Хэш берётся по файлам пакета, а не по одному target.json."""
    путь = собрать_пакет(tmp_path / "6tgu")
    было = target_stamp(путь)
    (путь.parent / "protein.pdb").write_text("другой белок", encoding="utf-8")

    assert target_stamp(путь)["package_sha256"] != было["package_sha256"]


def test_неполный_пакет_отвергается(tmp_path: Path) -> None:
    путь = собрать_пакет(tmp_path / "6tgu")
    (путь.parent / "pocket.pdb").unlink()

    with pytest.raises(RunError, match="pocket.pdb"):
        target_stamp(путь)


def test_совпадение_штампов(tmp_path: Path) -> None:
    путь = собрать_пакет(tmp_path / "6tgu")
    паспорт = {**ПАСПОРТ, "target": target_stamp(путь)}

    вердикт, _ = compare_target_stamp(паспорт, путь)

    assert вердикт == "match"


def test_расхождение_штампов_видно(tmp_path: Path) -> None:
    свой = собрать_пакет(tmp_path / "a", protonation="explicit")
    чужой = собрать_пакет(tmp_path / "b", protonation="implicit-prolif")
    паспорт = {**ПАСПОРТ, "target": target_stamp(свой)}

    вердикт, штамп = compare_target_stamp(паспорт, чужой)

    assert вердикт == "mismatch"
    assert штамп["protonation"] == "implicit-prolif"


def test_паспорт_без_штампа_помечается_отдельно(tmp_path: Path) -> None:
    """Прогон, сделанный до появления штампа: не ошибка, но и не совпадение."""
    путь = собрать_пакет(tmp_path / "6tgu")
    паспорт = {**ПАСПОРТ, "target": {"pdb_id": "6tgu", "klifs_structure_id": 12448}}

    вердикт, _ = compare_target_stamp(паспорт, путь)

    assert вердикт == "unstamped"


def test_версии_библиотек_пишутся_в_паспорт(tmp_path: Path) -> None:
    """Без версий фраза «пересчёт даёт те же границы» непроверяема.

    Зерно бутстрэпа фиксировано, но одинаковый поток чисел между версиями `numpy`
    не обещан, а поля о версиях в паспорте не было вовсе.
    """
    прогон = создать_прогон(tmp_path / "прогон")

    record_environment(прогон)

    записано = read_passport(прогон)["environment"]
    assert записано["python"].startswith("3.10")
    assert записано == library_versions()


def test_повторный_счёт_в_том_же_окружении_проходит(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "прогон")
    record_environment(прогон)

    record_environment(прогон)

    assert read_passport(прогон)["environment"] == library_versions()


def test_счёт_в_другом_окружении_отвергается(tmp_path: Path) -> None:
    """Числа двух версий несравнимы так же, как отпечатки двух режимов."""
    прогон = создать_прогон(tmp_path / "прогон")
    record_environment(прогон)
    паспорт = read_passport(прогон)
    паспорт["environment"] = {**паспорт["environment"], "numpy": "1.0.0"}
    (прогон / "run.json").write_text(json.dumps(паспорт, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(RunError, match="другом окружении"):
        record_environment(прогон)


def test_отсутствующий_пакет_в_версии_не_пишется() -> None:
    """В образе генерации DiffSBDD нет prolif; «prolif: null» означало бы другое."""
    версии = library_versions()

    assert all(значение for значение in версии.values())
    assert "numpy" in версии


def test_условие_записывается_в_паспорт(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "прогон")

    record_condition(прогон, "baseline")

    assert read_passport(прогон)["condition"] == "baseline"


def test_условие_обрезается_по_краям(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "прогон")

    record_condition(прогон, "  baseline  ")

    assert read_passport(прогон)["condition"] == "baseline"


def test_пустое_условие_отвергается(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "прогон")

    with pytest.raises(RunError, match="пустое условие"):
        record_condition(прогон, "   ")


def test_повторная_запись_того_же_условия_проходит(tmp_path: Path) -> None:
    """Цепочку прогона должно быть можно запустить дважды."""
    прогон = создать_прогон(tmp_path / "прогон")
    record_condition(прогон, "baseline")

    record_condition(прогон, "baseline")

    assert read_passport(прогон)["condition"] == "baseline"


def test_смена_условия_отвергается(tmp_path: Path) -> None:
    """Перемеченный прогон делает несравнимыми числа, уже процитированные из него."""
    прогон = создать_прогон(tmp_path / "прогон")
    record_condition(прогон, "baseline")

    with pytest.raises(RunError, match="уже помечен условием"):
        record_condition(прогон, "guided")

    assert read_passport(прогон)["condition"] == "baseline"


def test_запись_условия_не_трогает_остальной_паспорт(tmp_path: Path) -> None:
    прогон = создать_прогон(tmp_path / "прогон")
    было = read_passport(прогон)

    record_condition(прогон, "baseline")

    стало = read_passport(прогон)
    assert {ключ: стало[ключ] for ключ in было} == было


def test_штамп_проставляется_прогону_без_него(tmp_path: Path) -> None:
    """Прогоны от 23–24.08 сделаны до формата паспорта прогона и штампа не несут."""
    пакет = собрать_пакет(tmp_path / "6tgu")
    прогон = создать_прогон(tmp_path / "прогон", паспорт={**ПАСПОРТ, "target": {"pdb_id": "6tgu"}})

    паспорт = stamp_target(прогон, пакет)

    assert паспорт["target"]["protonation"] == "explicit"
    assert len(паспорт["target"]["package_sha256"]) == 64
    assert compare_target_stamp(read_passport(прогон), пакет)[0] == "match"


def test_штамп_не_трогает_прочие_поля_мишени(tmp_path: Path) -> None:
    пакет = собрать_пакет(tmp_path / "6tgu")
    прогон = создать_прогон(
        tmp_path / "прогон",
        паспорт={**ПАСПОРТ, "target": {"pdb_id": "6tgu", "klifs_structure_id": 12448}},
    )

    паспорт = stamp_target(прогон, пакет)

    assert паспорт["target"]["klifs_structure_id"] == 12448


def test_повторный_штамп_ничего_не_меняет(tmp_path: Path) -> None:
    пакет = собрать_пакет(tmp_path / "6tgu")
    прогон = создать_прогон(tmp_path / "прогон", паспорт={**ПАСПОРТ, "target": {"pdb_id": "6tgu"}})
    первый = stamp_target(прогон, пакет)

    второй = stamp_target(прогон, пакет)

    assert первый == второй


def test_чужой_штамп_не_переписывается(tmp_path: Path) -> None:
    """Штамп говорит, на чём прогон посчитан, а не на чём его хотят видеть."""
    пакет = собрать_пакет(tmp_path / "6tgu")
    прогон = создать_прогон(
        tmp_path / "прогон",
        паспорт={
            **ПАСПОРТ,
            "target": {
                "pdb_id": "6tgu",
                "protonation": "implicit-prolif",
                "package_sha256": "0" * 64,
            },
        },
    )

    with pytest.raises(RunError, match="штамп другого пакета"):
        stamp_target(прогон, пакет)

    assert read_passport(прогон)["target"]["package_sha256"] == "0" * 64
