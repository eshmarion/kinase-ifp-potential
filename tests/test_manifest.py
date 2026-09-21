"""Тесты манифеста выгрузки KLIFS.

Главный здесь — последний: он сверяет манифест репозитория с файлами `data/klifs/`
и валит `pytest`, если выгрузка изменилась, а манифест остался прежним. Без него
манифест повторил бы `Н1а` — запись, которая расходится с действительностью незаметно.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kinase_ifp.manifest import (
    KLIFS_DIR,
    MANIFEST_NAME,
    ManifestError,
    build_manifest,
    check_manifest,
    count_records,
    data_files,
    read_manifest,
    write_manifest,
)

# Имена из FILE_ORIGINS: во временном каталоге берутся настоящие, иначе сборка
# справедливо откажется работать с файлом неизвестного происхождения.
ВЫГРУЗКА = "structures.csv"
ПРОИЗВОДНЫЙ = "pocket_content.csv"


def сделать_каталог(tmp_path: Path) -> Path:
    каталог = tmp_path / "klifs"
    каталог.mkdir()
    (каталог / ВЫГРУЗКА).write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    (каталог / ПРОИЗВОДНЫЙ).write_text("pdb,class\n6tgu,diffsbdd-ready\n", encoding="utf-8")
    return каталог


def test_манифест_описывает_все_файлы_каталога(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    манифест = build_manifest(каталог)

    assert set(манифест["files"]) == {ВЫГРУЗКА, ПРОИЗВОДНЫЙ}
    assert манифест["files"][ВЫГРУЗКА]["records"] == 2
    assert манифест["files"][ПРОИЗВОДНЫЙ]["records"] == 1
    assert манифест["files"][ВЫГРУЗКА]["expected_command"] == "python scripts/fetch_klifs.py"


def test_версия_базы_пуста_и_объяснена(tmp_path: Path) -> None:
    """KLIFS версии не отдаёт; пустое поле обязано нести причину, а не молчать."""
    манифест = build_manifest(сделать_каталог(tmp_path))

    assert манифест["source"]["database_version"] is None
    assert манифест["source"]["database_version_note"]


def test_файл_вне_репозитория_остаётся_без_даты(tmp_path: Path) -> None:
    """Верхней границы обращения к базе нет — и манифест говорит об этом прямо."""
    манифест = build_manifest(сделать_каталог(tmp_path))
    запись = манифест["files"][ВЫГРУЗКА]

    assert запись["retrieved_not_later_than"] is None
    assert "не отслеживается git" in запись["date_basis"]


def test_файл_неизвестного_происхождения_обрывает_сборку(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    (каталог / "самодельный.csv").write_text("x\n1\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="происхождение не объявлено"):
        build_manifest(каталог)


def test_свежий_манифест_расхождений_не_даёт(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    write_manifest(каталог)

    assert check_manifest(каталог) == []


def test_дописанный_файл_ловится_по_числу_записей_и_хешу(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    write_manifest(каталог)
    (каталог / ВЫГРУЗКА).write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")

    расхождения = check_manifest(каталог)

    assert any("записей" in строка for строка in расхождения)
    assert any("sha256" in строка for строка in расхождения)


def test_новый_файл_без_пересборки_манифеста_ловится(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    write_manifest(каталог)
    (каталог / "kinases.csv").write_text("name\nCK2a2\n", encoding="utf-8")

    расхождения = check_manifest(каталог)

    assert any("в манифесте не объявлен" in строка for строка in расхождения)


def test_пропавший_файл_ловится(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    write_manifest(каталог)
    (каталог / ПРОИЗВОДНЫЙ).unlink()

    расхождения = check_manifest(каталог)

    assert any("на диске его нет" in строка for строка in расхождения)


def test_манифест_не_считает_сам_себя(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)
    write_manifest(каталог)

    assert MANIFEST_NAME not in {путь.name for путь in data_files(каталог)}
    assert check_manifest(каталог) == []


def test_перенос_внутри_поля_не_завышает_число_записей(tmp_path: Path) -> None:
    """Подсчёт переводов строки дал бы 3 записи вместо 2 и выглядел бы как порча файла."""
    путь = tmp_path / "с_переносом.csv"
    путь.write_text('a,b\n1,"две\nстроки"\n3,4\n', encoding="utf-8")

    assert count_records(путь) == 2


def test_отсутствие_манифеста_сообщает_команду_сборки(tmp_path: Path) -> None:
    каталог = сделать_каталог(tmp_path)

    with pytest.raises(ManifestError, match="klifs_manifest.py --write"):
        read_manifest(каталог)


@pytest.mark.skipif(not KLIFS_DIR.is_dir(), reason="каталог выгрузки KLIFS отсутствует")
def test_манифест_репозитория_не_отстал_от_выгрузки() -> None:
    """Выгрузка изменилась — пересоберите: `python scripts/klifs_manifest.py --write`."""
    assert check_manifest(KLIFS_DIR) == []


@pytest.mark.skipif(not KLIFS_DIR.is_dir(), reason="каталог выгрузки KLIFS отсутствует")
def test_каждый_файл_выгрузки_несёт_команду_воспроизведения() -> None:
    манифест = read_manifest(KLIFS_DIR)

    for имя, запись in манифест["files"].items():
        assert запись["expected_command"], f"{имя}: пустая команда воспроизведения"
        assert запись["sha256"], f"{имя}: пустой sha256"


@pytest.mark.skipif(not KLIFS_DIR.is_dir(), reason="каталог выгрузки KLIFS отсутствует")
def test_манифест_читается_как_json_и_знает_свою_схему() -> None:
    данные = json.loads((KLIFS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))

    assert данные["schema"] == 1
    assert данные["source"]["database"] == "KLIFS"


@pytest.mark.skipif(not KLIFS_DIR.is_dir(), reason="каталог выгрузки KLIFS отсутствует")
def test_команда_выгрузки_структур_зовётся_без_предела() -> None:
    """`--limit 0` у `fetch_klifs.py` даёт пустую таблицу, а не «все».

    Ключ объявлен с `default=None`, а `fetch_structures` проверяет `if limit is not None`:
    ноль не `None`, поэтому ноль означает `df.head(0)`. У `fetch_klifs_ifp.py` тот же
    на вид ключ работает наоборот — там `NO_LIMIT = 0`. Тест держит различие, потому
    что одинаковая запись для двух скриптов выглядит естественно и снова соблазнит.
    """
    манифест = read_manifest(KLIFS_DIR)

    for имя, запись in манифест["files"].items():
        команда = запись["expected_command"]
        if "fetch_klifs.py" in команда:
            assert "--limit" not in команда, f"{имя}: fetch_klifs.py с --limit вернёт пустоту"


@pytest.mark.skipif(not KLIFS_DIR.is_dir(), reason="каталог выгрузки KLIFS отсутствует")
def test_ожидаемая_команда_названа_ожидаемой() -> None:
    """Имя поля не должно читаться как запись факта: манифест фактического вызова не знает."""
    манифест = read_manifest(KLIFS_DIR)

    assert манифест["source"]["command_note"]
    for имя, запись in манифест["files"].items():
        assert "command" not in запись, f"{имя}: поле command вернулось под прежним именем"
        assert запись["expected_command"], f"{имя}: пустая ожидаемая команда"
