"""Проверка окружения: библиотеки, без которых проект не считается.

Если образ или окружение собраны неправильно, это видно здесь, а не через час
отладки в чужом модуле. Отдельно проверяется rdkit.Chem.Draw: он тянет системную
библиотеку libXrender, которой нет в slim-образах, а без него не импортируется prolif.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path


def test_химические_библиотеки_импортируются() -> None:
    for имя in ("rdkit", "prolif", "MDAnalysis", "pandas", "numpy", "sklearn"):
        importlib.import_module(имя)


def test_отрисовка_rdkit_доступна() -> None:
    importlib.import_module("rdkit.Chem.Draw")


def test_клиент_klifs_импортируется() -> None:
    importlib.import_module("opencadd.databases.klifs")


def test_пакеты_проекта_импортируются() -> None:
    for имя in ("kinase_ifp", "soft_potential", "integration"):
        importlib.import_module(имя)


def test_версия_python_ровно_3_10() -> None:
    """Версия зафиксирована, а не «какая нашлась».

    Разные версии дают разные наборы колёс и разное поведение библиотек — ровно
    то, ради устранения чего заведён контейнер.
    """
    import sys

    assert sys.version_info[:2] == (3, 10), f"ожидался Python 3.10, а не {sys.version.split()[0]}"


def test_все_обязательные_зависимости_установлены() -> None:
    """Объявленная в `pyproject.toml` зависимость обязана стоять в окружении.

    Ловит отставший образ. Однажды в него не вошла `vina`, и весь
    `tests/test_docking.py` молча уходил в skip: вывод `pytest` показывал
    «1545 passed, 1 skipped» и ничем не отличался от нормы, а пять тестов
    не исполнялись.

    Проверяется наличие дистрибутива, а не импорт: имя пакета и имя модуля совпадают
    далеко не всегда (`opencadd`, `rdkit-stubs`, `scikit-learn`). Зависимость, чей маркер
    не подходит текущей платформе, пропускается — на Windows так выпадает `vina`.
    """
    import tomli
    from packaging.requirements import Requirement

    корень = Path(__file__).resolve().parent.parent
    объявлены = tomli.loads((корень / "pyproject.toml").read_text(encoding="utf-8"))
    отсутствуют = []
    for строка in объявлены["project"]["dependencies"]:
        требование = Requirement(строка)
        if требование.marker is not None and not требование.marker.evaluate():
            continue
        try:
            metadata.distribution(требование.name)
        except metadata.PackageNotFoundError:
            отсутствуют.append(требование.name)

    assert not отсутствуют, (
        "в окружении нет объявленных зависимостей: "
        + ", ".join(отсутствуют)
        + ". Образ отстал от uv.lock — пересоберите: docker compose build dev"
    )
