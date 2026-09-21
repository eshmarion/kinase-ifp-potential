"""Тесты состава папки прогона: имена файлов объявлены один раз и доступны облаку.

Оба теста написаны по следам падения. `runs.py` брал `RUN_JSON` у `report.py`,
а тот импортирует pandas, которого в образе DiffSBDD нет: сэмплер упал бы уже
на удалённой машине, и локально это не воспроизводится никак.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import layout

# Модули, которые пользуются именами производных файлов, и что каждый из них берёт.
# Выход этапа (`results/`) сюда тоже входит: пока имена таблиц и рисунков жили
# строками в пяти местах, правило «имя объявлено один раз» на них не действовало.
ПОТРЕБИТЕЛИ: dict[str, tuple[str, ...]] = {
    "experiments.run_io": ("MOLECULES_SDF", "RUN_JSON", "FAILURES_CSV"),
    "experiments.runs": ("MOLECULES_SDF", "RUN_JSON"),
    "experiments.ranking": ("MOLECULES_SDF", "RANKING_CSV", "RUN_JSON"),
    "experiments.report": (
        "BEFORE_AFTER_CSV",
        "BEFORE_AFTER_MD",
        "CSV_EOL",
        "METRICS_CSV",
        "RANKING_CSV",
        "RUN_JSON",
        "SUMMARY_CSV",
    ),
    "experiments.registry": (
        "CSV_EOL",
        "INDEX_CSV",
        "METRICS_CSV",
        "POCKET_MODE_POCKET_IDS",
        "RANKING_CSV",
    ),
    "evaluation.figures": (
        "CSV_EOL",
        "FIGURES_ENVIRONMENT_JSON",
        "FIGURES_README",
        "PANELS_CSV",
    ),
}

# Расширения, по которым строка считается именем файла, а не просто текстом.
РАСШИРЕНИЯ_ФАЙЛОВ: frozenset[str] = frozenset({"csv", "md", "sdf", "json", "png"})

# Тяжёлые пакеты, которых нет в образе DiffSBDD. Любой из них, попав в цепочку
# импорта `layout`, вернёт нас к падению 24.08.
ТЯЖЁЛЫЕ = ("pandas", "MDAnalysis", "prolif", "rdkit", "numpy", "scipy")


@pytest.mark.parametrize(
    ("модуль", "имя"),
    [(модуль, имя) for модуль, имена in ПОТРЕБИТЕЛИ.items() for имя in имена],
)
def test_имя_объявлено_только_в_layout(модуль: str, имя: str) -> None:
    """Каждое имя — тот же объект, что в `layout`, а не собственная копия.

    Сравнение через `is`, а не `==`: копия с тем же значением проходит проверку
    на равенство и расходится позже, когда правят одну из двух.
    """
    свой = getattr(importlib.import_module(модуль), имя)

    assert свой is getattr(layout, имя)


def _литералы_имён() -> list[tuple[str, int, str]]:
    """Строковые литералы в `src/` и `scripts/`, совпадающие с именами из `layout`.

    Обход через `ast` и без докстрок: имя файла упоминается в пояснениях почти на
    каждой странице этого проекта, и текстовый поиск дал бы сотню ложных срабатываний.
    """
    имена = {
        значение
        for имя, значение in vars(layout).items()
        if not имя.startswith("_")
        and isinstance(значение, str)
        and значение.rsplit(".", 1)[-1] in РАСШИРЕНИЯ_ФАЙЛОВ
    }
    совпадения: list[tuple[str, int, str]] = []
    for каталог in ("src", "scripts"):
        for файл in sorted((КОРЕНЬ / каталог).rglob("*.py")):
            путь = str(файл.relative_to(КОРЕНЬ)).replace("\\", "/")
            if файл.name == "layout.py":
                continue
            дерево = ast.parse(файл.read_text(encoding="utf-8"), filename=str(файл))
            докстроки = {
                id(узел.body[0].value)
                for узел in ast.walk(дерево)
                if isinstance(
                    узел, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
                )
                and узел.body
                and isinstance(узел.body[0], ast.Expr)
                and isinstance(узел.body[0].value, ast.Constant)
            }
            for узел in ast.walk(дерево):
                if (
                    isinstance(узел, ast.Constant)
                    and isinstance(узел.value, str)
                    and узел.value in имена
                    and id(узел) not in докстроки
                ):
                    совпадения.append((путь, узел.lineno, узел.value))
    return совпадения


def test_имя_файла_не_пишется_литералом() -> None:
    """Правило «имя объявлено один раз» держится проверкой, а не памятью.

    Словарь `ПОТРЕБИТЕЛИ` выше ловит только тех, кто имя уже импортировал: модуль,
    написавший `"index.csv"` строкой, в него просто не попадёт. Так и вышло — правило
    завели 24.08 для папки прогона, а `before_after.csv`, `panels.csv` и имена рисунков
    остались строками в пяти местах, и ещё в четырёх местах так же
    жили `index.csv` и `failures.csv`.
    """
    плохие = _литералы_имён()

    assert not плохие, "имя файла записано литералом: " + "; ".join(
        f"{файл}:{строка} {значение!r}" for файл, строка, значение in плохие
    ) + ". Имена объявлены в experiments.layout — импортируйте оттуда"


def test_layout_не_тянет_тяжёлых_зависимостей() -> None:
    """`layout` обязан читаться там, где стоит только stdlib.

    Проверяется в отдельном процессе: в текущем pandas и rdkit давно импортированы
    другими тестами, и `sys.modules` ничего не покажет.
    """
    код = (
        "import sys; import experiments.layout; "
        f"print([имя for имя in {ТЯЖЁЛЫЕ!r} if имя in sys.modules])"
    )
    итог = subprocess.run(
        [sys.executable, "-c", код], capture_output=True, text=True, check=True
    )

    assert итог.stdout.strip() == "[]", (
        f"импорт experiments.layout потянул за собой {итог.stdout.strip()}. "
        "Имена файлов прогона обязаны читаться в облачном образе, где этих пакетов нет"
    )


def test_реестр_обходится_без_pandas() -> None:
    """`registry` брал имена у `report` и тянул pandas в модуль, которому он не нужен."""
    код = "import sys; import experiments.registry; print('pandas' in sys.modules)"
    итог = subprocess.run(
        [sys.executable, "-c", код], capture_output=True, text=True, check=True
    )

    assert итог.stdout.strip() == "False"


# Аргументы `tempfile`, которые становятся именем на файловой системе. Кириллица
# в любом из них ломает RDKit на Windows: путь уходит в C++ через ANSI, и чтение
# падает с «Bad input file». В контейнере Linux этого не видно вовсе, поэтому
# проверка идёт по исходникам, а не поведением — поведенческий тест был бы
# зелёным при любом префиксе.
ИМЕНУЮЩИЕ_АРГУМЕНТЫ = ("prefix", "suffix", "dir")

КОРЕНЬ = Path(__file__).resolve().parents[1]


def _временные_имена() -> list[tuple[str, int, str, str]]:
    """Собирает литералы `prefix`/`suffix`/`dir` у вызовов `tempfile` в `src/` и `scripts/`.

    Возвращает список `(файл, строка, аргумент, значение)`. Обход через `ast`, а не
    поиском по тексту: иначе в выдачу попадут те же слова из комментариев и докстрок.
    """
    совпадения: list[tuple[str, int, str, str]] = []
    for каталог in ("src", "scripts"):
        for файл in sorted((КОРЕНЬ / каталог).rglob("*.py")):
            дерево = ast.parse(файл.read_text(encoding="utf-8"), filename=str(файл))
            for узел in ast.walk(дерево):
                if not isinstance(узел, ast.Call):
                    continue
                вызов = ast.unparse(узел.func)
                if "tempfile" not in вызов and not вызов.startswith(
                    ("TemporaryDirectory", "NamedTemporaryFile", "mkdtemp", "mkstemp")
                ):
                    continue
                for аргумент in узел.keywords:
                    if аргумент.arg in ИМЕНУЮЩИЕ_АРГУМЕНТЫ and isinstance(
                        аргумент.value, ast.Constant
                    ):
                        значение = аргумент.value.value
                        if isinstance(значение, str):
                            совпадения.append(
                                (
                                    str(файл.relative_to(КОРЕНЬ)).replace("\\", "/"),
                                    узел.lineno,
                                    аргумент.arg or "",
                                    значение,
                                )
                            )
    return совпадения


def test_временные_каталоги_называются_латиницей() -> None:
    """Имя временного каталога не должно содержать кириллицы — её не переживает RDKit.

    Поломка 03.09: `restamp.py` создавал каталог с префиксом «сверка-», и шесть тестов
    падали на нативном Windows с «Bad input file», обвиняя при этом пакет мишени.
    """
    плохие = [совпадение for совпадение in _временные_имена() if not совпадение[3].isascii()]

    assert not плохие, "не-ASCII в имени временного файла или каталога: " + "; ".join(
        f"{файл}:{строка} {аргумент}={значение!r}" for файл, строка, аргумент, значение in плохие
    )


# Вызовы RDKit, которые принимают только путь строкой. Путь уходит в C++ через ANSI,
# и на Windows кириллица в нём даёт «Bad input file» / «Bad output file». Замены —
# `kinase_ifp.molecule_io.open_sdf` и `read_mol`, на запись — приём из
# `experiments.run_io.write_molecules`: писателю отдают открытый поток.
ТОЛЬКО_ПУТЬ: frozenset[str] = frozenset(
    {"SDMolSupplier", "MolFromMolFile", "MolFromPDBFile", "MolToPDBFile"}
)

# Принимают и поток, и путь; путём строкой пользоваться нельзя по той же причине.
ПОТОК_ИЛИ_ПУТЬ: frozenset[str] = frozenset({"SDWriter", "ForwardSDMolSupplier"})


def _вызовы_rdkit() -> list[tuple[str, int, str]]:
    """Находит в `src/` и `scripts/` вызовы RDKit, получающие путь строкой.

    Возвращает `(файл, строка, имя)`. Обход через `ast`: текстовый поиск ловил бы
    те же имена в комментариях и докстроках, которых в этом проекте много.
    """
    совпадения: list[tuple[str, int, str]] = []
    for каталог in ("src", "scripts"):
        for файл in sorted((КОРЕНЬ / каталог).rglob("*.py")):
            дерево = ast.parse(файл.read_text(encoding="utf-8"), filename=str(файл))
            for узел in ast.walk(дерево):
                if not isinstance(узел, ast.Call) or not isinstance(узел.func, ast.Attribute):
                    continue
                имя = узел.func.attr
                путём = имя in ТОЛЬКО_ПУТЬ or (
                    имя in ПОТОК_ИЛИ_ПУТЬ
                    and узел.args
                    and isinstance(узел.args[0], ast.Call)
                    and ast.unparse(узел.args[0].func) == "str"
                )
                if путём:
                    совпадения.append(
                        (str(файл.relative_to(КОРЕНЬ)).replace("\\", "/"), узел.lineno, имя)
                    )
    return совпадения


def test_rdkit_не_получает_путь_строкой() -> None:
    """Путь строкой RDKit на Windows не открывает, если в нём есть кириллица.

    Поломка 03.09: шесть тестов падали с «Bad input file», потому что
    временный каталог назывался по-русски. Префикс переименовали, но корень `%TEMP%`
    лежит в домашнем каталоге пользователя, а `config.PROJECT_ROOT` абсолютен — то есть
    кириллица приходит и без временных каталогов. Лечится не именами, а тем, что
    RDKit получает поток.

    Проверка по исходникам, а не поведением: в контейнере Linux такой путь
    открывается нормально, и поведенческий тест был бы зелёным при любом коде.
    """
    плохие = _вызовы_rdkit()

    assert not плохие, "RDKit получает путь строкой: " + "; ".join(
        f"{файл}:{строка} {имя}" for файл, строка, имя in плохие
    ) + ". Замены — kinase_ifp.molecule_io.open_sdf и read_mol"
