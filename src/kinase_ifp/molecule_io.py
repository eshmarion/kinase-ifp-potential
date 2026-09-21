"""Чтение молекул из файлов так, чтобы RDKit не получал путь строкой.

Правило, ради которого модуль существует, уже записано в `experiments.run_io` на
запись: RDKit передаёт путь в C++ через ANSI и на Windows не открывает ничего
с кириллицей — «Bad input file» на чтении, «Bad output file» на записи. Каталоги
прогонов латинские по формату паспорта прогона, но путь до них абсолютный (`config.PROJECT_ROOT`),
а временные каталоги лежат в `%TEMP%` внутри домашнего каталога пользователя. И то
и другое может содержать кириллицу, и 03.09 это уронило шесть тестов.

Модуль живёт в `kinase_ifp`, а не рядом с `run_io`, потому что им пользуются оба слоя:
`kinase_ifp` ничего не импортирует из `experiments`, и обратная зависимость перевернула
бы порядок пакетов. По той же причине внутри только `rdkit` и стандартная библиотека —
`structure_io` тянет MDAnalysis, которого в образе DiffSBDD нет.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rdkit import Chem


@contextmanager
def open_sdf(
    path: Path, *, sanitize: bool = True, removeHs: bool = False
) -> Iterator[Chem.ForwardSDMolSupplier]:
    """Открывает SDF на чтение через поток и отдаёт поставщик молекул.

    `ForwardSDMolSupplier`, а не `SDMolSupplier`: обход одноразовый, индексирования
    и `len()` по поставщику в проекте нет нигде. Неразобранная запись приходит как
    `None` — так же, как у `SDMolSupplier`, и вызывающий по-прежнему обязан её поймать
    и записать причину в `failures.csv`, а не пропустить молча.

    `removeHs=False` по умолчанию: водороды ставит производитель молекул,
    и без них ProLIF не отличит донор от акцептора.
    """
    with path.open("rb") as поток:
        yield Chem.ForwardSDMolSupplier(поток, sanitize=sanitize, removeHs=removeHs)


def read_mol(path: Path, *, removeHs: bool = False) -> Chem.Mol | None:
    """Первая молекула файла: текст читаем сами, разбираем блоком.

    Замена `Chem.MolFromMolFile`, которая берёт путь строкой. Поведение то же:
    `MolFromMolBlock` разбирает до `M  END`, поэтому на SDF из нескольких записей
    вернёт первую и SD-свойства не проставит — ровно как `MolFromMolFile`.
    Возвращает `None`, если RDKit молекулу не разобрал; проверять обязан вызывающий.
    """
    return Chem.MolFromMolBlock(path.read_text(encoding="utf-8"), removeHs=removeHs)
