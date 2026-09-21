"""Проверки классификации состава кармана (`src/kinase_ifp/pocket_content.py`).

Сети нет ни в одном тесте: классификация работает с текстом PDB, и текст собирается
здесь же из отдельных записей. Так проверяется именно правило отбора, а не доступность
RCSB.

Каждый тест отвечает одному из четырёх условий DiffSBDD, прочитанных в коде модели:
только стандартные аминокислоты в кармане, лиганд не ковалентный, QED не ниже порога
обучающей выборки, лиганд не соседствует со второй копией.
"""

from __future__ import annotations

import pytest

from kinase_ifp.config import (
    DIFFSBDD_DATASET_CROSSDOCKED,
    DIFFSBDD_DATASET_MOAD,
    DIFFSBDD_POCKET_CUTOFF_A,
    POCKET_CONTENT_ADDITIVE,
    POCKET_CONTENT_CLASSES,
    POCKET_CONTENT_COFACTOR,
    POCKET_CONTENT_COVALENT,
    POCKET_CONTENT_LOW_QED,
    POCKET_CONTENT_METAL,
    POCKET_CONTENT_MULTI_LIGAND,
    POCKET_CONTENT_READY,
)
from kinase_ifp.pocket_content import (
    PocketContentError,
    classify_pocket,
    is_covalent,
    ligand_copies,
    neighbours_within,
    parse_atoms,
)


def запись(
    вид: str,
    серия: int,
    имя_атома: str,
    остаток: str,
    цепь: str,
    номер: int,
    x: float,
    y: float,
    z: float,
    элемент: str,
) -> str:
    """Одна строка PDB по фиксированной раскладке колонок формата."""
    return (
        f"{вид:<6}{серия:>5} {имя_атома:^4}{'':1}{остаток:>3} {цепь}{номер:>4}{'':4}"
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}{'':10}{элемент:>2}"
    )


def структура(
    *,
    лиганд_в: tuple[float, float, float] = (0.0, 0.0, 0.0),
    белок_в: tuple[float, float, float] = (6.0, 0.0, 0.0),
    добавки: tuple[tuple[str, tuple[float, float, float]], ...] = (),
) -> str:
    """Минимальная структура: один атом лиганда, один атом белка, заданные гетерогруппы."""
    строки = [
        запись("ATOM", 1, "CA", "ALA", "A", 10, *белок_в, "C"),
        запись("HETATM", 2, "C1", "LIG", "A", 900, *лиганд_в, "C"),
    ]
    for номер, (код, точка) in enumerate(добавки, start=901):
        строки.append(запись("HETATM", номер + 10, "O", код, "A", номер, *точка, "O"))
    return "\n".join(строки) + "\nEND\n"


# --- разбор файла ----------------------------------------------------------


def test_разбор_читает_координаты_и_коды() -> None:
    атомы = parse_atoms(структура())
    assert [а.record for а in атомы] == ["ATOM", "HETATM"]
    assert атомы[1].resname == "LIG"
    assert атомы[1].x == pytest.approx(0.0)
    assert атомы[0].x == pytest.approx(6.0)


def test_пустой_файл_отвергается_с_причиной() -> None:
    with pytest.raises(PocketContentError, match="ни одной записи"):
        parse_atoms("HEADER    ничего\nEND\n")


def test_читается_только_первая_модель() -> None:
    """В ЯМР-файлах десятки моделей: без остановки на ENDMDL соседи умножились бы."""
    текст = (
        "MODEL        1\n"
        + структура()
        + "ENDMDL\n"
        + "MODEL        2\n"
        + структура(добавки=(("MG", (2.0, 0.0, 0.0)),))
        + "ENDMDL\n"
    )
    атомы = parse_atoms(текст)
    assert all(а.resname != "MG" for а in атомы)


def test_лиганд_не_найден_даёт_отказ_а_не_пустой_вердикт() -> None:
    with pytest.raises(PocketContentError, match="не найден"):
        classify_pocket(структура(), "0xxx", "ZZZ", qed=0.5)


# --- условие 1: в кармане только стандартные аминокислоты ------------------


def test_чистый_карман_годен() -> None:
    вердикт = classify_pocket(структура(), "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_READY
    assert вердикт.matches_diffsbdd
    assert вердикт.reasons == ()


def test_вода_в_кармане_не_мешает() -> None:
    """Карман модели воды не содержит по построению, наш IFP её тоже не видит."""
    текст = структура(добавки=(("HOH", (3.0, 0.0, 0.0)), ("HOH", (2.5, 1.0, 0.0))))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.matches_diffsbdd
    assert вердикт.water_atoms == 2
    assert вердикт.pocket_class == POCKET_CONTENT_READY


def test_ион_металла_снимает_соответствие() -> None:
    текст = структура(добавки=(("MG", (3.0, 0.0, 0.0)),))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_METAL
    assert not вердикт.matches_diffsbdd
    assert any("ион металла" in причина for причина in вердикт.reasons)


def test_добавка_кристаллизации_снимает_соответствие() -> None:
    текст = структура(добавки=(("EDO", (3.0, 0.0, 0.0)),))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_ADDITIVE


def test_кофактор_снимает_соответствие() -> None:
    текст = структура(добавки=(("ATP", (4.0, 0.0, 0.0)),))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_COFACTOR


def test_посторонняя_молекула_даёт_многолигандный_класс() -> None:
    текст = структура(добавки=(("XYZ", (3.5, 0.0, 0.0)),))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_MULTI_LIGAND


def test_дальняя_гетерогруппа_в_карман_не_входит() -> None:
    """Радиус — это радиус: ион в 20 Å от лиганда к карману отношения не имеет."""
    текст = структура(добавки=(("MG", (20.0, 0.0, 0.0)),))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.matches_diffsbdd


def test_радиус_по_умолчанию_равен_радиусу_модели() -> None:
    вердикт = classify_pocket(структура(), "0abc", "LIG", qed=0.5)
    assert вердикт.radius == DIFFSBDD_POCKET_CUTOFF_A == 8.0


def test_меньший_радиус_выводит_группу_из_кармана() -> None:
    текст = структура(добавки=(("MG", (6.0, 0.0, 0.0)),))
    широкий = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    узкий = classify_pocket(текст, "0abc", "LIG", qed=0.5, radius=4.5)
    assert not широкий.matches_diffsbdd
    assert узкий.matches_diffsbdd


# --- условие 2: лиганд не ковалентный --------------------------------------


def test_ковалентная_связь_с_белком_снимает_соответствие() -> None:
    текст = структура(белок_в=(1.8, 0.0, 0.0))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.pocket_class == POCKET_CONTENT_COVALENT
    assert not вердикт.matches_diffsbdd


def test_водородная_связь_ковалентной_не_считается() -> None:
    """2.8 Å между тяжёлыми атомами — водородная связь; порог стоит на 1.9 Å."""
    текст = структура(белок_в=(2.8, 0.0, 0.0))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.matches_diffsbdd


def test_водороды_не_дают_ложной_ковалентности() -> None:
    """Связь X–H короче любого порога и сработала бы на каждой структуре с водородами."""
    текст = (
        запись("ATOM", 1, "CA", "ALA", "A", 10, 6.0, 0.0, 0.0, "C")
        + "\n"
        + запись("ATOM", 2, "HA", "ALA", "A", 10, 1.0, 0.0, 0.0, "H")
        + "\n"
        + запись("HETATM", 3, "C1", "LIG", "A", 900, 0.0, 0.0, 0.0, "C")
        + "\nEND\n"
    )
    атомы = parse_atoms(текст)
    признак, расстояние = is_covalent(ligand_copies(атомы, "LIG")[0], атомы)
    assert not признак
    assert расстояние == pytest.approx(6.0)


# --- условие 3: QED не ниже порога обучающей выборки -----------------------


def test_низкий_qed_снимает_соответствие_в_ветке_moad() -> None:
    вердикт = classify_pocket(структура(), "0abc", "LIG", qed=0.2, dataset=DIFFSBDD_DATASET_MOAD)
    assert вердикт.pocket_class == POCKET_CONTENT_LOW_QED
    assert not вердикт.matches_diffsbdd


def test_в_ветке_crossdocked_порог_qed_не_применяется() -> None:
    """Наш чекпойнт из ветки CrossDocked, а там фильтра по QED нет вовсе.

    Применять его значило бы браковать комплексы по условию, которого модель не знает.
    """
    вердикт = classify_pocket(
        структура(), "0abc", "LIG", qed=0.05, dataset=DIFFSBDD_DATASET_CROSSDOCKED
    )
    assert вердикт.matches_diffsbdd
    assert вердикт.pocket_class == POCKET_CONTENT_READY


def test_неизвестная_ветка_отвергается() -> None:
    with pytest.raises(PocketContentError, match="Неизвестный набор условий"):
        classify_pocket(структура(), "0abc", "LIG", qed=0.5, dataset="pdbbind")


def test_неизмеренный_qed_не_выдаётся_за_пройденное_условие() -> None:
    вердикт = classify_pocket(
        структура(), "0abc", "LIG", qed=None, dataset=DIFFSBDD_DATASET_MOAD
    )
    assert вердикт.matches_diffsbdd, "неизмеренный QED не должен браковать структуру"
    assert any("не измерен" in причина for причина in вердикт.reasons), (
        "непроверенное условие обязано быть видно в причинах"
    )


# --- несколько причин сразу ------------------------------------------------


def test_показываются_все_причины_а_не_первая() -> None:
    текст = структура(добавки=(("MG", (3.0, 0.0, 0.0)), ("EDO", (3.5, 0.0, 0.0))))
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.1, dataset=DIFFSBDD_DATASET_MOAD)
    assert len(вердикт.reasons) == 3, вердикт.reasons
    assert вердикт.pocket_class == POCKET_CONTENT_METAL, "порядок важности: ион выше добавки"


def test_класс_всегда_из_закрытого_списка() -> None:
    случаи = [
        структура(),
        структура(добавки=(("MG", (3.0, 0.0, 0.0)),)),
        структура(добавки=(("XYZ", (3.0, 0.0, 0.0)),)),
        структура(белок_в=(1.5, 0.0, 0.0)),
    ]
    for текст in случаи:
        вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
        assert вердикт.pocket_class in POCKET_CONTENT_CLASSES


# --- копии лиганда ---------------------------------------------------------


def test_копии_лиганда_в_разных_карманах_не_бракуют_структуру() -> None:
    """Две молекулы в асимметричной единице — норма, а не второй лиганд в кармане."""
    текст = (
        структура().replace("END\n", "")
        + запись("HETATM", 50, "C1", "LIG", "B", 900, 40.0, 0.0, 0.0, "C")
        + "\nEND\n"
    )
    атомы = parse_atoms(текст)
    assert len(ligand_copies(атомы, "LIG")) == 2
    вердикт = classify_pocket(текст, "0abc", "LIG", qed=0.5)
    assert вердикт.ligand_copies == 2
    assert вердикт.matches_diffsbdd


def test_соседи_считаются_по_атомам_а_не_по_центрам() -> None:
    """Крупный остаток дотягивается до кармана краем — по центроиду он бы не попал."""
    лиганд = ligand_copies(parse_atoms(структура()), "LIG")[0]
    текст = структура(добавки=(("MG", (7.5, 0.0, 0.0)),))
    соседи = neighbours_within(лиганд, parse_atoms(текст), 8.0)
    assert [с.resname for с in соседи] == ["MG"]
