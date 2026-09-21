"""Тесты реконструкции правила `apolar` KLIFS.

Проверяется не «код отработал», а три вещи, на которых реконструкция стоит:
что биты эталона читаются по закреплённой раскладке;
что правило считается по тяжёлым атомам выбранных элементов и порогу;
и что сравнение альтернативных моделей отличает совпадение от расхождения —
именно оно 16.09 опровергло поспешный вывод, сделанный по двум структурам.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kinase_ifp.apolar_rule import (
    ApolarRuleError,
    Atom,
    aggregate,
    apolar_bits_by_rule,
    compare_alternate_models,
    has_contact,
    parse_pdb_atoms,
    parse_sdf_atoms,
    reference_apolar_bits,
    score_rules,
    score_rules_per_structure,
    select_atoms,
)
from kinase_ifp.config import APOLAR_ATOM_ELEMENTS, KLIFS_BITS_SHAPE

ФИКСТУРА = Path(__file__).resolve().parent / "fixtures" / "6tgu"
ПАКЕТ = ФИКСТУРА / "target.json"


@pytest.fixture(scope="module")
def эталон_6tgu() -> str:
    return json.loads(ПАКЕТ.read_text(encoding="utf-8"))["klifs_ifp_bits"]


def test_биты_эталона_читаются_по_позиционной_раскладке() -> None:
    """Раскладка (85, 7): `apolar` — первый бит позиции."""
    позиций, типов = KLIFS_BITS_SHAPE
    строка = ["0"] * (позиций * типов)
    строка[0] = "1"  # apolar позиции 1
    строка[(3 - 1) * типов + 1] = "1"  # F-F позиции 3, не apolar
    биты = reference_apolar_bits("".join(строка))
    assert биты[1] is True
    assert биты[3] is False
    assert sum(биты.values()) == 1


def test_эталон_неверной_длины_отвергается() -> None:
    with pytest.raises(ApolarRuleError, match="ожидается"):
        reference_apolar_bits("0101")


def test_водороды_не_читаются_ни_из_белка_ни_из_лиганда() -> None:
    """KLIFS считает по тяжёлым атомам: водороды дают +2 бита и 69 ложных."""
    белок = parse_pdb_atoms(ФИКСТУРА / "protein.pdb")
    лиганд = parse_sdf_atoms(ФИКСТУРА / "ligand.sdf")
    все_атомы = [а for атомы in белок.values() for а in атомы]
    assert все_атомы, "в фикстуре нет атомов белка"
    assert not [а for а in все_атомы if а.element == "H"]
    assert not [а for а in лиганд if а.element == "H"]


def test_ключ_остатка_совпадает_с_разметкой_позиций() -> None:
    """Сопоставление с позициями держится на совпадении ключей, а не на таблице."""
    белок = parse_pdb_atoms(ФИКСТУРА / "protein.pdb")
    разметка = json.loads(ПАКЕТ.read_text(encoding="utf-8"))["residue_to_position"]
    общие = set(белок) & set(разметка)
    assert len(общие) == len(разметка), (
        f"из {len(разметка)} остатков кармана в структуре найдено {len(общие)}"
    )


def test_отбор_по_элементам_и_пустой_набор() -> None:
    атомы = [Atom("C", (0.0, 0.0, 0.0)), Atom("S", (1.0, 0.0, 0.0)), Atom("N", (2.0, 0.0, 0.0))]
    assert len(select_atoms(атомы, ("C",))) == 1
    assert len(select_atoms(атомы, ("C", "S"))) == 2
    assert len(select_atoms(атомы, ())) == 3, "пустой набор — любой тяжёлый атом"


def test_контакт_считается_по_порогу_включительно() -> None:
    левые = [Atom("C", (0.0, 0.0, 0.0))]
    правые = [Atom("C", (4.5, 0.0, 0.0))]
    assert has_contact(левые, правые, 4.5) is True
    assert has_contact(левые, правые, 4.4) is False


def test_правило_воспроизводит_эталон_на_фикстуре(эталон_6tgu: str) -> None:
    """На 6tgu правило даёт все эталонные биты `apolar` и ни одного лишнего."""
    эталон = reference_apolar_bits(эталон_6tgu)
    наши = apolar_bits_by_rule(ПАКЕТ, APOLAR_ATOM_ELEMENTS)
    воспроизведено = sum(1 for поз in эталон if эталон[поз] and наши[поз])
    лишних = sum(1 for поз in эталон if наши[поз] and not эталон[поз])
    assert sum(эталон.values()) == 12, "у 6tgu 12 эталонных бит apolar"
    assert воспроизведено == 12
    assert лишних == 0


def test_score_rules_считает_полноту_и_точность(эталон_6tgu: str) -> None:
    итоги = score_rules({ПАКЕТ: эталон_6tgu}, {"правило": APOLAR_ATOM_ELEMENTS})
    assert len(итоги) == 1
    итог = итоги[0]
    assert итог.reference_total == 12
    assert итог.recall == pytest.approx(1.0)
    assert итог.precision == pytest.approx(1.0)


def test_слишком_большой_порог_даёт_лишние_биты(эталон_6tgu: str) -> None:
    """Порог не безразличен: за 4.5 A появляются срабатывания вне эталона."""
    итоги = score_rules({ПАКЕТ: эталон_6tgu}, {"правило": APOLAR_ATOM_ELEMENTS}, distance=7.0)
    assert итоги[0].extra > 0


def test_сравнение_моделей_отличает_совпадение_от_расхождения() -> None:
    длина = KLIFS_BITS_SHAPE[0] * KLIFS_BITS_SHAPE[1]
    одинаковые = "1" + "0" * (длина - 1)
    другие = "0" * (длина - 1) + "1"
    итог = compare_alternate_models(
        {1: одинаковые, 2: одинаковые, 3: одинаковые, 4: другие},
        [
            (1, "aaaa", "A", "A"),
            (2, "aaaa", "A", "B"),
            (3, "bbbb", "A", "A"),
            (4, "bbbb", "A", "B"),
        ],
    )
    assert итог["пар с несколькими моделями"] == 2
    assert итог["отпечатки совпали"] == 1
    assert итог["отпечатки разошлись"] == 1
    assert итог["различающихся бит всего"] == 2


def test_структура_без_второй_модели_в_пары_не_идёт() -> None:
    длина = KLIFS_BITS_SHAPE[0] * KLIFS_BITS_SHAPE[1]
    биты = "1" + "0" * (длина - 1)
    итог = compare_alternate_models({1: биты}, [(1, "aaaa", "A", "-")])
    assert итог["пар с несколькими моделями"] == 0


def test_отсутствующий_файл_сообщает_причину(tmp_path: Path) -> None:
    with pytest.raises(ApolarRuleError, match="нет файла"):
        parse_pdb_atoms(tmp_path / "нет.pdb")


def test_разбивка_по_структурам_считает_позиции_и_отрицательные(эталон_6tgu: str) -> None:
    """У структуры 85 позиций; те, где у эталона нет бита, — отрицательные случаи."""
    итоги = score_rules_per_structure({ПАКЕТ: эталон_6tgu}, {"правило": APOLAR_ATOM_ELEMENTS})
    assert len(итоги) == 1
    итог = итоги[0]
    assert итог.label == "6tgu", "метка берётся из имени каталога пакета"
    assert итог.positions == KLIFS_BITS_SHAPE[0]
    assert итог.reference_total == 12
    assert итог.negatives == KLIFS_BITS_SHAPE[0] - 12
    assert итог.recall == pytest.approx(1.0)


def test_сводка_складывается_из_разбивки_без_второго_прохода(эталон_6tgu: str) -> None:
    """Сводное и по-структурное числа считаются одним проходом и не могут разойтись."""
    правила = {"первое": APOLAR_ATOM_ELEMENTS, "второе": ("C",)}
    по_структурам = score_rules_per_structure({ПАКЕТ: эталон_6tgu}, правила)
    сводка = aggregate(по_структурам, правила)
    assert [и.rule for и in сводка] == ["первое", "второе"]
    for итог in сводка:
        свои = [и for и in по_структурам if и.rule == итог.rule]
        assert итог.reproduced == sum(и.reproduced for и in свои)
        assert итог.extra == sum(и.extra for и in свои)
        assert итог.reference_total == sum(и.reference_total for и in свои)
