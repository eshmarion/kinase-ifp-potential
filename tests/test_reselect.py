"""Проверки перевыбора выборок с учётом условий DiffSBDD (`src/kinase_ifp/reselect.py`).

Сети нет: классификация подменяется на месте, потому что проверяется не она (её держит
`test_pocket_content.py`), а порядок перебора — то, что берётся **первая годная**
в прежнем порядке ранжирования, а не просто первая попавшаяся.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kinase_ifp import reselect
from kinase_ifp.config import CALIBRATION_KINASES
from kinase_ifp.klifs import KlifsDataError, filter_structures, read_fingerprint_table
from kinase_ifp.pocket_content import PocketContentError, PocketVerdict
from kinase_ifp.reselect import (
    candidates_in_rank_order,
    first_matching,
    read_reselected_sample,
    reselect_calibration,
    select_reselected,
)


def таблица(строки: list[dict[str, object]]) -> pd.DataFrame:
    """Минимальная таблица структур KLIFS: только колонки, нужные ранжированию."""
    return pd.DataFrame(строки)


def строка(
    pdb: str,
    *,
    разрешение: float = 2.0,
    качество: float = 8.0,
    klifs_id: int = 1,
    модель: str = "-",
    лиганд: str = "LIG",
    киназа: str = "CK2a2",
) -> dict[str, object]:
    return {
        "structure.pdb_id": pdb,
        "structure.resolution": разрешение,
        "structure.qualityscore": качество,
        "structure.klifs_id": klifs_id,
        "structure.alternate_model": модель,
        "ligand.expo_id": лиганд,
        "kinase.klifs_name": киназа,
    }


def вердикт(pdb: str, *, годен: bool) -> PocketVerdict:
    return PocketVerdict(
        pdb_id=pdb,
        expo_id="LIG",
        pocket_class="diffsbdd-ready" if годен else "metal",
        matches_diffsbdd=годен,
        reasons=() if годен else ("ион металла в кармане: MG",),
        ligand_atoms=20,
        ligand_copies=1,
        radius=8.0,
        qed=0.5,
        water_atoms=0,
    )


@pytest.fixture
def подменить(monkeypatch: pytest.MonkeyPatch):
    """Подменяет классификацию словарём «код структуры → годна ли»."""

    def применить(ответы: dict[str, bool]) -> list[str]:
        спрошено: list[str] = []

        def поддельная(pdb_id: str, expo_id: str, cache_dir: Path, **kwargs: object):
            спрошено.append(pdb_id)
            if pdb_id not in ответы:
                raise PocketContentError(f"структура {pdb_id} не получена")
            return вердикт(pdb_id, годен=ответы[pdb_id])

        monkeypatch.setattr(reselect, "classify_structure", поддельная)
        return спрошено

    return применить


def test_порядок_кандидатов_совпадает_с_ранжированием() -> None:
    """Порядок не переписан, а взят у `rank_structures`: лучшее разрешение первым."""
    df = таблица(
        [
            строка("2aaa", разрешение=2.5, klifs_id=2),
            строка("1aaa", разрешение=0.9, klifs_id=1),
            строка("3aaa", разрешение=1.8, klifs_id=3),
        ]
    )
    коды = [код for код, _, _ in candidates_in_rank_order(df)]
    assert коды == ["1aaa", "3aaa", "2aaa"]


def test_повторы_кода_структуры_не_проверяются_дважды() -> None:
    """Разные записи KLIFS одной структуры дают один файл RCSB: загрузка не тратится."""
    df = таблица(
        [
            строка("1aaa", klifs_id=1, модель="A"),
            строка("1aaa", klifs_id=2, модель="B"),
        ]
    )
    assert [код for код, _, _ in candidates_in_rank_order(df)] == ["1aaa"]


def test_берётся_первая_годная_а_не_первая_попавшаяся(подменить) -> None:
    df = таблица(
        [
            строка("1aaa", разрешение=0.9, klifs_id=1),
            строка("2aaa", разрешение=1.5, klifs_id=2),
            строка("3aaa", разрешение=1.9, klifs_id=3),
        ]
    )
    спрошено = подменить({"1aaa": False, "2aaa": False, "3aaa": True})
    итог = first_matching(candidates_in_rank_order(df), Path("кэш"))
    assert итог.chosen is not None
    assert итог.chosen.pdb_id == "3aaa"
    assert итог.checked == 3
    assert спрошено == ["1aaa", "2aaa", "3aaa"], "порядок перебора нарушен"
    assert [о.pdb_id for о in итог.rejected] == ["1aaa", "2aaa"]


def test_видно_что_прежний_первый_кандидат_сменился(подменить) -> None:
    """Без этого «взяли вот эту» не отличить от «первая же подошла»."""
    df = таблица([строка("1aaa", разрешение=0.9), строка("2aaa", разрешение=1.5, klifs_id=2)])
    подменить({"1aaa": False, "2aaa": True})
    итог = first_matching(candidates_in_rank_order(df), Path("кэш"))
    assert итог.changed_from == "1aaa"


def test_если_первая_же_годна_смены_не_показывается(подменить) -> None:
    df = таблица([строка("1aaa")])
    подменить({"1aaa": True})
    итог = first_matching(candidates_in_rank_order(df), Path("кэш"))
    assert итог.changed_from is None


def test_предохранитель_ограничивает_число_загрузок(подменить) -> None:
    df = таблица([строка(f"{i}aaa", разрешение=1.0 + i / 10, klifs_id=i) for i in range(1, 9)])
    подменить({f"{i}aaa": i == 8 for i in range(1, 9)})
    итог = first_matching(candidates_in_rank_order(df), Path("кэш"), max_candidates=3)
    assert итог.chosen is None, "предохранитель не сработал"
    assert итог.checked == 3


def test_несостоявшаяся_загрузка_не_считается_отказом(подменить) -> None:
    """Структура, которую не удалось получить, попадает в `failed`, а не в `rejected`."""
    df = таблица([строка("1aaa", разрешение=0.9), строка("2aaa", разрешение=1.5, klifs_id=2)])
    подменить({"2aaa": True})  # про 1aaa словарь молчит — подделка бросит ошибку
    итог = first_matching(candidates_in_rank_order(df), Path("кэш"))
    assert итог.chosen is not None and итог.chosen.pdb_id == "2aaa"
    assert итог.rejected == []
    assert len(итог.failed) == 1


def test_киназа_без_годной_структуры_остаётся_в_выборке_пустой(подменить) -> None:
    """Молча уменьшать выборку нельзя: медиана по 11 киназам и по 12 — разные числа."""
    df = таблица(
        [
            строка("1aaa", киназа="CK2a2"),
            строка("2aaa", киназа="ALK2", klifs_id=2),
        ]
    )
    подменить({"1aaa": True, "2aaa": False})
    итоги = reselect_calibration(df, {1, 2}, ("CK2a2", "ALK2"), Path("кэш"))
    assert set(итоги) == {"CK2a2", "ALK2"}
    assert итоги["CK2a2"].chosen is not None
    assert итоги["ALK2"].chosen is None


def test_структура_без_эталонного_отпечатка_не_скачивается(подменить) -> None:
    """Сверка с KLIFS — смысл работы; мишень без эталона бесполезна и до загрузки."""
    df = таблица([строка("1aaa", klifs_id=1), строка("2aaa", klifs_id=2, разрешение=2.5)])
    спрошено = подменить({"1aaa": True, "2aaa": True})
    итоги = reselect_calibration(df, {2}, ("CK2a2",), Path("кэш"))
    assert спрошено == ["2aaa"], "структура без эталона всё-таки скачивалась"
    assert итоги["CK2a2"].chosen is not None


class TestСоставИзФайла:
    """Чтение записанного состава перевыбора: `read_reselected_sample`."""

    def test_коды_читаются_в_порядке_файла(self, tmp_path: Path) -> None:
        путь = tmp_path / "готовые.csv"
        путь.write_text(
            "pdb_id,expo_id,matches_diffsbdd\n5lvm,ADE,1\n4hyi,1AO,1\n", encoding="utf-8"
        )

        assert read_reselected_sample(путь) == ["5lvm", "4hyi"]

    def test_файла_нет_сообщение_называет_команду_пересборки(self, tmp_path: Path) -> None:
        with pytest.raises(KlifsDataError, match="classify_complexes"):
            read_reselected_sample(tmp_path / "нет-такого.csv")

    def test_негодная_строка_роняет_чтение(self, tmp_path: Path) -> None:
        """Тот же скрипт ключом --calibration пишет файл той же схемы с негодными строками."""
        путь = tmp_path / "готовые.csv"
        путь.write_text(
            "pdb_id,expo_id,matches_diffsbdd\n5lvm,ADE,1\n5lvo,ATP,0\n", encoding="utf-8"
        )

        with pytest.raises(KlifsDataError, match="5lvo"):
            read_reselected_sample(путь)

    def test_повтор_кода_роняет_чтение(self, tmp_path: Path) -> None:
        путь = tmp_path / "готовые.csv"
        путь.write_text(
            "pdb_id,expo_id,matches_diffsbdd\n5lvm,ADE,1\n5lvm,ADE,1\n", encoding="utf-8"
        )

        with pytest.raises(KlifsDataError, match="5lvm"):
            read_reselected_sample(путь)

    def test_файл_без_колонки_кодов_роняет_чтение(self, tmp_path: Path) -> None:
        путь = tmp_path / "готовые.csv"
        путь.write_text("kinase,expo_id\nPDK1,ADE\n", encoding="utf-8")

        with pytest.raises(KlifsDataError, match="pdb_id"):
            read_reselected_sample(путь)


class TestВыборкаПоСоставу:
    """Мост «записанный состав → таблица для `calibrate_sample`»: `select_reselected`."""

    def test_строки_идут_в_порядке_названного_состава(self) -> None:
        df = таблица(
            [
                строка("1aaa", klifs_id=1, киназа="CK2a2"),
                строка("2bbb", klifs_id=2, киназа="ALK2"),
            ]
        )

        итог = select_reselected(df, ["2bbb", "1aaa"], {1, 2})

        assert list(итог["structure.pdb_id"]) == ["2bbb", "1aaa"]
        assert list(итог["kinase.klifs_name"]) == ["ALK2", "CK2a2"]

    def test_берётся_лучшая_запись_кода_по_прежнему_ранжированию(self) -> None:
        """У одного кода несколько записей KLIFS: модели, цепи. Правило отбора прежнее."""
        df = таблица(
            [
                строка("1aaa", klifs_id=1, модель="B", разрешение=1.3),
                строка("1aaa", klifs_id=2, модель="A", разрешение=1.3),
            ]
        )

        итог = select_reselected(df, ["1aaa"], {1, 2})

        assert list(итог["structure.klifs_id"]) == [2], "взята не модель A"

    def test_отсутствующий_код_роняет_вызов(self) -> None:
        df = таблица([строка("1aaa", klifs_id=1)])

        with pytest.raises(KlifsDataError, match="9zzz"):
            select_reselected(df, ["1aaa", "9zzz"], {1})

    def test_код_без_эталонного_отпечатка_роняет_вызов(self) -> None:
        df = таблица([строка("1aaa", klifs_id=1), строка("2bbb", klifs_id=2, киназа="ALK2")])

        with pytest.raises(KlifsDataError, match="2bbb"):
            select_reselected(df, ["1aaa", "2bbb"], {1})

    def test_две_структуры_одной_киназы_роняют_вызов(self) -> None:
        """Киназа, вошедшая дважды, входит в медиану дважды."""
        df = таблица(
            [
                строка("1aaa", klifs_id=1, киназа="CK2a2"),
                строка("2bbb", klifs_id=2, киназа="CK2a2"),
            ]
        )

        with pytest.raises(KlifsDataError, match="CK2a2"):
            select_reselected(df, ["1aaa", "2bbb"], {1, 2})

    def test_недостающая_киназа_замечается(self) -> None:
        df = таблица([строка("1aaa", klifs_id=1, киназа="CK2a2")])

        with pytest.raises(KlifsDataError, match="ALK2"):
            select_reselected(df, ["1aaa"], {1}, expect_kinases=("CK2a2", "ALK2"))

    def test_лишняя_киназа_замечается(self) -> None:
        df = таблица(
            [
                строка("1aaa", klifs_id=1, киназа="CK2a2"),
                строка("2bbb", klifs_id=2, киназа="ALK2"),
            ]
        )

        with pytest.raises(KlifsDataError, match="ALK2"):
            select_reselected(df, ["1aaa", "2bbb"], {1, 2}, expect_kinases=("CK2a2",))


def test_мост_воспроизводит_перевыбор_без_обращения_к_rcsb() -> None:
    """Состав, полученный из файла, обязан совпасть с тем, что дал перевыбор по сети.

    Записи KLIFS прибиты числами: перевыбор 14.09 шёл с загрузкой структур с RCSB,
    а здесь тот же состав восстанавливается по таблице структур. Разойдись эти два
    пути — сравнение «было/стало» считалось бы не на той выборке, о которой написано.
    """
    корень = Path(__file__).resolve().parents[1]
    структуры = filter_structures(pd.read_csv(корень / "data" / "klifs" / "structures.csv"))
    эталоны = set(
        int(i)
        for i in read_fingerprint_table(корень / "data" / "klifs" / "klifs_ifp.csv")[
            "structure_id"
        ]
    )
    коды = read_reselected_sample(корень / "data" / "klifs" / "calibration_diffsbdd_ready.csv")

    выборка = select_reselected(структуры, коды, эталоны, expect_kinases=CALIBRATION_KINASES)

    assert list(выборка["structure.klifs_id"]) == [
        7112, 1538, 6164, 5429, 12448, 2605, 11418, 11801, 8079, 2892, 10344, 15267
    ]
    assert set(выборка["structure.alternate_model"]) == {"A"}
