"""Тесты выгрузки KLIFS и пакета мишени.

В KLIFS эти тесты не ходят: тест, зависящий от чужого сервера, падает от недоступности
сети и не воспроизводится. Проверяется то, что действительно наше, — правила отбора
и схема `target.json`. Сверка с живой базой делается запуском скриптов, а не pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from kinase_ifp import klifs as klifs_module
from kinase_ifp.config import (
    INHIBITOR_TYPE_I,
    INHIBITOR_TYPE_I5,
    INHIBITOR_TYPE_II,
    INHIBITOR_TYPE_UNKNOWN,
    KLIFS_FINGERPRINT_BATCH_SIZE,
    KLIFS_IFP_LENGTH,
    KLIFS_KINASE_BATCH_SIZE,
    LIGAND_CLASS_INHIBITOR_LIKE,
    LIGAND_CLASS_NONE,
    LIGAND_CLASS_NUCLEOTIDE,
    LIGAND_CLASSES,
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
    N_KLIFS_POSITIONS,
    NUCLEOTIDE_LIGAND_CODES,
    TARGET_LIGAND_CLASSES,
)
from kinase_ifp.klifs import (
    INHIBITOR_TYPE_COLUMN,
    LIGAND_CLASS_COLUMN,
    KlifsDataError,
    annotate_inhibitor_type,
    annotate_kinase_taxonomy,
    annotate_ligand_class,
    build_target_package,
    classify_inhibitor_type,
    classify_ligand,
    count_by_inhibitor_type,
    count_by_ligand_class,
    fetch_fingerprint_table,
    fetch_fingerprints,
    fetch_kinase_taxonomy,
    filter_structures,
    is_valid_fingerprint,
    merge_fingerprint_tables,
    read_fingerprint_table,
    select_best_per_kinase,
    select_target,
    summarize_kinases,
    validate_target_json,
)

VALID_BITS = "1" + "0" * (KLIFS_IFP_LENGTH - 1)

# Карман фикстуры: позиция → однобуквенный код остатка. Те же позиции стоят
# в residue_to_position ниже, и это не совпадение: сборка пакета сверяет обе стороны
# поштучно, а не только по числу занятых. Прежняя фикстура ставила буквы
# на позиции 1 и 2 при карте на 17 и 24 — счёт сходился, места не совпадали.
POCKET_RESIDUES = {17: "K", 24: "E"}

# Последовательность кармана: 85 символов, по одному на позицию выравнивания.
POCKET_SEQUENCE = "".join(
    POCKET_RESIDUES.get(позиция, "_") for позиция in range(1, N_KLIFS_POSITIONS + 1)
)


def make_structure_row(**overrides: Any) -> dict[str, Any]:
    """Строка таблицы структур KLIFS, проходящая все фильтры."""
    row: dict[str, Any] = {
        "structure.klifs_id": 1000,
        "structure.pdb_id": "1abc",
        "structure.chain": "A",
        "structure.resolution": 2.0,
        "structure.qualityscore": 8.0,
        "species.klifs": "Human",
        "ligand.expo_id": "STU",
        "kinase.klifs_name": "TestKinase",
        "structure.dfg": "in",
        "structure.ac_helix": "in",
        "structure.alternate_model": "-",
        "structure.pocket": POCKET_SEQUENCE,
    }
    row.update(overrides)
    return row


def make_target_package(base: Path, **overrides: Any) -> Path:
    """Создаёт корректный пакет мишени с реально существующими файлами."""
    base.mkdir(parents=True, exist_ok=True)
    for name in ("protein.pdb", "ligand.sdf", "pocket.pdb", "protein_noh.pdb"):
        (base / name).write_text("заглушка", encoding="utf-8")

    package: dict[str, Any] = {
        "klifs_structure_id": 1000,
        "pdb_id": "1abc",
        "chain": "A",
        "kinase_name": "TestKinase",
        "resolution": 2.0,
        "quality_score": 8.0,
        "protein_pdb": "protein.pdb",
        "ligand_sdf": "ligand.sdf",
        "pocket_pdb": "pocket.pdb",
        "protein_noh_pdb": "protein_noh.pdb",
        "protonation": "implicit-prolif",
        "protonation_tool": None,
        "residue_to_position": {"LYS33.A": 17, "GLU51.A": 24},
        "klifs_ifp_bits": VALID_BITS,
        "dfg": "in",
        "ac_helix": "in",
        "inhibitor_type": "I",
        "alternate_model": "A",
        "pocket_sequence": POCKET_SEQUENCE,
    }
    package.update(overrides)

    path = base / "target.json"
    path.write_text(json.dumps(package, ensure_ascii=False), encoding="utf-8")
    return path


class TestClassifyInhibitorType:
    @pytest.mark.parametrize(
        ("dfg", "ac_helix", "expected"),
        [
            ("in", "in", INHIBITOR_TYPE_I),
            ("in", "out", INHIBITOR_TYPE_I5),
            ("out", "in", INHIBITOR_TYPE_II),
            ("out", "out", INHIBITOR_TYPE_II),
            # Промежуточная конформация и незаполненные поля: тип не определён.
            # Такая структура не должна выдаваться за определённый тип.
            ("out-like", "in", INHIBITOR_TYPE_UNKNOWN),
            ("in", "na", INHIBITOR_TYPE_UNKNOWN),
            ("na", "in", INHIBITOR_TYPE_UNKNOWN),
            ("-", "-", INHIBITOR_TYPE_UNKNOWN),
            (None, None, INHIBITOR_TYPE_UNKNOWN),
        ],
    )
    def test_тип_по_конформации(self, dfg: Any, ac_helix: Any, expected: str) -> None:
        assert classify_inhibitor_type(dfg, ac_helix) == expected

    def test_регистр_и_пробелы_не_влияют(self) -> None:
        assert classify_inhibitor_type(" IN ", "In") == INHIBITOR_TYPE_I


class TestAnnotateAndCount:
    def test_размечает_всю_таблицу(self) -> None:
        rows = [
            make_structure_row(**{"structure.dfg": "in", "structure.ac_helix": "in"}),
            make_structure_row(**{"structure.dfg": "in", "structure.ac_helix": "out"}),
            make_structure_row(**{"structure.dfg": "out", "structure.ac_helix": "in"}),
        ]
        annotated = annotate_inhibitor_type(pd.DataFrame(rows))
        assert list(annotated[INHIBITOR_TYPE_COLUMN]) == [
            INHIBITOR_TYPE_I,
            INHIBITOR_TYPE_I5,
            INHIBITOR_TYPE_II,
        ]

    def test_не_портит_исходную_таблицу(self) -> None:
        df = pd.DataFrame([make_structure_row()])
        annotate_inhibitor_type(df)
        assert INHIBITOR_TYPE_COLUMN not in df.columns

    def test_считает_распределение_по_типам(self) -> None:
        rows = [
            make_structure_row(**{"structure.dfg": "in", "structure.ac_helix": "in"}),
            make_structure_row(**{"structure.dfg": "in", "structure.ac_helix": "in"}),
            make_structure_row(**{"structure.dfg": "out", "structure.ac_helix": "in"}),
        ]
        assert count_by_inhibitor_type(pd.DataFrame(rows)) == {
            INHIBITOR_TYPE_I: 2,
            INHIBITOR_TYPE_II: 1,
        }


class TestClassifyLigand:
    """Класс лиганда в ортостерическом кармане."""

    @pytest.mark.parametrize(
        ("expo_id", "expected"),
        [
            ("ATP", LIGAND_CLASS_NUCLEOTIDE),
            ("ANP", LIGAND_CLASS_NUCLEOTIDE),
            ("ADN", LIGAND_CLASS_NUCLEOTIDE),
            # Стауроспорин — неселективный, но ингибитор, и это не субстрат.
            ("STU", LIGAND_CLASS_INHIBITOR_LIKE),
            ("LU8", LIGAND_CLASS_INHIBITOR_LIKE),
            # Маркеры пропуска KLIFS: поле не заполнено, а не «лиганд с таким кодом».
            ("", LIGAND_CLASS_NONE),
            ("-", LIGAND_CLASS_NONE),
            ("_", LIGAND_CLASS_NONE),
            (None, LIGAND_CLASS_NONE),
        ],
    )
    def test_класс_по_коду(self, expo_id: Any, expected: str) -> None:
        assert classify_ligand(expo_id) == expected

    def test_регистр_и_пробелы_не_влияют(self) -> None:
        assert classify_ligand(" atp ") == LIGAND_CLASS_NUCLEOTIDE

    def test_список_классов_закрыт(self) -> None:
        """Любой результат принадлежит `LIGAND_CLASSES`, и список ровно из трёх значений.

        Тест держит допущение, а не поведение: новый класс нельзя завести, не тронув
        и список, и это место, — а значит, не объяснив границу в тексте.
        """
        assert set(LIGAND_CLASSES) == {
            LIGAND_CLASS_NUCLEOTIDE,
            LIGAND_CLASS_INHIBITOR_LIKE,
            LIGAND_CLASS_NONE,
        }
        коды = [*NUCLEOTIDE_LIGAND_CODES, "STU", "LU8", "XYZ", "", "-", None]
        assert all(classify_ligand(код) in LIGAND_CLASSES for код in коды)

    def test_подмена_кода_нуклеотида(self) -> None:
        """Нуклеотидом считается только то, что в `NUCLEOTIDE_LIGAND_CODES`.

        Падает, если список подменить произвольным: это ровно то допущение, которое
        придётся защищать в тексте, и оно не должно меняться незаметно.
        """
        for код in NUCLEOTIDE_LIGAND_CODES:
            assert classify_ligand(код) == LIGAND_CLASS_NUCLEOTIDE
        # Код, которого в списке нет, нуклеотидом объявлен быть не может.
        assert "XYZ" not in NUCLEOTIDE_LIGAND_CODES
        assert classify_ligand("XYZ") == LIGAND_CLASS_INHIBITOR_LIKE
        # Кристаллизационные добавки отдельным классом не заведены — по измерению:
        # в ортостерическом поле выборки их ноль. Попадут в `inhibitor-like`.
        assert classify_ligand("GOL") == LIGAND_CLASS_INHIBITOR_LIKE

    def test_размечает_всю_таблицу_не_трогая_исходную(self) -> None:
        rows = [
            make_structure_row(**{"ligand.expo_id": "ATP"}),
            make_structure_row(**{"ligand.expo_id": "STU"}),
            make_structure_row(**{"ligand.expo_id": "-"}),
        ]
        df = pd.DataFrame(rows)
        annotated = annotate_ligand_class(df)
        assert list(annotated[LIGAND_CLASS_COLUMN]) == [
            LIGAND_CLASS_NUCLEOTIDE,
            LIGAND_CLASS_INHIBITOR_LIKE,
            LIGAND_CLASS_NONE,
        ]
        assert LIGAND_CLASS_COLUMN not in df.columns
        assert count_by_ligand_class(df) == {
            LIGAND_CLASS_NUCLEOTIDE: 1,
            LIGAND_CLASS_INHIBITOR_LIKE: 1,
            LIGAND_CLASS_NONE: 1,
        }

    def test_без_колонки_лиганда_ошибка(self) -> None:
        df = pd.DataFrame([{"structure.pdb_id": "1abc"}])
        with pytest.raises(KlifsDataError):
            annotate_ligand_class(df)


class TestFilterStructures:
    def test_проходит_структура_без_нарушений(self) -> None:
        df = pd.DataFrame([make_structure_row()])
        assert len(filter_structures(df)) == 1

    def test_умолчание_не_отсекает_нуклеотиды(self) -> None:
        """Без явного аргумента комплекс с АТФ проходит — поведение прежнее.

        Это защита уже посчитанных чисел: на умолчании сидят выбор мишени 6tgu
        и выборка калибровки, где два комплекса из двенадцати нуклеотидные.
        Отсечение в умолчании сдвинуло бы их молча.
        """
        df = pd.DataFrame([make_structure_row(**{"ligand.expo_id": "ATP"})])
        отобранные = filter_structures(df)
        assert len(отобранные) == 1
        assert отобранные[LIGAND_CLASS_COLUMN].iloc[0] == LIGAND_CLASS_NUCLEOTIDE

    def test_явный_аргумент_оставляет_только_ингибиторные(self) -> None:
        df = pd.DataFrame(
            [
                make_structure_row(**{"ligand.expo_id": "ATP"}),
                make_structure_row(**{"ligand.expo_id": "STU"}),
            ]
        )
        отобранные = filter_structures(df, ligand_classes=TARGET_LIGAND_CLASSES)
        assert list(отобранные["ligand.expo_id"]) == ["STU"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("structure.resolution", MAX_RESOLUTION + 0.1),
            ("structure.qualityscore", MIN_QUALITY_SCORE - 0.1),
            ("species.klifs", "Mouse"),
            ("ligand.expo_id", "-"),
            # Решение 22.08.2026: работаем только по типу I.
            ("structure.dfg", "out"),          # тип II
            ("structure.ac_helix", "out"),     # тип I.5
            ("structure.ac_helix", "na"),      # тип не определён
        ],
    )
    def test_отбрасывает_нарушителя(self, field: str, value: Any) -> None:
        df = pd.DataFrame([make_structure_row(**{field: value})])
        assert filter_structures(df).empty

    def test_границы_порогов_включительны(self) -> None:
        # Структура ровно на пороге годится: пороги заданы как «не хуже», а не «лучше».
        df = pd.DataFrame(
            [
                make_structure_row(
                    **{
                        "structure.resolution": MAX_RESOLUTION,
                        "structure.qualityscore": MIN_QUALITY_SCORE,
                    }
                )
            ]
        )
        assert len(filter_structures(df)) == 1

    def test_ни_одна_прошедшая_строка_не_нарушает_фильтры(self) -> None:
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.resolution": 1.5}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.resolution": 3.5}),
            make_structure_row(**{"structure.klifs_id": 3, "structure.qualityscore": 4.0}),
            make_structure_row(**{"structure.klifs_id": 4, "species.klifs": "Mouse"}),
        ]
        result = filter_structures(pd.DataFrame(rows))

        assert (result["structure.resolution"] <= MAX_RESOLUTION).all()
        assert (result["structure.qualityscore"] >= MIN_QUALITY_SCORE).all()
        assert (result["species.klifs"] == "Human").all()

    def test_другой_тип_берётся_аргументом(self) -> None:
        # Выборка типа II нужна для второй мишени и для чисел обзора; получать её
        # правкой кода было бы неудобно, поэтому список типов — аргумент.
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.dfg": "in"}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.dfg": "out"}),
        ]
        result = filter_structures(pd.DataFrame(rows), inhibitor_types=(INHIBITOR_TYPE_II,))
        assert list(result["structure.klifs_id"]) == [2]

    def test_добавляет_колонку_типа(self) -> None:
        result = filter_structures(pd.DataFrame([make_structure_row()]))
        assert list(result[INHIBITOR_TYPE_COLUMN]) == [INHIBITOR_TYPE_I]

    def test_падает_при_отсутствии_колонки(self) -> None:
        df = pd.DataFrame([make_structure_row()]).drop(columns=["structure.qualityscore"])
        with pytest.raises(KlifsDataError, match="structure.qualityscore"):
            filter_structures(df)

    def test_вид_можно_не_ограничивать(self) -> None:
        # Киназы других организмов пригодны для той же задачи; species=None нужен,
        # чтобы посмотреть на полный список кандидатов до выбора мишени.
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "species.klifs": "Human"}),
            make_structure_row(**{"structure.klifs_id": 2, "species.klifs": "Mouse"}),
        ]
        result = filter_structures(pd.DataFrame(rows), species=None)
        assert sorted(result["structure.klifs_id"]) == [1, 2]


class TestSummarizeKinases:
    def test_группирует_по_киназе_и_виду(self) -> None:
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "kinase.klifs_name": "EGFR"}),
            make_structure_row(**{"structure.klifs_id": 2, "kinase.klifs_name": "EGFR"}),
            make_structure_row(**{"structure.klifs_id": 3, "kinase.klifs_name": "CDK2"}),
        ]
        summary = summarize_kinases(pd.DataFrame(rows))

        assert len(summary) == 2
        egfr = summary[summary["kinase.klifs_name"] == "EGFR"].iloc[0]
        assert egfr["n_structures"] == 2

    def test_одна_киназа_разных_видов_это_разные_строки(self) -> None:
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "species.klifs": "Human"}),
            make_structure_row(**{"structure.klifs_id": 2, "species.klifs": "Mouse"}),
        ]
        assert len(summarize_kinases(pd.DataFrame(rows))) == 2

    def test_лучшая_киназа_идёт_первой(self) -> None:
        rows = [
            make_structure_row(
                **{"kinase.klifs_name": "CDK2", "structure.resolution": 2.5}
            ),
            make_structure_row(
                **{"kinase.klifs_name": "EGFR", "structure.resolution": 1.1}
            ),
        ]
        summary = summarize_kinases(pd.DataFrame(rows))
        assert summary.iloc[0]["kinase.klifs_name"] == "EGFR"
        assert summary.iloc[0]["best_resolution"] == 1.1


class FakeInteractions:
    """Заглушка `session.interactions`: отдаёт заранее заданную таблицу."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.requested: list[int] | None = None
        # Все вызовы подряд: по ним проверяется разбиение запроса на пачки.
        self.calls: list[list[int]] = []

    def by_structure_klifs_id(self, structure_klifs_ids: list[int]) -> pd.DataFrame:
        self.requested = structure_klifs_ids
        self.calls.append(list(structure_klifs_ids))
        return self._df


class FakeKinases:
    """Заглушка `session.kinases`: отдаёт строки только для запрошенных имён.

    Отдавать всю таблицу было бы неверно: настоящий KLIFS возвращает по имени все виды,
    но не чужие киназы, и тест на пачки перестал бы что-либо проверять.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[list[str]] = []

    def by_kinase_name(self, names: list[str]) -> pd.DataFrame:
        self.calls.append(list(names))
        return self._df.loc[self._df["kinase.klifs_name"].isin(names)].copy()


class FakeSession:
    """Заглушка сессии KLIFS: в сеть не ходит.

    `kinases` необязателен: таблица киназ нужна только тестам таксономии, а прежние
    вызовы `FakeSession(df)` обязаны продолжать работать.
    """

    def __init__(self, df: pd.DataFrame, kinases: pd.DataFrame | None = None) -> None:
        self.interactions = FakeInteractions(df)
        self.kinases = FakeKinases(kinases if kinases is not None else pd.DataFrame())


class TestFetchFingerprints:
    def test_берёт_только_валидные_отпечатки(self) -> None:
        df = pd.DataFrame(
            [
                {"structure.klifs_id": 1, "interaction.fingerprint": VALID_BITS},
                {"structure.klifs_id": 2, "interaction.fingerprint": "0101"},
                {"structure.klifs_id": 3, "interaction.fingerprint": None},
            ]
        )
        assert fetch_fingerprints(FakeSession(df), [1, 2, 3]) == {1: VALID_BITS}

    def test_пустой_список_не_идёт_в_сеть(self) -> None:
        session = FakeSession(pd.DataFrame())
        assert fetch_fingerprints(session, []) == {}
        assert session.interactions.requested is None

    def test_падает_при_отсутствии_колонки(self) -> None:
        df = pd.DataFrame([{"structure.klifs_id": 1}])
        with pytest.raises(KlifsDataError, match="interaction.fingerprint"):
            fetch_fingerprints(FakeSession(df), [1])


def make_structures(*klifs_ids: int) -> pd.DataFrame:
    """Таблица структур из перечисленных KLIFS ID: pdb_id выводится из номера."""
    return pd.DataFrame(
        [
            make_structure_row(**{
                "structure.klifs_id": klifs_id,
                "structure.pdb_id": f"p{klifs_id}",
            })
            for klifs_id in klifs_ids
        ]
    )


def make_interactions(*klifs_ids: int) -> pd.DataFrame:
    """Ответ KLIFS: валидный отпечаток для каждой перечисленной структуры."""
    return pd.DataFrame(
        [
            {"structure.klifs_id": klifs_id, "interaction.fingerprint": VALID_BITS}
            for klifs_id in klifs_ids
        ]
    )


class TestFetchFingerprintTable:
    """Таблица эталонных отпечатков для сверки расчёта."""

    def test_каждая_строка_отпечатка_валидна(self) -> None:
        session = FakeSession(make_interactions(1, 2))
        table = fetch_fingerprint_table(session, make_structures(1, 2))

        assert list(table.columns) == ["structure_id", "pdb_id", "bits"]
        assert len(table) == 2
        for bits in table["bits"]:
            assert len(bits) == KLIFS_IFP_LENGTH
            assert set(bits) <= {"0", "1"}

    def test_pdb_id_берётся_из_таблицы_структур(self) -> None:
        session = FakeSession(make_interactions(7))
        table = fetch_fingerprint_table(session, make_structures(7))

        assert table.iloc[0]["structure_id"] == 7
        assert table.iloc[0]["pdb_id"] == "p7"

    def test_структура_без_отпечатка_в_таблицу_не_попадает(self) -> None:
        # KLIFS отдаёт отпечаток не для всякой структуры: это отсутствие данных,
        # а не ошибка расчёта, поэтому строка просто не появляется.
        interactions = pd.DataFrame(
            [
                {"structure.klifs_id": 1, "interaction.fingerprint": VALID_BITS},
                {"structure.klifs_id": 2, "interaction.fingerprint": None},
            ]
        )
        table = fetch_fingerprint_table(FakeSession(interactions), make_structures(1, 2))

        assert list(table["structure_id"]) == [1]

    def test_известные_отпечатки_не_запрашиваются_заново(self) -> None:
        session = FakeSession(make_interactions(1))
        table = fetch_fingerprint_table(session, make_structures(1), known={1: VALID_BITS})

        assert session.interactions.calls == []
        assert list(table["structure_id"]) == [1]

    def test_из_известных_берутся_только_нужные_структуры(self) -> None:
        # В файле эталонов могут лежать структуры, которых нет в текущей выборке:
        # таблица описывает переданные структуры, а не всё содержимое файла.
        session = FakeSession(make_interactions(1))
        table = fetch_fingerprint_table(
            session, make_structures(1), known={1: VALID_BITS, 99: VALID_BITS}
        )

        assert list(table["structure_id"]) == [1]

    def test_запрос_разбивается_на_пачки(self) -> None:
        klifs_ids = list(range(1, KLIFS_FINGERPRINT_BATCH_SIZE + 4))
        session = FakeSession(make_interactions(*klifs_ids))

        fetch_fingerprint_table(session, make_structures(*klifs_ids))

        assert len(session.interactions.calls) == 2
        assert len(session.interactions.calls[0]) == KLIFS_FINGERPRINT_BATCH_SIZE
        assert len(session.interactions.calls[1]) == 3

    def test_лишние_строки_ответа_игнорируются(self) -> None:
        # База может вернуть строку, которой мы не запрашивали. Такая строка в таблицу
        # не идёт: у неё нет `pdb_id`, и раньше на ней функция падала с KeyError.
        session = FakeSession(make_interactions(1, 2))
        table = fetch_fingerprint_table(session, make_structures(1))

        assert list(table["structure_id"]) == [1]

    def test_таблица_отсортирована_по_идентификатору(self) -> None:
        session = FakeSession(make_interactions(5, 2, 9))
        table = fetch_fingerprint_table(session, make_structures(5, 2, 9))

        assert list(table["structure_id"]) == [2, 5, 9]

    def test_падает_при_отсутствии_колонки(self) -> None:
        structures = pd.DataFrame([{"structure.klifs_id": 1}])
        with pytest.raises(KlifsDataError, match="structure.pdb_id"):
            fetch_fingerprint_table(FakeSession(pd.DataFrame()), structures)


class TestFingerprintTableFile:
    """Чтение и дозапись `data/klifs/klifs_ifp.csv`."""

    def test_отсутствующий_файл_даёт_пустую_таблицу(self, tmp_path: Path) -> None:
        table = read_fingerprint_table(tmp_path / "нет.csv")

        assert table.empty
        assert list(table.columns) == ["structure_id", "pdb_id", "bits"]

    def test_прочитанное_совпадает_с_записанным(self, tmp_path: Path) -> None:
        path = tmp_path / "klifs_ifp.csv"
        session = FakeSession(make_interactions(3))
        fetch_fingerprint_table(session, make_structures(3)).to_csv(path, index=False)

        table = read_fingerprint_table(path)

        assert list(table["structure_id"]) == [3]
        assert table.iloc[0]["bits"] == VALID_BITS

    def test_повреждённая_строка_останавливает_чтение(self, tmp_path: Path) -> None:
        # Испорченный эталон обязан остановить работу здесь: иначе он уедет в сверку
        # с нашим расчётом и объяснит расхождение не тем.
        path = tmp_path / "klifs_ifp.csv"
        pd.DataFrame(
            [{"structure_id": 3, "pdb_id": "p3", "bits": "0101"}]
        ).to_csv(path, index=False)

        with pytest.raises(KlifsDataError, match="595"):
            read_fingerprint_table(path)

    def test_дозапись_не_теряет_прежние_строки(self) -> None:
        старое = pd.DataFrame([{"structure_id": 1, "pdb_id": "p1", "bits": VALID_BITS}])
        новое = pd.DataFrame([{"structure_id": 2, "pdb_id": "p2", "bits": VALID_BITS}])

        merged = merge_fingerprint_tables(старое, новое)

        assert list(merged["structure_id"]) == [1, 2]

    def test_свежая_строка_вытесняет_прежнюю(self) -> None:
        свежие_биты = "0" * (KLIFS_IFP_LENGTH - 1) + "1"
        старое = pd.DataFrame([{"structure_id": 1, "pdb_id": "p1", "bits": VALID_BITS}])
        новое = pd.DataFrame([{"structure_id": 1, "pdb_id": "p1", "bits": свежие_биты}])

        merged = merge_fingerprint_tables(старое, новое)

        assert len(merged) == 1
        assert merged.iloc[0]["bits"] == свежие_биты

    def test_дозапись_к_пустой_таблице_сохраняет_тип_идентификатора(self) -> None:
        пустое = read_fingerprint_table(Path("нет.csv"))
        новое = pd.DataFrame([{"structure_id": 2, "pdb_id": "p2", "bits": VALID_BITS}])

        merged = merge_fingerprint_tables(пустое, новое)

        assert merged["structure_id"].dtype == "int64"


def all_fingerprints(*klifs_ids: int) -> dict[int, str]:
    """Словарь отпечатков, в котором есть все перечисленные структуры."""
    return dict.fromkeys(klifs_ids, VALID_BITS)


class TestSelectTarget:
    def test_выбирает_лучшее_разрешение(self) -> None:
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.resolution": 2.5}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.resolution": 1.2}),
        ]
        target = select_target(pd.DataFrame(rows), all_fingerprints(1, 2))
        assert target["structure.klifs_id"] == 2

    def test_при_равном_разрешении_выигрывает_качество(self) -> None:
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.qualityscore": 7.0}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.qualityscore": 9.0}),
        ]
        target = select_target(pd.DataFrame(rows), all_fingerprints(1, 2))
        assert target["structure.klifs_id"] == 2

    def test_выбор_не_зависит_от_порядка_строк(self) -> None:
        # Воспроизводимость выбора мишени обязательна: порядок ответа базы
        # не должен влиять на то, какую структуру мы защищаем в курсовой.
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.resolution": 2.5}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.resolution": 1.2}),
            make_structure_row(**{"structure.klifs_id": 3, "structure.resolution": 1.9}),
        ]
        fingerprints = all_fingerprints(1, 2, 3)
        straight = select_target(pd.DataFrame(rows), fingerprints)["structure.klifs_id"]
        reversed_ = select_target(pd.DataFrame(rows[::-1]), fingerprints)["structure.klifs_id"]
        assert straight == reversed_

    def test_из_двух_моделей_одной_структуры_берётся_A(self) -> None:
        # У 45% структур KLIFS несколько альтернативных моделей. Выбор не косметический:
        # у мишени 6tgu боковая цепь HIS161 между моделями расходится на 6.95 A, то есть
        # разворачивается целиком, и водородная связь с лигандом появляется или исчезает.
        rows = [
            make_structure_row(
                **{"structure.klifs_id": 1, "structure.alternate_model": "B"}
            ),
            make_structure_row(
                **{"structure.klifs_id": 2, "structure.alternate_model": "A"}
            ),
        ]
        target = select_target(pd.DataFrame(rows), all_fingerprints(1, 2))
        assert target["structure.klifs_id"] == 2

    def test_модель_не_перебивает_разрешение(self) -> None:
        # Модель различает записи одной структуры и не должна конкурировать
        # со структурами лучшего качества.
        rows = [
            make_structure_row(
                **{
                    "structure.klifs_id": 1,
                    "structure.resolution": 1.0,
                    "structure.alternate_model": "B",
                }
            ),
            make_structure_row(
                **{
                    "structure.klifs_id": 2,
                    "structure.resolution": 2.0,
                    "structure.alternate_model": "A",
                }
            ),
        ]
        target = select_target(pd.DataFrame(rows), all_fingerprints(1, 2))
        assert target["structure.klifs_id"] == 1

    def test_пропускает_структуру_без_эталонного_отпечатка(self) -> None:
        # Мишень без эталона бесполезна: сверка нашего расчёта с KLIFS и есть смысл
        # работы. Лучшая по разрешению структура уступает место следующей.
        rows = [
            make_structure_row(**{"structure.klifs_id": 1, "structure.resolution": 1.2}),
            make_structure_row(**{"structure.klifs_id": 2, "structure.resolution": 1.9}),
        ]
        target = select_target(pd.DataFrame(rows), all_fingerprints(2))
        assert target["structure.klifs_id"] == 2

    def test_падает_когда_отпечатков_нет_ни_у_кого(self) -> None:
        rows = [make_structure_row()]
        with pytest.raises(KlifsDataError, match="эталонного отпечатка"):
            select_target(pd.DataFrame(rows), {})

    def test_падает_на_пустой_таблице(self) -> None:
        with pytest.raises(KlifsDataError, match="не осталось"):
            select_target(pd.DataFrame(columns=list(make_structure_row())), {})


class TestIsValidFingerprint:
    def test_принимает_строку_нужной_длины(self) -> None:
        assert is_valid_fingerprint(VALID_BITS)

    @pytest.mark.parametrize(
        "value", ["0" * (KLIFS_IFP_LENGTH - 1), "2" * KLIFS_IFP_LENGTH, "", None]
    )
    def test_отвергает_негодное(self, value: Any) -> None:
        assert not is_valid_fingerprint(value)


class TestValidateTargetJson:
    def test_принимает_корректный_пакет(self, tmp_path: Path) -> None:
        validate_target_json(make_target_package(tmp_path / "1abc"))

    def test_допускает_отсутствие_эталонного_отпечатка(self, tmp_path: Path) -> None:
        # По формату пакета мишени klifs_ifp_bits может быть null: KLIFS отдаёт отпечаток
        # не для каждой структуры, и это не повод считать пакет битым.
        validate_target_json(make_target_package(tmp_path / "1abc", klifs_ifp_bits=None))

    def test_падает_без_обязательного_поля(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc")
        package = json.loads(path.read_text(encoding="utf-8"))
        del package["residue_to_position"]
        path.write_text(json.dumps(package), encoding="utf-8")

        with pytest.raises(KlifsDataError, match="residue_to_position"):
            validate_target_json(path)

    def test_падает_при_битой_длине_отпечатка(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc", klifs_ifp_bits="0101")
        with pytest.raises(KlifsDataError, match="klifs_ifp_bits"):
            validate_target_json(path)

    def test_падает_при_пустом_отображении_остатков(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc", residue_to_position={})
        with pytest.raises(KlifsDataError, match="residue_to_position"):
            validate_target_json(path)

    def test_падает_при_позиции_вне_диапазона(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc", residue_to_position={"LYS33.A": 99})
        with pytest.raises(KlifsDataError, match="LYS33.A"):
            validate_target_json(path)

    def test_падает_при_недопустимом_режиме_протонирования(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc", protonation="как-нибудь")
        with pytest.raises(KlifsDataError, match="protonation"):
            validate_target_json(path)

    def test_требует_инструмент_при_явном_протонировании(self, tmp_path: Path) -> None:
        # Режим explicit без названия инструмента невоспроизводим: в «Материалах
        # и методах» нечего написать о подготовке структуры.
        path = make_target_package(tmp_path / "1abc", protonation="explicit", protonation_tool=None)
        with pytest.raises(KlifsDataError, match="protonation_tool"):
            validate_target_json(path)

    def test_принимает_карман_с_пропусками(self, tmp_path: Path) -> None:
        # Позиций выравнивания всегда 85, но занятых остатков меньше: у 6tgu их 84,
        # а среди отобранных структур пропуски есть более чем у половины.
        sequence = "KEA" + "_" * (N_KLIFS_POSITIONS - 3)
        path = make_target_package(
            tmp_path / "1abc",
            pocket_sequence=sequence,
            residue_to_position={"LYS33.A": 17, "GLU51.A": 24, "ALA52.A": 25},
        )
        validate_target_json(path)

    def test_падает_при_неверной_длине_строки_кармана(self, tmp_path: Path) -> None:
        path = make_target_package(tmp_path / "1abc", pocket_sequence="KE")
        with pytest.raises(KlifsDataError, match="pocket_sequence"):
            validate_target_json(path)

    def test_падает_когда_остатки_и_пропуски_не_складываются(self, tmp_path: Path) -> None:
        # Ровно тот случай, ради которого поле и заведено: остаток потерян при сборке
        # отображения. Снаружи это неотличимо от законного пропуска — кроме этой суммы.
        path = make_target_package(
            tmp_path / "1abc",
            pocket_sequence="KEA" + "_" * (N_KLIFS_POSITIONS - 3),
            residue_to_position={"LYS33.A": 17, "GLU51.A": 24},
        )
        with pytest.raises(KlifsDataError, match="не складываются"):
            validate_target_json(path)

    def test_падает_при_чужом_типе_ингибирования(self, tmp_path: Path) -> None:
        # Пакет мишени типа II не должен молча дожить до расчётов: проект работает
        # по типу I, и это утверждение проверяется по данным, а не по памяти о фильтре.
        path = make_target_package(
            tmp_path / "1abc", dfg="out", ac_helix="in", inhibitor_type=INHIBITOR_TYPE_II
        )
        with pytest.raises(KlifsDataError, match="тип ингибирования"):
            validate_target_json(path)

    def test_падает_при_отсутствующем_файле(self, tmp_path: Path) -> None:
        base = tmp_path / "1abc"
        path = make_target_package(base)
        (base / "ligand.sdf").unlink()

        with pytest.raises(KlifsDataError, match="ligand.sdf"):
            validate_target_json(path)

    def test_падает_при_абсолютном_пути(self, tmp_path: Path) -> None:
        base = tmp_path / "1abc"
        path = make_target_package(base, protein_pdb=str((base / "protein.pdb").resolve()))

        with pytest.raises(KlifsDataError, match="абсолютный"):
            validate_target_json(path)

    def test_падает_без_проставленного_режима_протонирования(self, tmp_path: Path) -> None:
        # Пакет без режима — это пакет, у которого забыли второй шаг сборки. Раньше
        # он выглядел как осознанный 'implicit-prolif' и молча уезжал в расчёт:
        # 24.08 отпечатки разошлись вдвое. Сообщение обязано называть
        # команду, а не только диагноз.
        path = make_target_package(tmp_path / "1abc", protonation=None)

        with pytest.raises(KlifsDataError, match="annotate_protonation"):
            validate_target_json(path)

    def test_допускает_отсутствие_режима_при_сборке(self, tmp_path: Path) -> None:
        # Единственное послабление: пакет ещё собирается, режим проставит следующий шаг.
        path = make_target_package(tmp_path / "1abc", protonation=None)
        validate_target_json(path, require_protonation=False)


class TestBuildTargetPackageProtonation:
    """Сборка пакета не выдаёт режим по умолчанию за выбранный."""

    @staticmethod
    def _build(tmp_path: Path, **kwargs: Any) -> Path:
        # Сеть подменяется: обе функции ниже ходят в KLIFS, а проверяется здесь
        # содержимое собранного JSON, а не выгрузка.
        def fake_download(
            session: Any, structure_klifs_id: int, target_dir: Path, chain: str
        ) -> dict[str, str]:
            target_dir.mkdir(parents=True, exist_ok=True)
            names = {
                "protein_pdb": "protein.pdb",
                "ligand_sdf": "ligand.sdf",
                "pocket_pdb": "pocket.pdb",
                "protein_noh_pdb": "protein_noh.pdb",
            }
            for name in names.values():
                (target_dir / name).write_text("заглушка", encoding="utf-8")
            return names

        monkeypatch = pytest.MonkeyPatch()
        try:
            monkeypatch.setattr(klifs_module, "download_target_files", fake_download)
            monkeypatch.setattr(
                klifs_module,
                "build_residue_to_position",
                lambda session, structure_klifs_id, chain: {"LYS33.A": 17, "GLU51.A": 24},
            )
            return build_target_package(
                session=None,
                structure=pd.Series(make_structure_row()),
                targets_dir=tmp_path,
                fingerprint=VALID_BITS,
                **kwargs,
            )
        finally:
            monkeypatch.undo()

    def test_без_явного_режима_пакет_приходит_без_него(self, tmp_path: Path) -> None:
        path = self._build(tmp_path)
        package = json.loads(path.read_text(encoding="utf-8"))

        assert package["protonation"] is None
        assert package["protonation_tool"] is None

    def test_такой_пакет_не_проходит_строгую_проверку(self, tmp_path: Path) -> None:
        # Собрать пакет и бросить его на этом шаге можно, довести до расчёта — нет.
        path = self._build(tmp_path)

        with pytest.raises(KlifsDataError, match="annotate_protonation"):
            validate_target_json(path)

    def test_явный_режим_уважается(self, tmp_path: Path) -> None:
        path = self._build(tmp_path, protonation="implicit-prolif")
        package = json.loads(path.read_text(encoding="utf-8"))

        assert package["protonation"] == "implicit-prolif"
        validate_target_json(path)

    def test_негодный_режим_отвергается_сразу(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="protonation"):
            self._build(tmp_path, protonation="как-нибудь")


def make_taxonomy(*rows: tuple[str, str, str, str]) -> pd.DataFrame:
    """Ответ KLIFS на запрос по именам: имя, вид, группа, семейство."""
    return pd.DataFrame(
        [
            {
                "kinase.klifs_name": имя,
                "species.klifs": вид,
                "kinase.group": группа,
                "kinase.family": семейство,
            }
            for имя, вид, группа, семейство in rows
        ]
    )


# Ловушка: одно имя, два вида, разные группы.
CK2A2_TWO_SPECIES = make_taxonomy(
    ("CK2a2", "Human", "CMGC", "CK2"),
    ("CK2a2", "Mouse", "Other", "CK2"),
)


def make_summary(*rows: tuple[str, str]) -> pd.DataFrame:
    """Сводка по киназам в том виде, в каком её отдаёт `summarize_kinases`."""
    return pd.DataFrame(
        [
            {
                "kinase.klifs_name": имя,
                "species.klifs": вид,
                "n_structures": 1,
                "best_resolution": 1.0,
                "best_quality": 8.0,
            }
            for имя, вид in rows
        ]
    )


class TestFetchKinaseTaxonomy:
    def test_отдаёт_все_виды_запрошенного_имени(self) -> None:
        session = FakeSession(pd.DataFrame(), kinases=CK2A2_TWO_SPECIES)

        taxonomy = fetch_kinase_taxonomy(session, ["CK2a2"])

        assert list(taxonomy["species.klifs"]) == ["Human", "Mouse"]
        assert list(taxonomy["kinase.group"]) == ["CMGC", "Other"]

    def test_пустой_список_не_идёт_в_сеть(self) -> None:
        session = FakeSession(pd.DataFrame(), kinases=CK2A2_TWO_SPECIES)

        taxonomy = fetch_kinase_taxonomy(session, [])

        assert taxonomy.empty
        assert session.kinases.calls == []

    def test_имена_запрашиваются_пачками(self) -> None:
        имена = [f"K{n}" for n in range(KLIFS_KINASE_BATCH_SIZE * 2 + 3)]
        строки = make_taxonomy(*((имя, "Human", "TK", "SRC") for имя in имена))
        session = FakeSession(pd.DataFrame(), kinases=строки)

        taxonomy = fetch_kinase_taxonomy(session, имена)

        assert len(taxonomy) == len(имена)
        assert len(session.kinases.calls) == 3
        assert all(len(вызов) <= KLIFS_KINASE_BATCH_SIZE for вызов in session.kinases.calls)

    def test_повтор_имени_не_запрашивается_дважды(self) -> None:
        session = FakeSession(pd.DataFrame(), kinases=CK2A2_TWO_SPECIES)

        fetch_kinase_taxonomy(session, ["CK2a2", "CK2a2"])

        assert session.kinases.calls == [["CK2a2"]]

    def test_противоречивая_таксономия_останавливает(self) -> None:
        """Две группы на одну пару «имя + вид» означают, что ключ выбран неверно."""
        строки = make_taxonomy(
            ("CK2a2", "Human", "CMGC", "CK2"),
            ("CK2a2", "Human", "Other", "CK2"),
        )
        session = FakeSession(pd.DataFrame(), kinases=строки)

        with pytest.raises(KlifsDataError, match="CK2a2"):
            fetch_kinase_taxonomy(session, ["CK2a2"])

    def test_падает_при_отсутствии_колонки(self) -> None:
        строки = CK2A2_TWO_SPECIES.drop(columns="kinase.group")
        session = FakeSession(pd.DataFrame(), kinases=строки)

        with pytest.raises(KlifsDataError, match="kinase.group"):
            fetch_kinase_taxonomy(session, ["CK2a2"])


class TestAnnotateKinaseTaxonomy:
    def test_группа_берётся_по_паре_имя_и_вид(self) -> None:
        """Соединение по одному имени приписало бы человеческой CK2a2 мышиную группу."""
        сводка = make_summary(("CK2a2", "Human"))

        annotated = annotate_kinase_taxonomy(сводка, CK2A2_TWO_SPECIES)

        assert len(annotated) == 1
        assert annotated.loc[0, "kinase.group"] == "CMGC"
        assert annotated.loc[0, "kinase.family"] == "CK2"

    def test_виды_не_путаются_между_собой(self) -> None:
        сводка = make_summary(("CK2a2", "Mouse"))

        annotated = annotate_kinase_taxonomy(сводка, CK2A2_TWO_SPECIES)

        assert annotated.loc[0, "kinase.group"] == "Other"

    def test_киназа_без_таксономии_остаётся_в_сводке(self) -> None:
        """Отсутствие данных в KLIFS — не повод терять строку молча."""
        сводка = make_summary(("CK2a2", "Human"), ("NEIZVESTNAYA", "Human"))

        annotated = annotate_kinase_taxonomy(сводка, CK2A2_TWO_SPECIES)

        assert len(annotated) == 2
        assert pd.isna(annotated.loc[1, "kinase.group"])

    def test_порядок_и_число_строк_сохраняются(self) -> None:
        сводка = make_summary(("CK2a2", "Human"), ("EGFR", "Human"))
        строки = make_taxonomy(
            ("EGFR", "Human", "TK", "EGFR"),
            ("CK2a2", "Human", "CMGC", "CK2"),
        )

        annotated = annotate_kinase_taxonomy(сводка, строки)

        assert list(annotated["kinase.klifs_name"]) == ["CK2a2", "EGFR"]
        assert list(annotated["kinase.group"]) == ["CMGC", "TK"]

    def test_падает_на_чужой_таблице(self) -> None:
        with pytest.raises(KlifsDataError, match="kinase.family"):
            annotate_kinase_taxonomy(
                make_summary(("CK2a2", "Human")),
                CK2A2_TWO_SPECIES.drop(columns="kinase.family"),
            )


class TestSelectBestPerKinase:
    """Выборка для полной сверки: по одной лучшей структуре на киназу."""

    @staticmethod
    def _таблица() -> pd.DataFrame:
        return pd.DataFrame(
            [
                make_structure_row(
                    **{
                        "structure.klifs_id": 1,
                        "structure.pdb_id": "aaaa",
                        "kinase.klifs_name": "AlphaK",
                        "structure.resolution": 1.0,
                    }
                ),
                make_structure_row(
                    **{
                        "structure.klifs_id": 2,
                        "structure.pdb_id": "bbbb",
                        "kinase.klifs_name": "AlphaK",
                        "structure.resolution": 1.5,
                    }
                ),
                make_structure_row(
                    **{
                        "structure.klifs_id": 3,
                        "structure.pdb_id": "cccc",
                        "kinase.klifs_name": "BetaK",
                        "structure.resolution": 2.0,
                    }
                ),
            ]
        )

    def test_берётся_лучшая_структура_с_эталоном(self) -> None:
        """У структуры с лучшим разрешением эталона нет — берётся следующая, а не первая."""
        выборка = select_best_per_kinase(self._таблица(), ["AlphaK", "BetaK"], {2, 3})

        assert list(выборка["structure.pdb_id"]) == ["bbbb", "cccc"]

    def test_порядок_строк_совпадает_с_порядком_киназ(self) -> None:
        """Выборка должна читаться в том же порядке, в каком названы киназы."""
        выборка = select_best_per_kinase(self._таблица(), ["BetaK", "AlphaK"], {1, 3})

        assert list(выборка["kinase.klifs_name"]) == ["BetaK", "AlphaK"]

    def test_киназа_без_эталона_роняет_отбор(self) -> None:
        """Молча уменьшенная выборка сдвинула бы медиану сверки, и заметить это нечем."""
        with pytest.raises(KlifsDataError, match="BetaK"):
            select_best_per_kinase(self._таблица(), ["AlphaK", "BetaK"], {1})

    def test_пустой_список_киназ_отвергается(self) -> None:
        with pytest.raises(KlifsDataError, match="пуст"):
            select_best_per_kinase(self._таблица(), [], {1})
