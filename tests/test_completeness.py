"""Полнота кармана KLIFS в пакете мишени."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kinase_ifp.completeness import (
    FILLED_FIELD,
    MISSING_FIELD,
    PocketCompletenessError,
    check_recorded_completeness,
    check_sequence_agreement,
    completeness_fields,
    harmful_missing_positions,
    missing_positions_with_reference_bits,
    pocket_completeness,
    pocket_gap_statistics,
    reference_bits_at,
)
from kinase_ifp.config import KLIFS_IFP_LENGTH, N_KLIFS_INTERACTION_TYPES, N_KLIFS_POSITIONS
from kinase_ifp.klifs import MISSING_VALUE_MARKERS

ПУСТОЙ_ЭТАЛОН = "0" * KLIFS_IFP_LENGTH


def карта(*позиции: int) -> dict[str, int]:
    """Карта «остаток → позиция» для перечисленных позиций."""
    return {f"ALA{позиция}.A": позиция for позиция in позиции}


def полный_карман() -> dict[str, int]:
    """Карта, занимающая все 85 позиций выравнивания."""
    return карта(*range(1, N_KLIFS_POSITIONS + 1))


def строка_кармана(занятые: set[int]) -> str:
    """Строка `pocket_sequence` длиной 85, где занятые позиции помечены буквой."""
    return "".join(
        "A" if позиция in занятые else "_" for позиция in range(1, N_KLIFS_POSITIONS + 1)
    )


def эталон_с_битом(позиция: int, тип: int = 0) -> str:
    """Эталонный отпечаток, где стоит один бит — на данной позиции.

    Раскладка `(85 позиций, 7 типов)` блоками по позициям, поэтому
    индекс бита считается от начала блока позиции.
    """
    индекс = (позиция - 1) * N_KLIFS_INTERACTION_TYPES + тип
    биты = ["0"] * KLIFS_IFP_LENGTH
    биты[индекс] = "1"
    return "".join(биты)


class TestPocketCompleteness:
    def test_полный_карман_не_имеет_пропусков(self) -> None:
        полнота = pocket_completeness({"residue_to_position": полный_карман()})

        assert полнота.filled == N_KLIFS_POSITIONS
        assert полнота.missing == ()
        assert полнота.is_complete

    def test_называет_каждую_пустующую_позицию(self) -> None:
        карман = полный_карман()
        del карман["ALA50.A"]
        del карман["ALA7.A"]

        полнота = pocket_completeness({"residue_to_position": карман})

        # Перечень, а не флаг: по нему дальше сверяются биты эталона.
        assert полнота.missing == (7, 50)
        assert полнота.filled == N_KLIFS_POSITIONS - 2
        assert not полнота.is_complete

    def test_падает_на_пустой_карте_позиций(self) -> None:
        with pytest.raises(PocketCompletenessError, match="residue_to_position"):
            pocket_completeness({"residue_to_position": {}})

    def test_падает_на_позиции_вне_диапазона(self) -> None:
        with pytest.raises(PocketCompletenessError, match="диапазоне"):
            pocket_completeness({"residue_to_position": {"ALA1.A": N_KLIFS_POSITIONS + 1}})

    def test_падает_когда_два_остатка_стоят_на_одной_позиции(self) -> None:
        # Счёт занятых позиций при этом расходится с числом остатков, и честный
        # пропуск в кармане становится неотличим от ошибки сборки карты.
        with pytest.raises(PocketCompletenessError, match="два остатка"):
            pocket_completeness({"residue_to_position": {"ALA1.A": 17, "GLU2.A": 17}})


class TestReferenceBits:
    def test_вырезает_блок_из_семи_бит_на_позиции(self) -> None:
        пакет = {"klifs_ifp_bits": эталон_с_битом(50, тип=2)}

        assert reference_bits_at(пакет, 50) == "0010000"
        assert reference_bits_at(пакет, 51) == "0000000"

    def test_отсутствие_эталона_не_подменяется_нулями(self) -> None:
        # «Эталона нет» и «эталон пуст» — разные утверждения: первое означает, что
        # KLIFS отпечатка не отдал, второе — что взаимодействий нет.
        assert reference_bits_at({"klifs_ifp_bits": None}, 50) is None

    def test_падает_на_строке_неверной_длины(self) -> None:
        with pytest.raises(PocketCompletenessError, match=str(KLIFS_IFP_LENGTH)):
            reference_bits_at({"klifs_ifp_bits": "0101"}, 50)


class TestMissingWithReference:
    def test_различает_безвредную_дыру_и_задевающую_биты(self) -> None:
        карман = полный_карман()
        del карман["ALA50.A"]

        безвредный = {"residue_to_position": карман, "klifs_ifp_bits": ПУСТОЙ_ЭТАЛОН}
        вредный = {"residue_to_position": карман, "klifs_ifp_bits": эталон_с_битом(50)}

        assert missing_positions_with_reference_bits(безвредный) == {50: "0000000"}
        assert harmful_missing_positions(безвредный) == {}
        assert harmful_missing_positions(вредный) == {50: "1000000"}

    def test_бит_на_соседней_позиции_дыру_не_делает_вредной(self) -> None:
        # Ровно случай мишени 6tgu: дыра на позиции 50, а взаимодействие рядом, на 51.
        карман = полный_карман()
        del карман["ALA50.A"]
        пакет = {"residue_to_position": карман, "klifs_ifp_bits": эталон_с_битом(51)}

        assert harmful_missing_positions(пакет) == {}

    def test_без_эталона_сверять_нечего(self) -> None:
        карман = полный_карман()
        del карман["ALA50.A"]

        assert missing_positions_with_reference_bits(
            {"residue_to_position": карман, "klifs_ifp_bits": None}
        ) == {}


class TestSequenceAgreement:
    def test_согласованные_стороны_проходят(self) -> None:
        занятые = set(range(1, N_KLIFS_POSITIONS + 1)) - {50}
        пакет = {
            "residue_to_position": карта(*занятые),
            "pocket_sequence": строка_кармана(занятые),
        }

        check_sequence_agreement(пакет, MISSING_VALUE_MARKERS)

    def test_падает_когда_счёт_сходится_а_места_разные(self) -> None:
        # То, чего не ловит сумма «остатки + пропуски = 85» в validate_target_json:
        # занято по одной позиции с каждой стороны, но позиции разные.
        пакет = {
            "residue_to_position": карта(17),
            "pocket_sequence": строка_кармана({24}),
        }

        with pytest.raises(PocketCompletenessError, match="разошлись"):
            check_sequence_agreement(пакет, MISSING_VALUE_MARKERS)

    def test_падает_на_строке_неверной_длины(self) -> None:
        пакет = {"residue_to_position": карта(17), "pocket_sequence": "KE"}

        with pytest.raises(PocketCompletenessError, match="pocket_sequence"):
            check_sequence_agreement(пакет, MISSING_VALUE_MARKERS)


class TestRecordedFields:
    def test_пакет_без_полей_законен(self) -> None:
        # Это состояние пакета data/targets/6tgu/ и фикстуры tests/fixtures/6tgu/:
        # поля необязательны, потому что дописать их значило бы сменить package_sha256
        # и обесценить штампы шестнадцати прогонов.
        check_recorded_completeness({"residue_to_position": полный_карман()})

    def test_записанные_поля_сверяются_с_картой(self) -> None:
        карман = полный_карман()
        del карман["ALA50.A"]
        пакет: dict[str, Any] = {"residue_to_position": карман}
        пакет.update(completeness_fields(пакет))

        check_recorded_completeness(пакет)

        assert пакет[FILLED_FIELD] == N_KLIFS_POSITIONS - 1
        assert пакет[MISSING_FIELD] == [50]

    def test_падает_когда_число_занятых_не_сходится(self) -> None:
        пакет = {
            "residue_to_position": полный_карман(),
            FILLED_FIELD: 84,
            MISSING_FIELD: [],
        }

        with pytest.raises(PocketCompletenessError, match=FILLED_FIELD):
            check_recorded_completeness(пакет)

    def test_падает_когда_перечень_дыр_не_сходится(self) -> None:
        пакет = {
            "residue_to_position": полный_карман(),
            FILLED_FIELD: N_KLIFS_POSITIONS,
            MISSING_FIELD: [50],
        }

        with pytest.raises(PocketCompletenessError, match=MISSING_FIELD):
            check_recorded_completeness(пакет)

    def test_падает_когда_поле_одно_из_двух(self) -> None:
        # Половина ответа хуже его отсутствия: по ней не видно, что вторая потеряна.
        пакет = {"residue_to_position": полный_карман(), FILLED_FIELD: N_KLIFS_POSITIONS}

        with pytest.raises(PocketCompletenessError, match="одно поле"):
            check_recorded_completeness(пакет)


class TestGapStatistics:
    def test_считает_долю_неполных_и_частоту_позиций(self) -> None:
        строки = [
            строка_кармана(set(range(1, N_KLIFS_POSITIONS + 1))),
            строка_кармана(set(range(1, N_KLIFS_POSITIONS + 1)) - {50}),
            строка_кармана(set(range(1, N_KLIFS_POSITIONS + 1)) - {50, 7}),
        ]

        статистика = pocket_gap_statistics(строки, MISSING_VALUE_MARKERS)

        assert статистика.total == 3
        assert статистика.incomplete == 2
        assert статистика.incomplete_share == pytest.approx(2 / 3)
        assert статистика.most_common(2) == [(50, 2), (7, 1)]

    def test_строки_не_той_длины_не_считаются(self) -> None:
        # Строка кармана обязана нести 85 символов; любая другая говорит о поломанной
        # выгрузке, и считать по ней позиции нельзя.
        статистика = pocket_gap_statistics(["KE", строка_кармана({1})], MISSING_VALUE_MARKERS)

        assert статистика.total == 1

    def test_порядок_при_равной_частоте_задан_номером_позиции(self) -> None:
        # Иначе одно и то же измерение печатало бы разные списки от прогона к прогону.
        строки = [
            строка_кармана(set(range(1, N_KLIFS_POSITIONS + 1)) - {40}),
            строка_кармана(set(range(1, N_KLIFS_POSITIONS + 1)) - {9}),
        ]

        статистика = pocket_gap_statistics(строки, MISSING_VALUE_MARKERS)

        assert статистика.most_common(2) == [(9, 1), (40, 1)]

    def test_пустой_вход_не_делит_на_ноль(self) -> None:
        статистика = pocket_gap_statistics([], MISSING_VALUE_MARKERS)

        assert статистика.total == 0
        assert статистика.incomplete_share == 0.0


class TestНаМишени:
    """Числа полноты кармана на настоящем пакете, а не на синтетическом."""

    @staticmethod
    def пакет() -> dict[str, Any]:
        путь = Path("data/targets/6tgu/target.json")
        if not путь.is_file():
            pytest.skip("пакет мишени 6tgu недоступен")
        return json.loads(путь.read_text(encoding="utf-8"))

    def test_у_6tgu_занято_84_позиции_из_85(self) -> None:
        полнота = pocket_completeness(self.пакет())

        assert полнота.filled == 84
        assert полнота.missing == (50,)

    def test_пропуск_у_6tgu_безвреден(self) -> None:
        пакет = self.пакет()

        assert missing_positions_with_reference_bits(пакет) == {50: "0000000"}
        assert harmful_missing_positions(пакет) == {}

    def test_карта_позиций_и_строка_кармана_у_6tgu_сходятся(self) -> None:
        check_sequence_agreement(self.пакет(), MISSING_VALUE_MARKERS)
