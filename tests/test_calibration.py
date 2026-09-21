"""Тесты дымовой сверки с эталоном KLIFS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinase_ifp.calibration import (
    CalibrationError,
    CalibrationResult,
    StructureCalibration,
    TypeComparison,
    available_modes,
    calibrate_target,
    compare_with_reference,
    gross_failures,
    read_structure_calibration_csv,
    summarize_calibration,
    write_structure_calibration_csv,
)
from kinase_ifp.config import KLIFS_IFP_SHAPE, KLIFS_INTERACTION_TYPES
from kinase_ifp.fingerprint import compute_ifp
from kinase_ifp.molecule_io import read_mol
from kinase_ifp.protonate import prepare_ligand

_HYD = KLIFS_INTERACTION_TYPES.index("HYD")
_DON = KLIFS_INTERACTION_TYPES.index("DON")
_ACC = KLIFS_INTERACTION_TYPES.index("ACC")


@pytest.fixture(scope="module")
def отпечаток_и_эталон(target_json: Path) -> tuple[np.ndarray, np.ndarray]:
    from kinase_ifp.pocket import load_pocket

    карман = load_pocket(target_json)
    assert карман.reference_ifp is not None
    лиганд = prepare_ligand(read_mol(карман.ligand_path))
    return compute_ifp(карман, лиганд), карман.reference_ifp


def test_оба_режима_доступны(target_json: Path) -> None:
    """Сборка пакета даёт и protein_noh.pdb, значит сравнивать есть с чем."""
    assert set(available_modes(target_json)) == {"explicit", "implicit-prolif"}


def test_explicit_точнее_implicit(target_json: Path) -> None:
    """Измерение, а не суждение: режим выбирается по числу."""
    результаты = {r.protonation: r for r in calibrate_target(target_json)}

    assert результаты["explicit"].tanimoto_all > результаты["implicit-prolif"].tanimoto_all


def test_ложных_срабатываний_нет(отпечаток_и_эталон: tuple[np.ndarray, np.ndarray]) -> None:
    """Бит, которого нет у KLIFS, означал бы ошибку раскладки или адресации остатков."""
    наш, эталон = отпечаток_и_эталон
    результат = compare_with_reference(наш, эталон, "explicit")

    assert результат.extra == 0


def test_по_типам_скора_совпадение_полное(
    отпечаток_и_эталон: tuple[np.ndarray, np.ndarray],
) -> None:
    """Всё расхождение с KLIFS сидит в гидрофобных, а они в скор не входят.

    Число измерено на 6tgu и держится тестом: если оно поедет, это либо поломка
    расчёта, либо смена мишени — и то и другое обязано быть замечено.
    """
    наш, эталон = отпечаток_и_эталон
    результат = compare_with_reference(наш, эталон, "explicit")

    assert результат.tanimoto_scoring == pytest.approx(1.0)


def test_на_исправном_отпечатке_поломок_нет(
    отпечаток_и_эталон: tuple[np.ndarray, np.ndarray],
) -> None:
    наш, эталон = отпечаток_и_эталон

    assert gross_failures(compare_with_reference(наш, эталон, "explicit"), наш, эталон) == []


def test_пустой_отпечаток_ловится(отпечаток_и_эталон: tuple[np.ndarray, np.ndarray]) -> None:
    _, эталон = отпечаток_и_эталон
    пустой = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)

    поломки = gross_failures(compare_with_reference(пустой, эталон, "explicit"), пустой, эталон)

    assert any("пуст" in п for п in поломки)


def test_только_гидрофобный_отпечаток_ловится(
    отпечаток_и_эталон: tuple[np.ndarray, np.ndarray],
) -> None:
    """Ровно так выглядел бы расчёт без водородов — главный дефект черновика."""
    наш, эталон = отпечаток_и_эталон
    только_hyd = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    только_hyd[_HYD] = наш[_HYD]

    поломки = gross_failures(
        compare_with_reference(только_hyd, эталон, "explicit"), только_hyd, эталон
    )

    assert any("гидрофобные" in п for п in поломки)


def test_инверсия_донор_акцептор_ловится(
    отпечаток_и_эталон: tuple[np.ndarray, np.ndarray],
) -> None:
    """Направленность легко перепутать: в ProLIF имя про лиганд, в KLIFS — про белок."""
    _, эталон = отпечаток_и_эталон
    перепутанный = np.zeros(KLIFS_IFP_SHAPE, dtype=np.uint8)
    перепутанный[_ACC] = эталон[_DON]
    перепутанный[_DON] = эталон[_ACC]

    поломки = gross_failures(
        compare_with_reference(перепутанный, эталон, "explicit"), перепутанный, эталон
    )

    assert any("инвертированной" in п for п in поломки)


def test_мишень_без_эталона_отвергается(tmp_path: Path, target_json: Path) -> None:
    """Сверять не с чем — это ошибка, а не повод посчитать сверку успешной."""
    import json
    import shutil

    from kinase_ifp.calibration import CalibrationError

    shutil.copytree(target_json.parent, tmp_path / "6tgu")
    пакет = tmp_path / "6tgu" / "target.json"
    данные = json.loads(пакет.read_text(encoding="utf-8"))
    данные["klifs_ifp_bits"] = None
    пакет.write_text(json.dumps(данные, indent=2, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(CalibrationError, match="эталонного отпечатка"):
        calibrate_target(пакет, modes=["explicit"])


# --------------------------------------------------------------------------------------
# Полная сверка на выборке структур
# --------------------------------------------------------------------------------------


def _результат(
    *, protonation: str = "explicit", hyd: tuple[int, int, int] = (5, 12, 5), don: int = 0
) -> CalibrationResult:
    """Синтетический результат сверки: гидрофобные биты и, при желании, водородные связи.

    Сети и структур здесь не нужно: агрегация работает с числами, а числа задаются прямо.
    """
    by_type = tuple(
        TypeComparison(
            interaction_type=name,
            ours=hyd[0] if name == "HYD" else (don if name == "DON" else 0),
            reference=hyd[1] if name == "HYD" else (don if name == "DON" else 0),
            shared=hyd[2] if name == "HYD" else (don if name == "DON" else 0),
        )
        for name in KLIFS_INTERACTION_TYPES
    )
    return CalibrationResult(
        protonation=protonation,
        tanimoto_all=0.5,
        tanimoto_scoring=1.0 if don else 0.0,
        by_type=by_type,
    )


def _строка(
    pdb_id: str,
    *,
    tanimoto_all: float,
    tanimoto_scoring: float = 0.0,
    don: int = 1,
    protonation: str = "explicit",
) -> StructureCalibration:
    результат = _результат(protonation=protonation, don=don)
    return StructureCalibration(
        pdb_id=pdb_id,
        kinase=pdb_id.upper(),
        group="CMGC",
        klifs_structure_id=hash(pdb_id) % 10_000,
        result=CalibrationResult(
            protonation=protonation,
            tanimoto_all=tanimoto_all,
            tanimoto_scoring=tanimoto_scoring,
            by_type=результат.by_type,
        ),
    )


def test_типы_скора_не_определены_если_эталон_только_гидрофобный() -> None:
    """Случай 3bhy: воспроизводить по типам скора нечего, и это не ноль (docs/metrics.md, 3.9)."""
    assert _результат(don=0).scoring_defined is False
    assert _результат(don=2).scoring_defined is True


def test_сводка_считает_медиану_и_iqr() -> None:
    строки = [
        _строка("a", tanimoto_all=0.2, tanimoto_scoring=0.4),
        _строка("b", tanimoto_all=0.4, tanimoto_scoring=0.6),
        _строка("c", tanimoto_all=0.6, tanimoto_scoring=0.8),
        _строка("d", tanimoto_all=0.8, tanimoto_scoring=1.0),
    ]

    сводка = summarize_calibration(строки)

    assert сводка.tanimoto_all == pytest.approx((0.5, 0.35, 0.65))
    assert сводка.n_structures == 4
    assert сводка.n_scoring_undefined == 0


def test_структура_без_типов_скора_выпадает_только_из_своей_медианы() -> None:
    """3bhy входит в медиану по всем семи типам и не входит в медиану по типам скора."""
    строки = [
        _строка("a", tanimoto_all=0.2, tanimoto_scoring=0.5),
        _строка("b", tanimoto_all=0.6, tanimoto_scoring=0.9),
        _строка("hyd", tanimoto_all=0.1, tanimoto_scoring=0.0, don=0),
    ]

    сводка = summarize_calibration(строки)

    assert сводка.n_scoring_undefined == 1
    # Медиана по типам скора — по двум оставшимся структурам, а не по трём.
    assert сводка.tanimoto_scoring[0] == pytest.approx(0.7)
    # Медиана по всем типам — по всем трём: 0.1, 0.2, 0.6.
    assert сводка.tanimoto_all[0] == pytest.approx(0.2)


def test_разбивка_по_типам_суммируется_по_той_же_выборке() -> None:
    """Медианы и разбивка обязаны быть посчитаны на одном наборе."""
    строки = [_строка("a", tanimoto_all=0.3), _строка("b", tanimoto_all=0.5)]

    сводка = summarize_calibration(строки)
    гидрофобный = next(t for t in сводка.by_type if t.interaction_type == "HYD")

    assert сводка.n_structures == len(строки)
    assert гидрофобный.ours == 10 and гидрофобный.reference == 24
    assert гидрофобный.missed == 14


def test_смешанные_режимы_в_сводку_не_идут() -> None:
    """Отпечатки explicit и implicit-prolif несравнимы."""
    строки = [
        _строка("a", tanimoto_all=0.3),
        _строка("b", tanimoto_all=0.5, protonation="implicit-prolif"),
    ]

    with pytest.raises(CalibrationError, match="режимы протонирования"):
        summarize_calibration(строки)


def test_пустая_выборка_отвергается() -> None:
    with pytest.raises(CalibrationError, match="пуста"):
        summarize_calibration([])


class TestПоСтруктурныйФайл:
    """Запись и чтение по-структурных чисел: на них стоит сравнение прежней сверки с новой."""

    def test_записанное_читается_обратно_без_потерь(self, tmp_path: Path) -> None:
        строки = [
            _строка("6tgu", tanimoto_all=0.588, tanimoto_scoring=1.0),
            _строка("3bhy", tanimoto_all=0.067, don=0),
        ]
        путь = tmp_path / "по-структурам.csv"

        write_structure_calibration_csv(строки, путь)
        прочитанные = read_structure_calibration_csv(путь)

        assert [с.pdb_id for с in прочитанные] == ["6tgu", "3bhy"]
        assert [с.kinase for с in прочитанные] == ["6TGU", "3BHY"]
        assert [с.result.by_type for с in прочитанные] == [с.result.by_type for с in строки]
        assert прочитанные[0].result.tanimoto_all == pytest.approx(0.588)
        assert прочитанные[1].result.scoring_defined is False

    def test_медианы_по_прочитанному_те_же(self, tmp_path: Path) -> None:
        """Ради этого файл и читается: «было» обязано считаться той же функцией, что «стало»."""
        строки = [
            _строка("6tgu", tanimoto_all=0.588, tanimoto_scoring=1.0),
            _строка("3war", tanimoto_all=0.727, tanimoto_scoring=1.0),
            _строка("6ra7", tanimoto_all=0.529, tanimoto_scoring=0.8),
            _строка("3bhy", tanimoto_all=0.067, don=0),
        ]
        путь = tmp_path / "по-структурам.csv"
        write_structure_calibration_csv(строки, путь)

        было = summarize_calibration(строки)
        стало = summarize_calibration(read_structure_calibration_csv(путь))

        assert стало.tanimoto_all == pytest.approx(было.tanimoto_all)
        assert стало.tanimoto_scoring == pytest.approx(было.tanimoto_scoring)
        assert стало.n_structures == было.n_structures
        assert стало.n_scoring_undefined == было.n_scoring_undefined == 1

    def test_испорченный_флаг_роняет_чтение(self, tmp_path: Path) -> None:
        путь = tmp_path / "по-структурам.csv"
        write_structure_calibration_csv([_строка("6tgu", tanimoto_all=0.588)], путь)
        строка_данных = путь.read_text(encoding="utf-8").splitlines()[1]
        испорченная = строка_данных.replace(",1,", ",0,", 1)
        путь.write_text(
            путь.read_text(encoding="utf-8").replace(строка_данных, испорченная), encoding="utf-8"
        )

        with pytest.raises(CalibrationError, match="scoring_defined"):
            read_structure_calibration_csv(путь)

    def test_файла_нет_сообщение_называет_путь(self, tmp_path: Path) -> None:
        with pytest.raises(CalibrationError, match="нет-такого"):
            read_structure_calibration_csv(tmp_path / "нет-такого.csv")

    def test_прежние_числа_выборки_читаются(self) -> None:
        """Файл от 03.09 лежит в репозитории и служит колонкой «было» в разделе 4."""
        путь = (
            Path(__file__).resolve().parents[1] / "data" / "calibration" / "e04b_per_structure.csv"
        )

        строки = read_structure_calibration_csv(путь)

        explicit = [с for с in строки if с.protonation == "explicit"]
        assert len(explicit) == 12
        сводка = summarize_calibration(explicit)
        assert сводка.tanimoto_all[0] == pytest.approx(0.358, abs=0.001)
        assert сводка.tanimoto_scoring[0] == pytest.approx(0.750, abs=0.001)
        assert сводка.n_scoring_undefined == 1
