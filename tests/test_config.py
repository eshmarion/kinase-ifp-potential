"""Проверка инвариантов конфигурации.

Константы раскладки KLIFS связаны между собой: перепутанные местами 7 и 85 дают
фингерпринт правильной длины и полностью неверного содержания. Такую ошибку не видно
ни по исключению, ни глазами, поэтому связи между константами проверяются тестом.
"""

from __future__ import annotations

from kinase_ifp import config


def test_длина_фингерпринта_равна_произведению_размерностей() -> None:
    assert config.KLIFS_IFP_LENGTH == 595
    assert config.KLIFS_IFP_LENGTH == config.N_KLIFS_POSITIONS * config.N_KLIFS_INTERACTION_TYPES


def test_раскладка_согласована_с_длиной() -> None:
    типов, позиций = config.KLIFS_IFP_SHAPE
    assert (типов, позиций) == (config.N_KLIFS_INTERACTION_TYPES, config.N_KLIFS_POSITIONS)
    assert типов * позиций == config.KLIFS_IFP_LENGTH


def test_список_типов_совпадает_с_их_числом() -> None:
    assert len(config.KLIFS_INTERACTION_TYPES) == config.N_KLIFS_INTERACTION_TYPES
    assert len(set(config.KLIFS_INTERACTION_TYPES)) == config.N_KLIFS_INTERACTION_TYPES


def test_соответствие_klifs_prolif_покрывает_все_типы() -> None:
    типы = set(config.KLIFS_INTERACTION_TYPES)
    assert set(config.KLIFS_TO_PROLIF) == типы
    assert set(config.KLIFS_TO_PROLIF_IMPLICIT_H) == типы


def test_режим_без_водородов_отличается_только_водородными_связями() -> None:
    отличия = {
        ключ
        for ключ in config.KLIFS_TO_PROLIF
        if config.KLIFS_TO_PROLIF[ключ] != config.KLIFS_TO_PROLIF_IMPLICIT_H[ключ]
    }
    assert отличия == {"DON", "ACC"}


def test_у_каждого_типа_prolif_есть_пороги() -> None:
    используемые = set(config.KLIFS_TO_PROLIF.values()) | set(
        config.KLIFS_TO_PROLIF_IMPLICIT_H.values()
    )
    assert используемые <= set(config.PROLIF_PARAMETERS)


def test_ключевые_позиции_лежат_внутри_кармана() -> None:
    for имя, позиция in config.KEY_POSITIONS.items():
        assert 1 <= позиция <= config.N_KLIFS_POSITIONS, имя


def test_позиции_шарнира_заполнены_и_лежат_внутри_кармана() -> None:
    # Решение №17: позиции взяты из разметки KLIFS (`residue.klifs_region`), а не из
    # догадки. Пустой кортеж означал бы, что вопрос №10 снова открыт.
    assert config.HINGE_POSITIONS
    for позиция in config.HINGE_POSITIONS:
        assert 1 <= позиция <= config.N_KLIFS_POSITIONS


def test_шарнир_не_пересекается_с_другими_ключевыми_позициями() -> None:
    # Шарнир, gatekeeper, DFG и αC-Glu — разные участки кармана. Пересечение означало бы
    # съехавшую нумерацию, а такую ошибку по одному отпечатку не увидеть.
    assert not set(config.HINGE_POSITIONS) & set(config.KEY_POSITIONS.values())


def test_шарнир_идёт_сразу_за_gatekeeper() -> None:
    # Проверено на 25 киназах: GK=45, hinge=46..48. Взаимное расположение участков
    # фиксировано выравниванием KLIFS и не зависит от киназы.
    assert min(config.HINGE_POSITIONS) == config.KEY_POSITIONS["gatekeeper"] + 1


def test_корень_проекта_определён_верно() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
