"""Полнота кармана KLIFS в пакете мишени.

Отпечаток считается по остаткам из `residue_to_position` — по позициям канонического
выравнивания кармана KLIFS, которых всегда 85. Занята ли позиция у конкретной
структуры, прежде не проверялось нигде, а **пропущенный остаток даёт на своей
позиции ноль, неотличимый от честного «взаимодействия здесь нет»**: метрика не падает,
исключения нет, ни один тест не краснеет.

Модуль намеренно не импортирует ни `klifs`, ни `fingerprint`: пакет мишени собирает
`klifs.build_target_package`, и обратная зависимость замкнула бы импорты в кольцо
(`klifs` → `completeness` → `fingerprint` → `pocket` → `klifs`). Отсюда и собственный
класс ошибки: `KlifsDataError` живёт в `klifs`, и сборка пакета переводит ошибку
полноты в него сама.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any, Final

from kinase_ifp.config import KLIFS_BITS_SHAPE, KLIFS_IFP_LENGTH, N_KLIFS_POSITIONS

# Имена полей пакета мишени. Поля **необязательные**: `target.json` входит в файлы,
# по которым считается `package_sha256` (`experiments.runs.TARGET_PACKAGE_FILES`),
# и дописывание полей в уже собранный пакет `data/targets/6tgu/` сменило бы его хэш —
# шестнадцать прогонов в `runs/` стали бы «произведёнными на другом пакете», а
# `score_run` отказался бы по ним считать. Поэтому поля получают только
# пакеты, собираемые заново, а у старых полнота считается этим модулем на лету.
# Поле обязательно: без него пропуск остаётся незамеченным.
FILLED_FIELD: Final[str] = "pocket_positions_filled"
MISSING_FIELD: Final[str] = "pocket_positions_missing"


class PocketCompletenessError(RuntimeError):
    """Полнота кармана не считается или не сходится с записанной в пакете."""


@dataclass(frozen=True)
class PocketCompleteness:
    """Сколько позиций кармана занято и какие пустуют.

    `missing` хранится перечнем, а не флагом «полон / неполон»: по нему дальше
    сверяются биты эталона, и знать, что дыра есть, без знания, где она, бесполезно.
    """

    filled: int
    missing: tuple[int, ...]

    @property
    def is_complete(self) -> bool:
        """Заняты ли все 85 позиций выравнивания."""
        return not self.missing


def pocket_completeness(package: dict[str, Any]) -> PocketCompleteness:
    """Считает полноту кармана по `residue_to_position` пакета мишени.

    Принимает разобранный `target.json`, возвращает число занятых позиций и перечень
    пустующих. Поднимает `PocketCompletenessError`, если карта позиций отсутствует,
    пуста, содержит позицию вне диапазона 1..85 или ставит два остатка на одну позицию.

    Два остатка на одной позиции — не мелочь: число остатков перестаёт равняться числу
    позиций, и расхождение «84 остатка против 85 позиций» становится неразличимо
    с честным пропуском.
    """
    mapping = package.get("residue_to_position")
    if not isinstance(mapping, dict) or not mapping:
        raise PocketCompletenessError(
            "в пакете мишени нет непустого residue_to_position — считать полноту нечем"
        )

    занято: dict[int, str] = {}
    for остаток, позиция in mapping.items():
        if not isinstance(позиция, int) or isinstance(позиция, bool):
            raise PocketCompletenessError(
                f"позиция остатка {остаток} равна {позиция!r}, ожидалось целое число"
            )
        if not 1 <= позиция <= N_KLIFS_POSITIONS:
            raise PocketCompletenessError(
                f"позиция остатка {остаток} равна {позиция}, ожидалось "
                f"значение в диапазоне 1..{N_KLIFS_POSITIONS}"
            )
        if позиция in занято:
            raise PocketCompletenessError(
                f"позицию {позиция} занимают два остатка: {занято[позиция]} и {остаток}. "
                "Карта позиций собрана неверно: число остатков перестало равняться "
                "числу занятых позиций, и пропуск в кармане стал неотличим от ошибки"
            )
        занято[позиция] = str(остаток)

    пропущены = tuple(
        позиция for позиция in range(1, N_KLIFS_POSITIONS + 1) if позиция not in занято
    )
    return PocketCompleteness(filled=len(занято), missing=пропущены)


def reference_bits_at(package: dict[str, Any], position: int) -> str | None:
    """Возвращает семь бит эталонного отпечатка KLIFS на данной позиции.

    `None` означает, что эталона у структуры нет вовсе (`klifs_ifp_bits: null` —
    законное состояние пакета, KLIFS отдаёт отпечаток не для всякой структуры).
    Отсутствие эталона и эталон из одних нулей — разные вещи, и возвращать вместо
    первого строку нулей значило бы подменить «неизвестно» на «взаимодействий нет».

    Раскладка строки — `(85 позиций, 7 типов)`, блоками по позициям,
    поэтому блок позиции вырезается подстрокой, а не вычисляется по шагу.
    """
    if not 1 <= position <= N_KLIFS_POSITIONS:
        raise PocketCompletenessError(
            f"позиция {position} вне диапазона 1..{N_KLIFS_POSITIONS}"
        )

    bits = package.get("klifs_ifp_bits")
    if bits is None:
        return None
    if not isinstance(bits, str) or len(bits) != KLIFS_IFP_LENGTH:
        raise PocketCompletenessError(
            f"klifs_ifp_bits должен быть строкой из {KLIFS_IFP_LENGTH} символов либо null, "
            f"получено {type(bits).__name__} длиной "
            f"{len(bits) if isinstance(bits, str) else '?'}"
        )

    _, типов = KLIFS_BITS_SHAPE
    начало = (position - 1) * типов
    return bits[начало : начало + типов]


def missing_positions_with_reference_bits(package: dict[str, Any]) -> dict[int, str]:
    """Для каждой пустующей позиции — блок бит эталона на ней.

    Пустой словарь означает одно из двух: карман полон либо эталона у структуры нет.
    Различать эти случаи должен вызывающий — по `pocket_completeness` и по
    `klifs_ifp_bits`; здесь они оба значат «сверять нечего».

    Позиция, на которой у эталона стоит хоть один бит, — это **систематически
    невоспроизводимое взаимодействие**: остатка в структуре нет, значит наш расчёт
    не поставит там бита никогда, сколько бы верной ни была метрика.
    """
    полнота = pocket_completeness(package)
    биты: dict[int, str] = {}
    for позиция in полнота.missing:
        блок = reference_bits_at(package, позиция)
        if блок is None:
            return {}
        биты[позиция] = блок
    return биты


def harmful_missing_positions(package: dict[str, Any]) -> dict[int, str]:
    """Только те пустующие позиции, на которых эталон несёт биты.

    Ответ прост: пустой словарь — «пропуски безвредны», непустой —
    перечень мест, где расхождение с эталоном объясняется отсутствующим остатком,
    а не методом расчёта.
    """
    return {
        позиция: блок
        for позиция, блок in missing_positions_with_reference_bits(package).items()
        if "1" in блок
    }


def completeness_fields(package: dict[str, Any]) -> dict[str, Any]:
    """Пара полей пакета мишени, готовая к записи в `target.json`."""
    полнота = pocket_completeness(package)
    return {FILLED_FIELD: полнота.filled, MISSING_FIELD: list(полнота.missing)}


@dataclass(frozen=True)
class GapStatistics:
    """Насколько часто карман неполон в выгрузке и какие позиции пустуют чаще прочих."""

    total: int
    incomplete: int
    frequency: dict[int, int]

    @property
    def incomplete_share(self) -> float:
        """Доля структур с неполным карманом."""
        return self.incomplete / self.total if self.total else 0.0

    def most_common(self, count: int) -> list[tuple[int, int]]:
        """Самые частые пустующие позиции: пары «позиция, у скольких структур пуста».

        Порядок при равных частотах задан номером позиции, а не порядком словаря:
        иначе одно и то же измерение печатало бы разные списки от прогона к прогону.
        """
        упорядоченные = sorted(self.frequency.items(), key=lambda пара: (-пара[1], пара[0]))
        return упорядоченные[:count]


def pocket_gap_statistics(
    sequences: Iterable[str], gap_markers: Collection[str]
) -> GapStatistics:
    """Считает частоту пустующих позиций по строкам кармана `structure.pocket`.

    Нужна, чтобы число по одной мишени читалось верно: «у 6tgu карман неполон» звучит
    как дефект отбора, а на фоне базы видно, норма это или исключение.

    Строки не той длины отбрасываются, а не чинятся: строка кармана обязана нести
    ровно `N_KLIFS_POSITIONS` символов, и любая другая говорит не о полноте кармана,
    а о поломанной выгрузке — считать по ней позиции нельзя.
    """
    маркеры = set(gap_markers)
    всего = 0
    неполных = 0
    частота: dict[int, int] = {}
    for строка in sequences:
        if not isinstance(строка, str) or len(строка) != N_KLIFS_POSITIONS:
            continue
        всего += 1
        дыры = [
            номер for номер, символ in enumerate(строка, 1) if символ in маркеры
        ]
        if дыры:
            неполных += 1
        for позиция in дыры:
            частота[позиция] = частота.get(позиция, 0) + 1
    return GapStatistics(total=всего, incomplete=неполных, frequency=частота)


def check_sequence_agreement(
    package: dict[str, Any], gap_markers: Collection[str]
) -> None:
    """Сверяет пустующие позиции карты остатков с пропусками в `pocket_sequence`.

    `gap_markers` передаётся аргументом, а не берётся из `klifs.MISSING_VALUE_MARKERS`:
    импорт оттуда замкнул бы кольцо (см. преамбулу модуля), а собственная копия набора
    маркеров разошлась бы с оригиналом молча.

    Проверка **строже**, чем сумма «остатки + пропуски = 85» в `validate_target_json`:
    та ловит потерянный остаток, но пропускает случай, когда счёт сошёлся, а дыры
    стоят не там. Применяется при сборке пакета, где обе стороны приходят из одной
    выгрузки KLIFS; уже собранные пакеты через неё не прогоняются — у них строка
    и карта могли разойтись прежде, и объявлять их негодными
    задним числом значило бы обесценить посчитанные прогоны.
    """
    sequence = package.get("pocket_sequence")
    if not isinstance(sequence, str) or len(sequence) != N_KLIFS_POSITIONS:
        raise PocketCompletenessError(
            f"pocket_sequence должна быть строкой из {N_KLIFS_POSITIONS} символов, "
            f"получено {len(sequence) if isinstance(sequence, str) else type(sequence).__name__}"
        )

    маркеры = set(gap_markers)
    по_строке = tuple(
        номер
        for номер, символ in enumerate(sequence, 1)
        if символ in маркеры or символ.strip().lower() in маркеры
    )
    по_карте = pocket_completeness(package).missing
    if по_строке != по_карте:
        raise PocketCompletenessError(
            f"пропуски в pocket_sequence стоят на позициях {list(по_строке)}, "
            f"а в residue_to_position пустуют {list(по_карте)}. Две стороны одной "
            "выгрузки KLIFS разошлись: дальше расчёт ставил бы нули не там, где "
            "остатка нет, и расхождение с эталоном списали бы на метод"
        )


def check_recorded_completeness(package: dict[str, Any]) -> None:
    """Сверяет записанные в пакете поля полноты с посчитанными по карте позиций.

    Полей может не быть вовсе — это законно (см. комментарий к `FILLED_FIELD`), и тогда
    проверять нечего. Если поля есть, но не сходятся с `residue_to_position`, значит
    паспорт правили мимо сборки пакета, и дальше по нему считать нельзя: биты эталона
    сверялись бы с перечнем дыр, которого в структуре нет.
    """
    записано_заполнено = package.get(FILLED_FIELD)
    записано_пропущено = package.get(MISSING_FIELD)
    if записано_заполнено is None and записано_пропущено is None:
        return
    if записано_заполнено is None or записано_пропущено is None:
        raise PocketCompletenessError(
            f"в пакете есть только одно поле полноты из двух: {FILLED_FIELD}="
            f"{записано_заполнено!r}, {MISSING_FIELD}={записано_пропущено!r}. "
            "Половина ответа хуже его отсутствия — по ней не видно, что вторая половина "
            "потеряна"
        )

    полнота = pocket_completeness(package)
    if записано_заполнено != полнота.filled:
        raise PocketCompletenessError(
            f"{FILLED_FIELD}={записано_заполнено!r}, а по residue_to_position занято "
            f"{полнота.filled} позиций"
        )
    if not isinstance(записано_пропущено, list) or [
        int(позиция) for позиция in записано_пропущено
    ] != list(полнота.missing):
        raise PocketCompletenessError(
            f"{MISSING_FIELD}={записано_пропущено!r}, а по residue_to_position пустуют "
            f"позиции {list(полнота.missing)}"
        )
