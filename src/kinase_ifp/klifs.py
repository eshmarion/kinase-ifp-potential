"""Выгрузка структур KLIFS, выбор мишени и сборка пакета мишени.

Модуль отвечает за вход в проект: из базы KLIFS отбираются структуры человеческих
киназ приемлемого качества, из них процедурой выбирается одна мишень, и на неё
собирается `data/targets/<pdb_id>/target.json`. Всё остальное — расчёт отпечатка,
метрики, генерация — читает этот файл.

Сессия KLIFS передаётся аргументом, а не создаётся при импорте: иначе модуль нельзя
импортировать без сети, и тесты становятся невозможны.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd
from rdkit import Chem

from kinase_ifp.completeness import (
    FILLED_FIELD,
    MISSING_FIELD,
    PocketCompletenessError,
    check_recorded_completeness,
    check_sequence_agreement,
    completeness_fields,
)
from kinase_ifp.config import (
    DEFAULT_SPECIES,
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
    MAX_RESOLUTION,
    MIN_QUALITY_SCORE,
    N_KLIFS_POSITIONS,
    NUCLEOTIDE_LIGAND_CODES,
    TARGET_INHIBITOR_TYPES,
)
from kinase_ifp.structure_io import rebuild_pdb_files

# Колонка с типом ингибирования — наша, а не KLIFS: имя без точки, чтобы её нельзя было
# спутать с колонками базы (`structure.dfg` и прочими).
INHIBITOR_TYPE_COLUMN: Final[str] = "inhibitor_type"

# Колонка с классом лиганда — тоже наша, имя без точки по той же причине.
LIGAND_CLASS_COLUMN: Final[str] = "ligand_class"

# Колонки таблицы структур, на которые опирается отбор. Имена — из opencadd
# (`opencadd.databases.klifs`), проверены живым прогоном 22.08.2026.
#
# Колонки `interaction.fingerprint` здесь намеренно нет, хотя в таблице структур она
# присутствует: при выгрузке `all_structures()` она приходит пустой (проверено на 200
# структурах — 0 заполненных). Эталонные отпечатки KLIFS отдаёт только отдельным
# запросом `interactions.by_structure_klifs_id`, см. `fetch_fingerprints`.
STRUCTURE_COLUMNS: Final[tuple[str, ...]] = (
    "structure.klifs_id",
    "structure.pdb_id",
    "structure.chain",
    "structure.resolution",
    "structure.qualityscore",
    "species.klifs",
    "ligand.expo_id",
    "kinase.klifs_name",
    "structure.dfg",
    "structure.ac_helix",
    "structure.alternate_model",
    "structure.pocket",
)

# Какие альтернативные модели берём по умолчанию (решение 22.08.2026).
#
# В кристалле часть боковых цепей занимает несколько положений, и KLIFS заводит на
# каждое отдельную запись: альтернативы есть у 45% пар (pdb_id, цепь). Выбор не
# косметический — у мишени 6tgu боковая цепь HIS161 между моделями A и B расходится
# на 6.95 A, то есть разворачивается целиком, и водородная связь с лигандом в одной
# модели есть, а в другой невозможна.
#
# `-` означает, что альтернатив нет вовсе. Долей заселённости KLIFS не отдаёт, поэтому
# «главную» модель по данным не определить: берём первую по алфавиту, как принято.
PREFERRED_ALTERNATE_MODELS: Final[frozenset[str]] = frozenset({"-", "A"})

# Как KLIFS отмечает пустое поле. Прочерк стоит у структур без лиганда, подчёркивание —
# у незанятых позиций выравнивания кармана (проверено 22.08.2026: в таблице кармана
# ровно 85 строк, и позиции без остатка приходят как `_`). NaN появляется после чтения
# таблицы в pandas. Список общий: одни и те же обозначения встречаются в разных местах.
MISSING_VALUE_MARKERS: Final[frozenset[str]] = frozenset({"", "-", "_", "nan", "none"})

PROTONATION_MODES: Final[frozenset[str]] = frozenset({"explicit", "implicit-prolif"})

# Поля формата пакета мишени, без которых пакет мишени считается неполным.
TARGET_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "klifs_structure_id",
    "pdb_id",
    "chain",
    "kinase_name",
    "resolution",
    "quality_score",
    "protein_pdb",
    "ligand_sdf",
    "pocket_pdb",
    # Расширение формата пакета мишени от 22.08.2026: белок без водородов. Нужен второму режиму
    # сверки (`implicit-prolif`), чтобы выбор режима делало измерение, а не суждение.
    "protein_noh_pdb",
    "protonation",
    "protonation_tool",
    "residue_to_position",
    "klifs_ifp_bits",
    # Расширение формата пакета мишени от 22.08.2026: конформация кармана и выведенный из неё
    # тип ингибирования. Проект работает только по типу I, и это утверждение должно
    # проверяться по данным, а не по памяти о том, какой фильтр стоял при выгрузке.
    "dfg",
    "ac_helix",
    "inhibitor_type",
    # Какая альтернативная модель кристалла использована. Без этого поля неизвестно,
    # по каким координатам считан отпечаток: у 45% структур моделей несколько.
    "alternate_model",
    # Последовательность кармана: 85 символов, по одному на позицию выравнивания,
    # `_` — позиция, у которой в этой киназе нет остатка. Делает разницу между числом
    # остатков и числом позиций самопроверяемой (см. validate_target_json).
    "pocket_sequence",
)


class KlifsDataError(RuntimeError):
    """Данные KLIFS не соответствуют ожиданиям и обработаны быть не могут.

    Отдельный класс нужен, чтобы отличать «база отдала не то» от ошибок нашего кода:
    первое чинится правкой запроса или выбором другой структуры, второе — правкой кода.
    """


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    """Проверяет, что в таблице есть все нужные колонки.

    При расхождении поднимает `KlifsDataError` со списком фактических имён: молчаливое
    обращение к отсутствующей колонке дало бы `KeyError` без подсказки, что именно
    поменялось в API.
    """
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise KlifsDataError(
            f"{source}: в таблице нет колонок {missing}. "
            f"Фактические колонки: {sorted(df.columns)}"
        )


def _is_present(value: Any) -> bool:
    """Проверяет, что поле KLIFS заполнено, а не помечено прочерком или NaN.

    Тип аргумента — `Any`, а не `object`: значение приходит из ячейки таблицы и может
    быть строкой, числом, `None` или `pd.NA`, а `pd.isna` объявлена только для
    конкретных типов и на `object` не согласуется.
    """
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in MISSING_VALUE_MARKERS


def is_valid_fingerprint(value: Any) -> bool:
    """Проверяет, что значение — эталонный отпечаток KLIFS: 595 символов «0»/«1»."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    text = str(value).strip()
    return len(text) == KLIFS_IFP_LENGTH and set(text) <= {"0", "1"}


def classify_inhibitor_type(dfg: Any, ac_helix: Any) -> str:
    """Определяет тип ингибирования по паре конформационных признаков KLIFS.

    Принимает значения колонок `structure.dfg` и `structure.ac_helix`, возвращает одну
    из меток `INHIBITOR_TYPE_*` из `config.py`.

    Это тип по конформации кармана, а не проверенный способ связывания лиганда: прямой
    аннотации типа ингибитора KLIFS не даёт. Промежуточная конформация `out-like`
    и незаполненные поля дают `unknown` — такие структуры не должны попадать в выборку
    под видом определённого типа.
    """
    if not _is_present(dfg):
        return INHIBITOR_TYPE_UNKNOWN

    dfg_state = str(dfg).strip().lower()
    if dfg_state == "out":
        return INHIBITOR_TYPE_II
    if dfg_state != "in":
        return INHIBITOR_TYPE_UNKNOWN

    if not _is_present(ac_helix):
        return INHIBITOR_TYPE_UNKNOWN

    ac_state = str(ac_helix).strip().lower()
    if ac_state == "in":
        return INHIBITOR_TYPE_I
    if ac_state == "out":
        return INHIBITOR_TYPE_I5
    return INHIBITOR_TYPE_UNKNOWN


def annotate_inhibitor_type(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к таблице структур колонку с типом ингибирования.

    Разметка делается для всей таблицы, а не только для отобранного типа: числа
    «сколько структур какого типа» нужны и в «Материалах и методах», и в литобзоре,
    и для второй мишени как проверки переносимости.
    """
    _require_columns(df, ("structure.dfg", "structure.ac_helix"), "таблица структур KLIFS")

    annotated = df.copy()
    annotated[INHIBITOR_TYPE_COLUMN] = [
        classify_inhibitor_type(dfg, ac_helix)
        for dfg, ac_helix in zip(df["structure.dfg"], df["structure.ac_helix"], strict=True)
    ]
    return annotated


def count_by_inhibitor_type(df: pd.DataFrame) -> dict[str, int]:
    """Считает, сколько структур какого типа ингибирования в таблице."""
    annotated = df if INHIBITOR_TYPE_COLUMN in df.columns else annotate_inhibitor_type(df)
    counts = annotated[INHIBITOR_TYPE_COLUMN].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def classify_ligand(expo_id: Any) -> str:
    """Определяет класс лиганда в ортостерическом кармане по коду PDB.

    Принимает значение колонки `ligand.expo_id`, возвращает одну из меток
    `LIGAND_CLASS_*` из `config.py`: `nucleotide` — субстрат киназы или его имитация,
    `inhibitor-like` — всё прочее при заполненном поле, `none` — поле пустое.

    Зачем разделение: DiffSBDD генерирует лиганд в пустом кармане,
    а комплекс с собственным субстратом описывает другую ситуацию. Отпечаток такой
    структуры посчитан верно, но как эталон он относится не к тому явлению — у структур
    с нуклеотидом медиана гидрофобных бит эталона 7 против 13 у ингибиторных.

    **Аллостерическое поле на класс не влияет.** Нуклеотид
    в `ligand_allosteric.expo_id` встречается у 8 структур выгрузки, из них одна несёт
    ингибитор в ортостерическом кармане. Класс определяется ортостерическим полем,
    потому что эталонный отпечаток считается по ортостерическому карману: что стоит
    в аллостерическом сайте, в этот отпечаток не входит. Присутствие субстрата рядом
    остаётся видимым в исходной колонке и классом не скрывается.

    Стауроспорин здесь законно получает `inhibitor-like`: это ингибитор, пусть
    и неселективный.
    """
    if not _is_present(expo_id):
        return LIGAND_CLASS_NONE
    if str(expo_id).strip().upper() in NUCLEOTIDE_LIGAND_CODES:
        return LIGAND_CLASS_NUCLEOTIDE
    return LIGAND_CLASS_INHIBITOR_LIKE


def annotate_ligand_class(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к таблице структур колонку с классом лиганда.

    Разметка делается для всей таблицы, а не только для рабочей выборки: структуры
    с нуклеотидом нужны как контрольная группа — с ними показывается, что вывод
    не зависит от их включения. Удалять их из выгрузки поэтому нельзя.
    """
    _require_columns(df, ("ligand.expo_id",), "таблица структур KLIFS")

    annotated = df.copy()
    annotated[LIGAND_CLASS_COLUMN] = [classify_ligand(expo_id) for expo_id in df["ligand.expo_id"]]
    return annotated


def count_by_ligand_class(df: pd.DataFrame) -> dict[str, int]:
    """Считает, сколько структур какого класса лиганда в таблице."""
    annotated = df if LIGAND_CLASS_COLUMN in df.columns else annotate_ligand_class(df)
    counts = annotated[LIGAND_CLASS_COLUMN].value_counts()
    return {str(name): int(count) for name, count in counts.items()}


def filter_structures(
    df: pd.DataFrame,
    inhibitor_types: tuple[str, ...] = TARGET_INHIBITOR_TYPES,
    species: str | None = DEFAULT_SPECIES,
    ligand_classes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Отбирает структуры, пригодные для расчёта взаимодействий.

    Принимает таблицу структур KLIFS, возвращает её подмножество с добавленной колонкой
    типа ингибирования: нужный вид, разрешение не хуже `MAX_RESOLUTION`, оценка качества
    не ниже `MIN_QUALITY_SCORE`, есть лиганд и тип ингибирования входит в
    `inhibitor_types`. Наличие эталонного отпечатка здесь не проверяется — его нет
    в этой таблице, он забирается `fetch_fingerprints`.

    Границы включительны: структура ровно на пороге проходит. Пороги берутся из
    `config.py` и здесь не дублируются. Список типов вынесен в аргумент, чтобы выборку
    другого типа можно было получить, не трогая код.

    `species=None` снимает ограничение по виду: киназы других организмов пригодны
    для той же задачи, и отбрасывать их до того, как посмотрели на список, незачем.

    `ligand_classes` отсекает структуры по классу лиганда и **по умолчанию
    не применяется**. Умолчание здесь не лень: на дефолте сидят три вызова —
    `select_target.py` (выбор мишени 6tgu), `calibrate_ifp.py` (выборка калибровки
    сверки, где два комплекса из двенадцати нуклеотидные) и `fetch_klifs.py`. Отсечение
    в умолчании поменяло бы уже посчитанные и записанные в текст числа молча — ровно
    тот класс тихой ошибки, ради которого классификация и заводилась. Рабочая выборка
    получается явным `ligand_classes=TARGET_LIGAND_CLASSES` в месте сборки, и это видно
    в коде вызова, а не спрятано в значении по умолчанию.
    """
    _require_columns(df, STRUCTURE_COLUMNS, "таблица структур KLIFS")
    annotated = annotate_ligand_class(annotate_inhibitor_type(df))

    keep = (
        (annotated["structure.resolution"] <= MAX_RESOLUTION)
        & (annotated["structure.qualityscore"] >= MIN_QUALITY_SCORE)
        & annotated["ligand.expo_id"].map(_is_present)
        & annotated[INHIBITOR_TYPE_COLUMN].isin(inhibitor_types)
    )
    if species is not None:
        keep &= annotated["species.klifs"] == species
    if ligand_classes is not None:
        keep &= annotated[LIGAND_CLASS_COLUMN].isin(ligand_classes)
    return annotated.loc[keep].copy()


def summarize_kinases(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по киназам: сколько пригодных структур у каждой и какая из них лучшая.

    Принимает уже отфильтрованную таблицу структур, возвращает по одной строке на пару
    «киназа + вид», отсортированную так же, как выбирается мишень: сначала лучшее
    разрешение. Нужна, чтобы выбирать мишень по списку кандидатов,
    а не по одной строке, и чтобы дать в текст курсовой числа о доступных данных.
    """
    _require_columns(
        df,
        ("kinase.klifs_name", "species.klifs", "structure.resolution", "structure.qualityscore"),
        "таблица отобранных структур",
    )

    grouped = df.groupby(["kinase.klifs_name", "species.klifs"], dropna=False)
    summary = grouped.agg(
        n_structures=("structure.klifs_id", "count"),
        best_resolution=("structure.resolution", "min"),
        best_quality=("structure.qualityscore", "max"),
    ).reset_index()

    return summary.sort_values(
        by=["best_resolution", "best_quality", "n_structures"],
        ascending=[True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def rank_structures(df: pd.DataFrame) -> pd.DataFrame:
    """Упорядочивает структуры по пригодности: лучшее разрешение первым.

    Порядок: разрешение по возрастанию, при равенстве выше оценка качества, затем
    предпочтительная альтернативная модель, затем меньший KLIFS ID. Без сортировки
    «первая строка» зависела бы от порядка ответа базы, и выбор мишени перестал бы быть
    воспроизводимым: в «Материалах и методах» такой отбор описать нечем.

    Модель стоит после разрешения и качества намеренно: она различает записи **одной
    и той же** структуры, а не конкурирует со структурами лучшего качества.
    """
    _require_columns(
        df,
        (
            "structure.resolution",
            "structure.qualityscore",
            "structure.klifs_id",
            "structure.alternate_model",
        ),
        "таблица структур KLIFS",
    )

    ranked = df.copy()
    ranked["_model_priority"] = [
        0 if str(model).strip() in PREFERRED_ALTERNATE_MODELS else 1
        for model in ranked["structure.alternate_model"]
    ]
    ranked = ranked.sort_values(
        by=[
            "structure.resolution",
            "structure.qualityscore",
            "_model_priority",
            "structure.klifs_id",
        ],
        ascending=[True, False, True, True],
        kind="stable",
    )
    return ranked.drop(columns="_model_priority")


def fetch_fingerprints(session: Any, structure_klifs_ids: list[int]) -> dict[int, str]:
    """Забирает эталонные 595-битные отпечатки KLIFS для перечисленных структур.

    Возвращает словарь «KLIFS ID структуры → битовая строка», куда попадают только
    валидные отпечатки: у части структур база отпечаток не отдаёт вовсе.

    Отдельный запрос нужен потому, что в таблице `all_structures()` колонка
    `interaction.fingerprint` приходит пустой — проверено живым прогоном 22.08.2026.
    """
    if not structure_klifs_ids:
        return {}

    df = session.interactions.by_structure_klifs_id(structure_klifs_ids)
    if not isinstance(df, pd.DataFrame):
        raise KlifsDataError(f"KLIFS вернул {type(df)!r} вместо таблицы взаимодействий")

    _require_columns(
        df, ("structure.klifs_id", "interaction.fingerprint"), "таблица взаимодействий KLIFS"
    )

    fingerprints: dict[int, str] = {}
    for klifs_id, bits in zip(df["structure.klifs_id"], df["interaction.fingerprint"], strict=True):
        if is_valid_fingerprint(bits):
            fingerprints[int(klifs_id)] = str(bits).strip()
    return fingerprints


# Колонки файла эталонных отпечатков `data/klifs/klifs_ifp.csv`. Имена
# без точек — это наш файл, а не таблица KLIFS, и путать их нельзя.
FINGERPRINT_TABLE_COLUMNS: Final[tuple[str, ...]] = ("structure_id", "pdb_id", "bits")


def fetch_fingerprint_table(
    session: Any,
    structures: pd.DataFrame,
    known: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Собирает таблицу эталонных отпечатков KLIFS для перечисленных структур.

    `structures` — таблица структур KLIFS, `known` — уже выгруженные отпечатки
    (их не запрашивают заново). Возвращает таблицу с колонками
    `FINGERPRINT_TABLE_COLUMNS`, отсортированную по `structure_id`.

    Структура, для которой KLIFS отпечатка не отдал, в таблицу не попадает: это
    отсутствие данных, а не ошибка расчёта. Молча такая структура не исчезает —
    разницу между числом запрошенных и числом полученных строк печатает CLI.

    Запрос идёт пачками по `KLIFS_FINGERPRINT_BATCH_SIZE`: идентификаторы клиент KLIFS
    подставляет в URL, и один список на всю выборку упёрся бы в ограничение на длину
    адреса.
    """
    _require_columns(
        structures, ("structure.klifs_id", "structure.pdb_id"), "таблица структур KLIFS"
    )

    pdb_by_id = {
        int(klifs_id): str(pdb_id)
        for klifs_id, pdb_id in zip(
            structures["structure.klifs_id"], structures["structure.pdb_id"], strict=True
        )
    }
    known = dict(known or {})

    # Уже выгруженные отпечатки берутся из `known`, остальные запрашиваются по сети.
    fingerprints = {
        klifs_id: bits for klifs_id, bits in known.items() if klifs_id in pdb_by_id
    }
    pending = [klifs_id for klifs_id in pdb_by_id if klifs_id not in known]
    for start in range(0, len(pending), KLIFS_FINGERPRINT_BATCH_SIZE):
        batch = pending[start : start + KLIFS_FINGERPRINT_BATCH_SIZE]
        # В ответе берутся только запрошенные структуры: таблица описывает то, что
        # ей передали, а лишняя строка от базы означала бы отпечаток без `pdb_id`.
        fingerprints.update(
            {
                klifs_id: bits
                for klifs_id, bits in fetch_fingerprints(session, batch).items()
                if klifs_id in pdb_by_id
            }
        )

    table = pd.DataFrame(
        [
            {"structure_id": klifs_id, "pdb_id": pdb_by_id[klifs_id], "bits": bits}
            for klifs_id, bits in fingerprints.items()
        ],
        columns=list(FINGERPRINT_TABLE_COLUMNS),
    )
    return table.sort_values("structure_id", kind="stable").reset_index(drop=True)


def read_fingerprint_table(path: Path) -> pd.DataFrame:
    """Читает файл эталонных отпечатков; отсутствующего файла достаточно для пустой таблицы.

    Битовые строки проверяются при чтении. Повреждённый файл обязан остановить работу
    здесь: иначе испорченный эталон уедет в сверку с нашим расчётом и объяснит
    расхождение не тем.
    """
    if not path.exists():
        return pd.DataFrame(columns=list(FINGERPRINT_TABLE_COLUMNS))

    table = pd.read_csv(path, dtype={"structure_id": int, "pdb_id": str, "bits": str})
    _require_columns(table, FINGERPRINT_TABLE_COLUMNS, f"файл эталонных отпечатков {path}")

    broken = [
        int(structure_id)
        for structure_id, bits in zip(table["structure_id"], table["bits"], strict=True)
        if not is_valid_fingerprint(bits)
    ]
    if broken:
        raise KlifsDataError(
            f"{path}: у структур {broken[:10]} отпечаток не является строкой "
            f"из {KLIFS_IFP_LENGTH} символов «0»/«1» (всего испорченных строк: {len(broken)})"
        )
    return table[list(FINGERPRINT_TABLE_COLUMNS)]


def merge_fingerprint_tables(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Дописывает свежие отпечатки к уже выгруженным, не теряя прежних строк.

    Файл эталонов дописывается, а не переписывается: выгрузка идёт частями (сначала
    лучшие структуры, остальные позже той же командой), и второй запуск с другим
    `--limit` не должен стирать результат первого. При совпадении `structure_id`
    остаётся свежая строка.
    """
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset="structure_id", keep="last")
    merged["structure_id"] = merged["structure_id"].astype(int)
    return merged.sort_values("structure_id", kind="stable").reset_index(drop=True)


def select_target(df: pd.DataFrame, fingerprints: dict[int, str]) -> pd.Series:
    """Выбирает мишень: лучшая по качеству структура, у которой есть эталонный отпечаток.

    Мишень не называется от себя, а выбирается
    правилом. Структуры без эталонного отпечатка пропускаются — сверка нашего расчёта
    с KLIFS составляет смысл работы, и мишень без эталона для неё бесполезна.
    """
    if df.empty:
        raise KlifsDataError(
            "После фильтрации не осталось ни одной структуры. Ослабьте фильтры "
            "в config.py или увеличьте выборку (--limit)."
        )

    for _, row in rank_structures(df).iterrows():
        if int(row["structure.klifs_id"]) in fingerprints:
            return row

    raise KlifsDataError(
        f"Ни у одной из {len(df)} отобранных структур нет эталонного отпечатка KLIFS. "
        f"Увеличьте число кандидатов, для которых запрашиваются отпечатки."
    )


def select_best_per_kinase(
    df: pd.DataFrame,
    kinase_names: Sequence[str],
    fingerprint_ids: Collection[int],
) -> pd.DataFrame:
    """По одной лучшей структуре на киназу — выборка для полной сверки.

    Принимает отфильтрованную таблицу структур, имена киназ (`CALIBRATION_KINASES`)
    и идентификаторы структур, для которых есть эталонный отпечаток KLIFS. Возвращает
    таблицу из одной строки на киназу в порядке `kinase_names`.

    Правило отбора то же, что у `select_target`: лучшая по `rank_structures` структура,
    у которой есть эталон. Поэтому выборка описывается процедурой, а не перечнем PDB —
    в «Материалах и методах» перечислять коды структур не приходится.

    Киназа, у которой такой структуры нет, роняет вызов с её именем: выборка сверки
    обязана быть той, которую назвали, а молча уменьшенная выборка изменила бы медиану
    и осталась бы незамеченной.
    """
    _require_columns(df, ("kinase.klifs_name", "structure.klifs_id"), "таблица структур KLIFS")
    if not kinase_names:
        raise KlifsDataError("Список киназ для сверки пуст: выборку строить не из чего")

    rows: list[pd.Series] = []
    missing: list[str] = []
    for name in kinase_names:
        candidates = rank_structures(df.loc[df["kinase.klifs_name"] == name])
        with_reference = candidates.loc[
            [int(i) in fingerprint_ids for i in candidates["structure.klifs_id"]]
        ]
        if with_reference.empty:
            missing.append(name)
            continue
        rows.append(with_reference.iloc[0])

    if missing:
        raise KlifsDataError(
            f"Для киназ {missing} нет ни одной отобранной структуры с эталонным отпечатком "
            f"KLIFS. Пополните data/klifs/klifs_ifp.csv (scripts/fetch_klifs_ifp.py --limit 0) "
            f"или измените CALIBRATION_KINASES"
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def fetch_structures(session: Any, limit: int | None = None) -> pd.DataFrame:
    """Забирает таблицу структур KLIFS и размечает её по типу ингибирования.

    `session` — объект, созданный `opencadd.databases.klifs.setup_remote()`.
    `limit` ограничивает число строк и нужен только для быстрой проверки конвейера;
    при `None` берутся все структуры.

    Фильтры качества здесь не применяются: отбор — задача `filter_structures`.
    Разделение нужно, чтобы посчитать, сколько структур какого типа отсеялось, —
    без исходной таблицы такие числа взять негде.
    """
    df = session.structures.all_structures()
    if not isinstance(df, pd.DataFrame):
        raise KlifsDataError(f"KLIFS вернул {type(df)!r} вместо таблицы структур")

    if limit is not None:
        df = df.head(limit)

    _require_columns(df, STRUCTURE_COLUMNS, "таблица структур KLIFS")
    return annotate_ligand_class(annotate_inhibitor_type(df))


# Колонки таксономии киназ по классификации Manning: группа и семейство внутри группы.
# В таблице структур эти поля приходят пустыми, отдаёт их только запрос по имени киназы
#.
#
# Групп **девять, а не восемь**. `kinases.all_kinase_groups()` перечисляет восемь
# (AGC, CAMK, CK1, CMGC, Other, STE, TK, TKL), но `by_kinase_name` отдаёт ещё
# `Atypical` — у 13 из наших 227 киназ именно она. Послойный отбор, перебирающий
# список из `all_kinase_groups()`, потерял бы этот слой молча: киназы в нём есть,
# а группы в списке нет. Измерено 31.08.2026 на выгрузке `data/klifs/kinases.csv`.
KINASE_TAXONOMY_COLUMNS: Final[tuple[str, ...]] = ("kinase.group", "kinase.family")

# Ключ киназы — **пара** «имя + вид», а не имя. Одно имя даёт разную группу у разных
# организмов: CK2a2 у человека относится к CMGC, у мыши — к Other. Соединение по одному
# имени молча приписало бы части киназ мышиную группу, и выглядело бы это как обычные
# данные, а не как ошибка.
KINASE_KEY_COLUMNS: Final[tuple[str, ...]] = ("kinase.klifs_name", "species.klifs")


def fetch_kinase_taxonomy(session: Any, names: list[str]) -> pd.DataFrame:
    """Забирает группу и семейство киназ по их именам.

    `session` — объект `opencadd.databases.klifs.setup_remote()`, `names` — имена
    киназ (`kinase.klifs_name`). Возвращает таблицу с колонками `KINASE_KEY_COLUMNS`
    и `KINASE_TAXONOMY_COLUMNS`, по строке на пару «имя + вид».

    Запрос идёт по именам, потому что `structures.all_structures()` и `all_kinases()`
    группу и семейство не отдают вовсе: в таблице структур колонки есть в схеме,
    но пусты у всех строк.

    В ответе оказываются **все** виды для запрошенного имени, а не только нужный:
    отсюда пара в ключе. Строку с противоречивой таксономией функция не выбирает
    и не усредняет, а останавливается — расхождение означает, что ключ выбран неверно.
    """
    if not names:
        return pd.DataFrame(columns=[*KINASE_KEY_COLUMNS, *KINASE_TAXONOMY_COLUMNS])

    unique = list(dict.fromkeys(str(name) for name in names))
    parts: list[pd.DataFrame] = []
    for start in range(0, len(unique), KLIFS_KINASE_BATCH_SIZE):
        batch = unique[start : start + KLIFS_KINASE_BATCH_SIZE]
        df = session.kinases.by_kinase_name(batch)
        if not isinstance(df, pd.DataFrame):
            raise KlifsDataError(f"KLIFS вернул {type(df)!r} вместо таблицы киназ")
        _require_columns(
            df, (*KINASE_KEY_COLUMNS, *KINASE_TAXONOMY_COLUMNS), "таблица киназ KLIFS"
        )
        parts.append(df.loc[:, [*KINASE_KEY_COLUMNS, *KINASE_TAXONOMY_COLUMNS]])

    taxonomy = pd.concat(parts, ignore_index=True).drop_duplicates()
    duplicates = taxonomy.duplicated(subset=list(KINASE_KEY_COLUMNS), keep=False)
    if duplicates.any():
        конфликт = taxonomy.loc[duplicates, list(KINASE_KEY_COLUMNS)].drop_duplicates()
        пары = ", ".join(f"{имя} ({вид})" for имя, вид in конфликт.itertuples(index=False))
        raise KlifsDataError(
            f"KLIFS отдал разную таксономию для одной пары «имя + вид»: {пары}"
        )
    return taxonomy.sort_values(list(KINASE_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def annotate_kinase_taxonomy(kinases: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Приписывает сводке по киназам группу и семейство, соединяя по паре «имя + вид».

    `kinases` — результат `summarize_kinases`, `taxonomy` — результат
    `fetch_kinase_taxonomy`. Возвращает ту же сводку с двумя добавленными колонками;
    порядок строк сохраняется.

    Киназа, которой в `taxonomy` нет, остаётся с пустыми полями: это отсутствие данных
    в KLIFS, а не ошибка расчёта. Молча такая киназа не теряется — сколько их, печатает
    вызывающий скрипт.
    """
    _require_columns(kinases, KINASE_KEY_COLUMNS, "сводка по киназам")
    _require_columns(
        taxonomy, (*KINASE_KEY_COLUMNS, *KINASE_TAXONOMY_COLUMNS), "таблица таксономии киназ"
    )

    # `validate` — не украшение: при двух строках таксономии на один ключ соединение
    # размножило бы строки сводки, и число киназ в отчёте тихо выросло бы.
    return kinases.merge(
        taxonomy.loc[:, [*KINASE_KEY_COLUMNS, *KINASE_TAXONOMY_COLUMNS]],
        on=list(KINASE_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )


def _relative_to(path: Path, base: Path) -> str:
    """Возвращает путь относительно каталога мишени.

    Пути в пакете относительные: `target.json` должен пережить копирование
    в Modal Volume, а абсолютный путь с локальной машины там не существует.
    """
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError as error:
        raise KlifsDataError(
            f"Файл {path} лежит вне каталога мишени {base}, относительный путь построить "
            f"нельзя"
        ) from error


def download_target_files(
    session: Any, structure_klifs_id: int, target_dir: Path, chain: str
) -> dict[str, str]:
    """Скачивает белок, карман и лиганд мишени в каталог мишени.

    Возвращает словарь с относительными путями для полей `protein_pdb`, `pocket_pdb`,
    `protein_noh_pdb` и `ligand_sdf` формата пакета мишени.

    Белок и карман KLIFS в удалённом режиме отдаёт **только в mol2** (в PDB доступен
    лишь `complex`, проверено 22.08.2026), а формат пакета мишени требует PDB. Поэтому исходный
    mol2 сохраняется рядом как есть, а PDB строится из него в `structure_io`: прямая
    конвертация средствами MDAnalysis теряет имя и номер остатка, и отпечаток выходит
    пустым (подробности и измерение — в docstring `structure_io`). Цепь берётся
    аргументом: в формате mol2 её нет.

    Лиганд KLIFS отдаёт в mol2, а пакету нужен SDF, поэтому лиганд проходит через
    RDKit и записывается `SDWriter`.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Лиганд качается сырым mol2 наравне с белком и карманом: KLIFS отдаёт его
    # протонированным конвейером MOE, и именно по этим водородам посчитан эталонный
    # отпечаток. В `ligand.sdf` они не доживают — `to_rdkit` читает mol2 с умолчанием
    # `removeHs=True`, — а достроенные нами водороды вращающихся групп стоят иначе
    # и меняют биты водородной связи (`docs/calibration.md`, 7.3). Файл в хэш пакета
    # не входит (`TARGET_PACKAGE_FILES`), поэтому уже посчитанные прогоны он не трогает.
    for entity in ("protein", "pocket", "ligand"):
        text = session.coordinates.to_text(structure_klifs_id, entity=entity, extension="mol2")
        if not text:
            raise KlifsDataError(
                f"KLIFS не отдал {entity} для структуры {structure_klifs_id}"
            )
        (target_dir / f"{entity}.mol2").write_text(text, encoding="utf-8")

    paths: dict[str, str] = rebuild_pdb_files(target_dir, chain)

    # compute2d=False обязателен: значение по умолчанию True пересчитывает координаты
    # в плоские, а нам нужна кристаллическая поза лиганда в кармане.
    mol = session.coordinates.to_rdkit(
        structure_klifs_id, entity="ligand", extension="mol2", compute2d=False
    )
    if mol is None:
        raise KlifsDataError(
            f"RDKit не разобрал лиганд структуры {structure_klifs_id}: to_rdkit вернул None"
        )

    # Рядом с лигандом в файле может лежать ион или молекула растворителя — тогда RDKit
    # читает несколько несвязанных кусков как одну молекулу. Дальше по конвейеру это
    # проявилось бы не ошибкой, а неверными числами: конформеры разъехались бы,
    # `connected` обнулилась, а посторонний ион добавил бы в скор контакты, которых
    # у лиганда нет. Проверяем здесь, пока лечится выбором другой мишени.
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        raise KlifsDataError(
            f"Лиганд структуры {structure_klifs_id} состоит из {len(fragments)} несвязанных "
            f"фрагментов: для набора поз и расчёта отпечатка нужна одна молекула"
        )

    ligand_path = target_dir / "ligand.sdf"
    # SDWriter получает открытый поток, а не строку пути: путь он передаёт в C++
    # через ANSI и на Windows падает с «Bad output file» на кириллице. Каталог
    # `data/targets/` латинский, но он лежит внутри домашнего каталога пользователя,
    # а тот может быть любым. То же правило, что в `experiments.run_io`.
    with ligand_path.open("w", encoding="utf-8", newline="") as stream:
        writer = Chem.SDWriter(stream)
        writer.write(mol)
        writer.close()
    paths["ligand_sdf"] = _relative_to(ligand_path, target_dir)

    return paths


def build_residue_to_position(session: Any, structure_klifs_id: int, chain: str) -> dict[str, int]:
    """Строит отображение остатка кармана в каноническую позицию KLIFS 1..85.

    Ключ — остаток в нотации ProLIF (`RESNAME<номер>.<цепь>`), значение — позиция KLIFS.
    Имя остатка берётся из координат кармана: таблица `pockets` даёт номер и позицию,
    но не трёхбуквенное имя, а ProLIF адресует остатки именно по имени.

    Цепь подставляется одна на весь карман — из таблицы структур: KLIFS выравнивает
    карман в пределах одной цепи, а формат mol2 сведений о цепи не хранит.
    """
    pocket = session.pockets.by_structure_klifs_id(structure_klifs_id)
    _require_columns(pocket, ("residue.id", "residue.klifs_id"), "таблица остатков кармана")

    atoms = session.coordinates.to_dataframe(structure_klifs_id, entity="pocket", extension="mol2")
    _require_columns(atoms, ("residue.name", "residue.id"), "координаты кармана")

    # Одно имя на остаток: в таблице атомов каждый остаток встречается столько раз,
    # сколько в нём атомов.
    name_by_residue_id = (
        atoms.drop_duplicates(subset="residue.id").set_index("residue.id")["residue.name"].to_dict()
    )

    mapping: dict[str, int] = {}
    for _, row in pocket.iterrows():
        residue_id = row["residue.id"]
        position = row["residue.klifs_id"]

        # KLIFS помечает незанятые позиции выравнивания прочерком: остатка в структуре
        # нет, и включать его в отображение не во что.
        if not _is_present(residue_id) or not _is_present(position):
            continue

        position = int(position)
        if not 1 <= position <= N_KLIFS_POSITIONS:
            raise KlifsDataError(
                f"Позиция KLIFS {position} вне диапазона 1..{N_KLIFS_POSITIONS} "
                f"(остаток {residue_id})"
            )

        residue_name = name_by_residue_id.get(residue_id)
        if residue_name is None:
            raise KlifsDataError(
                f"Остаток {residue_id} есть в таблице кармана, но отсутствует в координатах: "
                f"имя остатка для нотации ProLIF взять неоткуда"
            )

        mapping[f"{residue_name}{residue_id}.{chain}"] = position

    if not mapping:
        raise KlifsDataError(
            f"Для структуры {structure_klifs_id} не построено ни одного соответствия "
            f"остаток→позиция KLIFS"
        )
    return mapping


# Порядок ключей `target.json`. Нужен не для красоты: `package_sha256`
# считается по байтам файла, а `json.dumps` сохраняет порядок вставки. Пакет, собранный
# до появления `protein_noh_pdb`, чинился на месте `annotate_protonation`, и новый ключ
# уезжал в конец — тот же по содержанию пакет давал другой хэш, и расчёт отказывался
# считать по «чужому» пакету, который на самом деле свой.
TARGET_JSON_KEYS: Final[tuple[str, ...]] = (
    "klifs_structure_id",
    "pdb_id",
    "chain",
    "kinase_name",
    "resolution",
    "quality_score",
    "protein_pdb",
    "ligand_sdf",
    "pocket_pdb",
    "protein_noh_pdb",
    "protonation",
    "protonation_tool",
    "residue_to_position",
    "klifs_ifp_bits",
    "dfg",
    "ac_helix",
    "inhibitor_type",
    "alternate_model",
    "pocket_sequence",
    # Полнота кармана. Стоит в конце намеренно: ключи дописаны к готовому
    # порядку, и пакет, собранный до появления полей, отличается от свежего только
    # хвостом — сравнивать такие файлы глазами всё ещё можно.
    FILLED_FIELD,
    MISSING_FIELD,
)


def canonical_target_json(package: dict[str, Any]) -> dict[str, Any]:
    """Раскладывает поля пакета мишени в порядке `TARGET_JSON_KEYS`.

    Незнакомые ключи не теряются, а дописываются следом в исходном порядке: молча
    выбросить поле хуже, чем оставить хэш зависимым от него.
    """
    известные = {имя: package[имя] for имя in TARGET_JSON_KEYS if имя in package}
    прочие = {имя: значение for имя, значение in package.items() if имя not in известные}
    return {**известные, **прочие}


def build_target_package(
    session: Any,
    structure: pd.Series,
    targets_dir: Path,
    fingerprint: str | None,
    protonation: str | None = None,
    protonation_tool: str | None = None,
) -> Path:
    """Собирает `data/targets/<pdb_id>/target.json` по формату пакета мишени.

    `fingerprint` — эталонная битовая строка из `fetch_fingerprints`; `None` допустим
    и означает, что KLIFS отпечатка для этой структуры не отдал.

    `protonation=None` — «режим ещё не проставлен»: сборка пакета сама водороды
    не считает. Режим проставляет `protonation.annotate_protonation`
    по факту подсчёта водородов, а окончательный выбор между режимами делает измерение
    измерением, не этим кодом. Такой пакет не проходит
    `validate_target_json` и до расчёта отпечатка не доживает.

    Режима по умолчанию здесь нет намеренно. Раньше стояло `implicit-prolif`, и пакет,
    собранный без второго шага, выглядел как осознанно выбранный режим: 24.08 два пакета
    разошлись режимами, отпечатки разошлись вдвое и попали в записи как сопоставимые
.
    """
    if protonation is not None and protonation not in PROTONATION_MODES:
        raise ValueError(f"protonation={protonation!r}, допустимы {sorted(PROTONATION_MODES)}")

    pdb_id = str(structure["structure.pdb_id"])
    structure_klifs_id = int(structure["structure.klifs_id"])

    # Цепь входит в ключи residue_to_position, поэтому пустое значение здесь испортило бы
    # адресацию остатков в ProLIF молча — на этапе расчёта отпечатка, а не сборки пакета.
    if not _is_present(structure["structure.chain"]):
        raise KlifsDataError(
            f"У структуры {structure_klifs_id} ({pdb_id}) не заполнена цепь: построить "
            f"нотацию остатков ProLIF нельзя"
        )
    chain = str(structure["structure.chain"]).strip()

    target_dir = targets_dir / pdb_id
    paths = download_target_files(session, structure_klifs_id, target_dir, chain)
    residue_to_position = build_residue_to_position(session, structure_klifs_id, chain)

    package: dict[str, Any] = {
        "klifs_structure_id": structure_klifs_id,
        "pdb_id": pdb_id,
        "chain": chain,
        "kinase_name": str(structure["kinase.klifs_name"]),
        "resolution": float(structure["structure.resolution"]),
        "quality_score": float(structure["structure.qualityscore"]),
        "protein_pdb": paths["protein_pdb"],
        "ligand_sdf": paths["ligand_sdf"],
        "pocket_pdb": paths["pocket_pdb"],
        "protein_noh_pdb": paths["protein_noh_pdb"],
        "protonation": protonation,
        "protonation_tool": protonation_tool,
        "residue_to_position": residue_to_position,
        "klifs_ifp_bits": str(fingerprint) if is_valid_fingerprint(fingerprint) else None,
        "dfg": str(structure["structure.dfg"]),
        "ac_helix": str(structure["structure.ac_helix"]),
        "inhibitor_type": classify_inhibitor_type(
            structure["structure.dfg"], structure["structure.ac_helix"]
        ),
        "alternate_model": str(structure["structure.alternate_model"]).strip(),
        "pocket_sequence": str(structure["structure.pocket"]).strip(),
    }

    # Полнота кармана считается при сборке, а не при чтении: позиция без
    # остатка даёт в отпечатке ноль, неотличимый от честного «взаимодействия здесь нет»,
    # и знать об этом нужно до расчёта, а не после сверки с эталоном.
    # Здесь же — единственное место, где карта позиций и строка кармана сверяются
    # поштучно: обе приходят из одной выгрузки, и разойтись они могут только ошибкой.
    try:
        check_sequence_agreement(package, MISSING_VALUE_MARKERS)
        package.update(completeness_fields(package))
    except PocketCompletenessError as error:
        raise KlifsDataError(f"{pdb_id}: карман собран неверно — {error}") from error

    target_json = target_dir / "target.json"
    # Перевод строки задан явно. На Windows `write_text` заменяет его парой CR+LF,
    # и пакет, собранный нативно, отличался бы от собранного в контейнере каждой
    # строкой сразу. Хэш пакета считается по байтам файла (`package_sha256`,
    # формат паспорта прогона), поэтому расчёт признал бы такой пакет чужим — при том что
    # `.gitattributes` держит `eol=lf` и в git обе версии выглядят одинаково.
    target_json.write_text(
        json.dumps(package, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    # Единственное место, где пакет без режима протонирования считается законным:
    # он ещё собирается, режим проставит следующий шаг. Все остальные вызовы —
    # строгие, поэтому пакет, брошенный на этом шаге, дальше не пройдёт.
    validate_target_json(target_json, require_protonation=False)
    return target_json


def validate_target_json(path: Path, *, require_protonation: bool = True) -> None:
    """Проверяет пакет мишени на соответствие формату пакета мишени.

    Ничего не возвращает; при любом расхождении поднимает `KlifsDataError` с указанием,
    что именно не так. Вызывается сразу после сборки: пакет, который не проходит
    собственную проверку, не должен дожить до расчёта отпечатка.

    `require_protonation=False` разрешает `protonation: null` — пакет в процессе сборки,
    режим которому ещё предстоит проставить. Умолчание строгое: забывчивость обязана
    вести к отказу, а не к пропуску, поэтому послабление запрашивается явно и только
    сборкой пакета.
    """
    if not path.is_file():
        raise KlifsDataError(f"Файл пакета мишени не найден: {path}")

    try:
        package = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise KlifsDataError(f"{path}: файл не разбирается как JSON — {error}") from error

    if not isinstance(package, dict):
        raise KlifsDataError(f"{path}: ожидался объект JSON, получен {type(package).__name__}")

    missing = [field for field in TARGET_REQUIRED_FIELDS if field not in package]
    if missing:
        raise KlifsDataError(f"{path}: в пакете мишени нет полей {missing}")

    base = path.parent
    for field in ("protein_pdb", "ligand_sdf", "pocket_pdb", "protein_noh_pdb"):
        value = package[field]
        if not isinstance(value, str) or not value:
            raise KlifsDataError(f"{path}: поле {field} пусто")
        if Path(value).is_absolute():
            raise KlifsDataError(
                f"{path}: путь {field}={value!r} абсолютный, формат пакета мишени требует "
                f"относительного — иначе пакет не переживёт копирование в Modal Volume"
            )
        if not (base / value).is_file():
            raise KlifsDataError(f"{path}: файл {field}={value!r} не существует")

    protonation = package["protonation"]
    if protonation is None:
        if require_protonation:
            raise KlifsDataError(
                f"{path}: режим протонирования не проставлен. Прогоните "
                f"scripts/annotate_protonation.py --target {path} — он посчитает водороды "
                f"и запишет protonation. Отпечатки, посчитанные в разных режимах, "
                f"несравнимы между собой"
            )
    elif protonation not in PROTONATION_MODES:
        raise KlifsDataError(
            f"{path}: protonation={protonation!r}, допустимы {sorted(PROTONATION_MODES)}"
        )
    if protonation == "explicit" and not package["protonation_tool"]:
        raise KlifsDataError(
            f"{path}: при protonation='explicit' поле protonation_tool обязано называть "
            f"инструмент и версию — режим должен быть виден в данных"
        )

    sequence = package["pocket_sequence"]
    if not isinstance(sequence, str) or len(sequence) != N_KLIFS_POSITIONS:
        raise KlifsDataError(
            f"{path}: pocket_sequence должна быть строкой из {N_KLIFS_POSITIONS} символов, "
            f"получено {len(sequence) if isinstance(sequence, str) else type(sequence).__name__}"
        )

    mapping = package["residue_to_position"]
    if not isinstance(mapping, dict) or not mapping:
        raise KlifsDataError(f"{path}: residue_to_position пусто")

    for residue, position in mapping.items():
        if not isinstance(position, int) or not 1 <= position <= N_KLIFS_POSITIONS:
            raise KlifsDataError(
                f"{path}: позиция остатка {residue} равна {position!r}, ожидалось целое "
                f"в диапазоне 1..{N_KLIFS_POSITIONS}"
            )

    # Позиций выравнивания всегда 85, но часть из них у конкретной киназы не занята:
    # остатка там нет вовсе. Отсюда и берётся расхождение «84 остатка против 85 позиций».
    # Равенство ниже отличает законный пропуск от остатка, потерянного при сборке:
    # ошибка в отображении позиций выглядела бы точно так же, но нарушила бы его.
    # Проверяется после самих позиций: сумма — агрегат, и без валидных слагаемых
    # её нарушение ничего не объясняет.
    gaps = sum(1 for letter in sequence if letter in MISSING_VALUE_MARKERS)
    if gaps + len(mapping) != N_KLIFS_POSITIONS:
        raise KlifsDataError(
            f"{path}: {len(mapping)} остатков и {gaps} пропусков в pocket_sequence "
            f"не складываются в {N_KLIFS_POSITIONS} позиций — часть остатков потеряна "
            f"при сборке отображения"
        )

    # Поля полноты кармана проверяются при наличии, а не требуются:
    # `target.json` входит в файлы, по которым считается `package_sha256`, и сделать
    # поле обязательным значило бы дописать его в пакет `data/targets/6tgu/`, сменив
    # хэш — шестнадцать прогонов в `runs/` стали бы «произведёнными на другом пакете»
    # Поэтому поля
    # несут пакеты, собранные заново, а у старых полнота считается на лету.
    try:
        check_recorded_completeness(package)
    except PocketCompletenessError as error:
        raise KlifsDataError(f"{path}: {error}") from error

    inhibitor_type = package["inhibitor_type"]
    if inhibitor_type not in TARGET_INHIBITOR_TYPES:
        raise KlifsDataError(
            f"{path}: тип ингибирования {inhibitor_type!r} не входит в "
            f"{list(TARGET_INHIBITOR_TYPES)}. Проект работает только по этим типам; "
            f"расширение списка — правка TARGET_INHIBITOR_TYPES в config.py"
        )

    bits = package["klifs_ifp_bits"]
    if bits is not None and not is_valid_fingerprint(bits):
        raise KlifsDataError(
            f"{path}: klifs_ifp_bits должен быть строкой из {KLIFS_IFP_LENGTH} символов "
            f"«0»/«1» либо null"
        )


def fetch_kinase_groups(session: Any, kinase_names: Sequence[str]) -> dict[tuple[str, str], str]:
    """Группа киназы (`AGC`, `CMGC`, `TK`, …) по паре «имя киназы, вид».

    Отдельный запрос нужен по той же причине, что и у `fetch_fingerprints`: в таблице
    `all_structures()` колонки `kinase.group`, `kinase.family` и `kinase.names`
    приходят **пустыми во всех 7465 строках** — проверено 17.09.2026 на выгрузке
    в `data/klifs/`. Не отдаёт групп и `all_kinases()`: у него шесть колонок,
    и групп среди них нет. Отдаёт только `by_kinase_name`.

    Ключ парный, а не по имени: одно и то же имя KLIFS носят киназы разных видов,
    и группы у них расходятся — `CK2a2` у человека числится в `CMGC`, а у другого
    вида в `Other`. Отбор по одному имени смешал бы их молча.

    Имена уходят пачками: список KLIFS подставляет в адрес запроса, и на двух сотнях
    имён запрос упёрся бы в предел длины — то же ограничение, что у отпечатков.
    """
    имена = sorted({str(имя) for имя in kinase_names if isinstance(имя, str) and имя.strip()})
    if not имена:
        return {}

    группы: dict[tuple[str, str], str] = {}
    for начало in range(0, len(имена), KLIFS_FINGERPRINT_BATCH_SIZE):
        пачка = имена[начало : начало + KLIFS_FINGERPRINT_BATCH_SIZE]
        df = session.kinases.by_kinase_name(пачка)
        if not isinstance(df, pd.DataFrame):
            raise KlifsDataError(f"KLIFS вернул {type(df)!r} вместо таблицы киназ")
        _require_columns(
            df, ("kinase.klifs_name", "kinase.group", "species.klifs"), "таблица киназ KLIFS"
        )
        for имя, группа, вид in zip(
            df["kinase.klifs_name"], df["kinase.group"], df["species.klifs"], strict=True
        ):
            if isinstance(группа, str) and группа.strip():
                группы[(str(имя), str(вид))] = группа.strip()
    return группы
