"""Перевыбор мишени и выборки сверки с учётом условий DiffSBDD.

`select_target` и `select_best_per_kinase` (`klifs.py`) отбирают структуру по качеству
кристалла и наличию эталонного отпечатка. Про состав кармана они ничего не знают —
и не могут: в пакете KLIFS гетерогрупп нет вовсе. Здесь тот же порядок ранжирования
дополняется одной проверкой: структура берётся первая **годная**, то есть та, чей
карман соответствует условиям генеративной модели.

Порядок не переписывается, а вызывается: `rank_structures` — то же место, что
в `klifs.select_target`. Иначе процедура отбора существовала бы в двух видах,
и расхождение между ними никто бы не заметил.

Перебор идёт сверху вниз и **останавливается на первой годной**: скачивать всю базу
незачем, а число проверенных кандидатов и причины отказа возвращаются вместе
с результатом — без них «взяли вот эту» не отличить от «первая попавшаяся подошла».
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kinase_ifp.config import DIFFSBDD_DATASET_CROSSDOCKED, DIFFSBDD_POCKET_CUTOFF_A
from kinase_ifp.klifs import KlifsDataError, rank_structures
from kinase_ifp.pocket_content import PocketContentError, PocketVerdict, classify_structure

#: Предохранитель: сколько кандидатов проверять, прежде чем признать поиск неудачным.
#: Каждая проверка — загрузка файла с RCSB, и перебор без предела означал бы скачивание
#: тысяч структур на вопрос, ответ на который обычно находится в первом десятке.
#:
#: **Значение 25 оказалось мало и поднято по измерению.** У ALK2 первые 25 кандидатов
#: подряд несут в кармане сульфат или этиленгликоль — белок кристаллизуют в сульфатном
#: буфере, и сульфат садится в фосфат-связывающее место. Годная структура нашлась
#: на 57-м месте (`8r7g`). При 25 киназа выпадала из выборки, и выпадение выглядело бы
#: свойством киназы, а не следствием предохранителя.
DEFAULT_MAX_CANDIDATES: int = 80


@dataclass
class Reselection:
    """Итог перевыбора: что взято, что отвергнуто и почему."""

    chosen: PocketVerdict | None
    chosen_row: pd.Series | None
    rejected: list[PocketVerdict]
    failed: list[str]
    checked: int

    @property
    def changed_from(self) -> str | None:
        """Код структуры, стоявшей первой по прежнему правилу, если он не совпал."""
        if not self.rejected:
            return None
        return self.rejected[0].pdb_id


def candidates_in_rank_order(df: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    """Кандидаты в порядке `rank_structures`, по одному на код PDB.

    Дубликаты по коду PDB отбрасываются: разные записи KLIFS одной структуры (цепи
    и альтернативные модели) дают один и тот же файл RCSB, и проверять его повторно
    значит тратить загрузку на заведомо тот же ответ.
    """
    результат: list[tuple[str, str, pd.Series]] = []
    виденные: set[str] = set()
    for _, строка in rank_structures(df).iterrows():
        код = str(строка["structure.pdb_id"]).strip().lower()
        лиганд = str(строка["ligand.expo_id"]).strip()
        if not код or код in виденные or not лиганд:
            continue
        виденные.add(код)
        результат.append((код, лиганд, строка))
    return результат


def first_matching(
    candidates: Sequence[tuple[str, str, pd.Series]],
    cache_dir: Path,
    *,
    radius: float = DIFFSBDD_POCKET_CUTOFF_A,
    dataset: str = DIFFSBDD_DATASET_CROSSDOCKED,
    with_qed: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> Reselection:
    """Первый кандидат, чей карман соответствует условиям модели.

    Возвращает `Reselection` с вердиктом, отвергнутыми кандидатами и кандидатами,
    которые не удалось проверить. Пустой `chosen` означает, что среди проверенных
    годных нет, — это результат, а не ошибка: он говорит, что ограничение слишком
    жёсткое для этой выборки.
    """
    отвергнутые: list[PocketVerdict] = []
    не_вышло: list[str] = []
    проверено = 0

    for код, лиганд, строка in candidates[:max_candidates]:
        проверено += 1
        try:
            вердикт = classify_structure(
                код,
                лиганд,
                cache_dir,
                radius=radius,
                with_qed=with_qed,
                dataset=dataset,
            )
        except PocketContentError as ошибка:
            не_вышло.append(f"{код} {лиганд}: {ошибка}")
            continue
        if вердикт.matches_diffsbdd:
            return Reselection(
                chosen=вердикт,
                chosen_row=строка,
                rejected=отвергнутые,
                failed=не_вышло,
                checked=проверено,
            )
        отвергнутые.append(вердикт)

    return Reselection(
        chosen=None, chosen_row=None, rejected=отвергнутые, failed=не_вышло, checked=проверено
    )


def reselect_target(
    structures: pd.DataFrame,
    fingerprint_ids: Collection[int],
    cache_dir: Path,
    **kwargs: object,
) -> Reselection:
    """Мишень: лучшая структура, у которой есть эталонный отпечаток **и** годный карман.

    Два прежних условия сохраняются полностью — меняется только то, что к ним добавлено
    третье. Структура без эталона пропускается до всякой загрузки: сверка с KLIFS
    составляет смысл работы, и тратить на неё сеть незачем.
    """
    if structures.empty:
        raise KlifsDataError("После фильтрации не осталось ни одной структуры")
    с_эталоном = structures[structures["structure.klifs_id"].isin(set(fingerprint_ids))]
    if с_эталоном.empty:
        raise KlifsDataError("Ни у одной отобранной структуры нет эталонного отпечатка KLIFS")
    return first_matching(candidates_in_rank_order(с_эталоном), cache_dir, **kwargs)  # type: ignore[arg-type]


def reselect_calibration(
    structures: pd.DataFrame,
    fingerprint_ids: Collection[int],
    kinase_names: Sequence[str],
    cache_dir: Path,
    **kwargs: object,
) -> dict[str, Reselection]:
    """По одной годной структуре на киназу — замена `select_best_per_kinase` для сверки.

    Киназа, у которой годной структуры не нашлось, остаётся в словаре с пустым `chosen`.
    Молча уменьшать выборку нельзя: медиана по девяти киназам и по двенадцати — разные
    числа, и разница обязана быть видна, а не спрятана в длине списка.
    """
    с_эталоном = structures[structures["structure.klifs_id"].isin(set(fingerprint_ids))]
    итог: dict[str, Reselection] = {}
    for имя in kinase_names:
        свои = с_эталоном[с_эталоном["kinase.klifs_name"] == имя]
        if свои.empty:
            итог[имя] = Reselection(
                chosen=None, chosen_row=None, rejected=[], failed=[], checked=0
            )
            continue
        итог[имя] = first_matching(candidates_in_rank_order(свои), cache_dir, **kwargs)  # type: ignore[arg-type]
    return итог


def read_reselected_sample(path: Path) -> list[str]:
    """Коды структур записанного состава перевыбора, в порядке файла.

    Файл пишет `scripts/classify_complexes.py --reselect-calibration`. Читать его,
    а не звать перевыбор заново, нужно затем, что перевыбор идёт по сети: каждая
    проверка кармана — загрузка структуры с RCSB, и повторять её ради состава,
    который уже записан, незачем.

    Строка с `matches_diffsbdd = 0` роняет чтение, а не отбрасывается. Тот же скрипт
    ключами `--calibration` и `--sample` пишет файл **той же схемы** с негодными
    строками, и тихая фильтрация такого файла молча дала бы выборку из пяти киназ
    вместо двенадцати.
    """
    if not path.is_file():
        raise KlifsDataError(
            f"Нет файла состава {path}. Соберите его командой "
            f"scripts/classify_complexes.py --reselect-calibration --out {path}"
        )

    таблица = pd.read_csv(path)
    if "pdb_id" not in таблица.columns:
        raise KlifsDataError(f"В {path} нет колонки pdb_id: это не файл состава перевыбора")

    коды = [str(код).strip().lower() for код in таблица["pdb_id"]]

    if "matches_diffsbdd" in таблица.columns:
        негодные = [
            код
            for код, годна in zip(коды, таблица["matches_diffsbdd"], strict=True)
            if int(годна) != 1
        ]
        if негодные:
            raise KlifsDataError(
                f"В {path} есть структуры, не прошедшие условия DiffSBDD: {негодные}. "
                f"Это файл классификации, а не перевыбора: пересоберите его ключом "
                f"--reselect-calibration"
            )

    повторы = sorted({код for код in коды if коды.count(код) > 1})
    if повторы:
        raise KlifsDataError(f"В {path} код структуры встречается дважды: {повторы}")

    return коды


def select_reselected(
    structures: pd.DataFrame,
    pdb_ids: Sequence[str],
    fingerprint_ids: Collection[int],
    *,
    expect_kinases: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Строки KLIFS названного состава — вход `calibration.calibrate_sample`.

    Принимает отфильтрованную таблицу структур, коды состава (`read_reselected_sample`)
    и идентификаторы структур с эталонным отпечатком. Возвращает по одной строке
    на код, в порядке `pdb_ids`.

    Правило выбора записи внутри кода то же, что у `select_best_per_kinase`: лучшая
    по `rank_structures` из тех, у которых есть эталон. У одного кода записей KLIFS
    несколько — альтернативные модели и цепи, — и брать первую попавшуюся нельзя:
    боковые цепи моделей расходятся, а с ними и отпечаток.

    Любое расхождение состава роняет вызов: отсутствующий код, код без эталона,
    две структуры одной киназы, несовпадение с `expect_kinases`. Причина у всех одна —
    медиана по одиннадцати киназам и по двенадцати это разные числа, и сравнение
    «было/стало» на разных выборках не значит ничего. Молча уменьшенная выборка
    выглядела бы свойством метода, а не следствием пропущенной структуры.
    """
    if not pdb_ids:
        raise KlifsDataError("Состав перевыбора пуст: выборку строить не из чего")

    эталоны = set(fingerprint_ids)
    коды_таблицы = structures["structure.pdb_id"].astype(str).str.strip().str.lower()

    строки: list[pd.Series] = []
    нет_записи: list[str] = []
    нет_эталона: list[str] = []
    for код in pdb_ids:
        свои = structures.loc[коды_таблицы == код.strip().lower()]
        if свои.empty:
            нет_записи.append(код)
            continue
        с_эталоном = свои.loc[[int(i) in эталоны for i in свои["structure.klifs_id"]]]
        if с_эталоном.empty:
            нет_эталона.append(код)
            continue
        строки.append(rank_structures(с_эталоном).iloc[0])

    if нет_записи:
        raise KlifsDataError(
            f"В таблице структур нет записей для {нет_записи}. Состав перевыбора и выгрузка "
            f"KLIFS разошлись: обновите выгрузку либо пересоберите состав"
        )
    if нет_эталона:
        raise KlifsDataError(
            f"Нет эталонного отпечатка KLIFS для {нет_эталона}. Пополните "
            f"data/klifs/klifs_ifp.csv (scripts/fetch_klifs_ifp.py --limit 0)"
        )

    выборка = pd.DataFrame(строки).reset_index(drop=True)
    киназы = [str(имя) for имя in выборка["kinase.klifs_name"]]
    дубли = sorted({имя for имя in киназы if киназы.count(имя) > 1})
    if дубли:
        raise KlifsDataError(
            f"Киназа входит в состав дважды: {дубли}. В медиану она попала бы дважды"
        )

    if expect_kinases is not None:
        лишние = sorted(set(киназы) - set(expect_kinases))
        недостающие = sorted(set(expect_kinases) - set(киназы))
        if лишние or недостающие:
            raise KlifsDataError(
                f"Состав перевыбора не совпадает с ожидаемым: лишние киназы {лишние}, "
                f"недостающие {недостающие}"
            )

    return выборка
