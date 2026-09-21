"""Сходства позиционных отпечатков взаимодействий.

Две меры, а не одна. Танимото симметричен и штрафует молекулу за контакты, которых
нет у эталона; Tversky с `alpha=1, beta=0` такого штрафа не накладывает и отвечает
ровно на вопрос «какую долю взаимодействий эталона молекула воспроизвела»
(`docs/metrics.md`, раздел 3.1). Обе метрики нужны в отчёте вместе: одна без другой
даёт однобокую картину.

Здесь только арифметика над битовыми массивами: ни RDKit, ни ProLIF, ни структур.
Отпечатки производит `fingerprint.compute_ifp`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kinase_ifp.config import (
    KLIFS_IFP_SHAPE,
    KLIFS_INTERACTION_TYPES,
    N_KLIFS_INTERACTION_TYPES,
    N_KLIFS_POSITIONS,
)


def select_types(ifp: np.ndarray, types: Sequence[str]) -> np.ndarray:
    """Оставляет в отпечатке строки только перечисленных типов взаимодействий.

    Принимает массив формы `KLIFS_IFP_SHAPE` = (7, 85) и имена типов из
    `KLIFS_INTERACTION_TYPES`; возвращает массив формы (len(types), 85) в порядке
    аргумента `types`. Нужен скору, который считается по
    `SCORING_INTERACTION_TYPES` — всем типам, кроме `HYD`.

    Отдельная функция, а не аргумент у `tanimoto` и `tversky`: так сигнатуры мер
    сходства остаются ровно такими, как записано в формате мер сходства, и код, написанный
    на формат, не приходится править.

    Поднимает `ValueError` при неверной форме входа, неизвестном имени типа
    и при повторе имени в списке.
    """
    bits = _as_bits(ifp, "ifp")
    if bits.shape != KLIFS_IFP_SHAPE:
        raise ValueError(
            f"Отбор типов возможен только по полному отпечатку формы {KLIFS_IFP_SHAPE}, "
            f"получена форма {bits.shape}"
        )
    if not types:
        raise ValueError("Список типов пуст: сходство считать не по чему")

    unknown = [t for t in types if t not in KLIFS_INTERACTION_TYPES]
    if unknown:
        raise ValueError(
            f"Неизвестные типы взаимодействий {unknown}; допустимы {list(KLIFS_INTERACTION_TYPES)}"
        )

    # Повтор имени удваивает вес типа: срез получает две одинаковые строки, и мера
    # считает их независимыми битами. Молекула с одним типом объявляется полностью
    # воспроизводящей эталон с двумя. Сверить состав типов у пары
    # срезов `_as_pair` не может — она видит только форму, — поэтому перехватываем
    # здесь, где имена ещё известны.
    duplicates = sorted({t for t in types if list(types).count(t) > 1})
    if duplicates:
        raise ValueError(
            f"Типы взаимодействий повторяются {duplicates}: повтор удваивает вклад типа "
            "в меру сходства"
        )

    return np.asarray(ifp)[type_rows(types)]


def type_rows(types: Sequence[str]) -> list[int]:
    """Номера строк отпечатка для перечисленных типов взаимодействий.

    Отдельно от `select_types` затем, что отбирать по типам приходится не только биты:
    вариант потенциала М2 (`kinase_ifp.potential`) так же режет по типам массив **весов**,
    а он вещественный и проверку `_as_bits` не прошёл бы. Проверки имён при этом обязаны
    остаться теми же — иначе два места расходятся, и опечатка в имени типа выясняется
    только по неверному числу.
    """
    if not types:
        raise ValueError("Список типов пуст: сходство считать не по чему")
    unknown = [t for t in types if t not in KLIFS_INTERACTION_TYPES]
    if unknown:
        raise ValueError(
            f"Неизвестные типы взаимодействий {unknown}; допустимы {list(KLIFS_INTERACTION_TYPES)}"
        )
    duplicates = sorted({t for t in types if list(types).count(t) > 1})
    if duplicates:
        raise ValueError(
            f"Типы взаимодействий повторяются {duplicates}: повтор удваивает вклад типа "
            "в меру сходства"
        )
    return [KLIFS_INTERACTION_TYPES.index(t) for t in types]


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Коэффициент Танимото двух отпечатков: |A ∩ B| / |A ∪ B|.

    Принимает массивы одинаковой формы — (7, 85) либо срез `select_types`. Симметричен.
    При пустом объединении битов возвращает 0.0: два отпечатка без
    единого взаимодействия не «полностью похожи», а несравнимы, и 1.0 здесь завысила бы
    метрику у молекул, которые в карман вообще не попали.
    """
    first, second = _as_pair(a, b)

    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first & second)) / union


def tversky(a: np.ndarray, b: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> float:
    """Несимметричное сходство Tversky: c / (c + alpha·|A∖B| + beta·|B∖A|).

    **Порядок аргументов важен.** `alpha` штрафует биты, которые есть только у первого
    аргумента, `beta` — только у второго. Поэтому «доля взаимодействий эталона,
    воспроизведённых молекулой» из `docs/metrics.md` (3.1) — это `tversky(эталон,
    молекула)` со значениями по умолчанию `alpha=1, beta=0`. Переставленные местами
    аргументы дают долю битов молекулы: число правдоподобное, но отвечающее на другой
    вопрос, и ошибка такого рода из таблицы не видна.

    Частные случаи формулы: `alpha=beta=1` — Танимото, `alpha=beta=0.5` — Дайс.
    При нулевом знаменателе возвращает 0.0 — как и `tanimoto`.
    """
    if alpha < 0 or beta < 0:
        raise ValueError(f"Веса Tversky не могут быть отрицательными: alpha={alpha}, beta={beta}")

    first, second = _as_pair(a, b)

    common = int(np.count_nonzero(first & second))
    only_first = int(np.count_nonzero(first & ~second))
    only_second = int(np.count_nonzero(second & ~first))

    denominator = common + alpha * only_first + beta * only_second
    if denominator == 0:
        return 0.0
    return float(common) / float(denominator)


def as_full_ifp(ifp: np.ndarray, name: str = "ifp") -> np.ndarray:
    """Проверяет, что отпечаток полный (7, 85), и возвращает булев массив.

    Нужна там, где тип взаимодействия ищется по его номеру в `KLIFS_INTERACTION_TYPES`:
    в `evaluation.target` и в `kinase_ifp.hinge`. На срезе те же номера указали бы
    на другие типы — ошибка, которую арифметика не заметит, поэтому срез здесь
    недопустим, в отличие от мер сходства.
    """
    bits = _as_bits(ifp, name)
    if bits.shape != KLIFS_IFP_SHAPE:
        raise ValueError(
            f"Отпечаток {name}: ожидался полный отпечаток формы {KLIFS_IFP_SHAPE}, "
            f"получена {bits.shape} — срез по типам взаимодействий здесь не годится"
        )
    return bits


def _as_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Проверяет пару отпечатков и переводит их в булевы массивы.

    Формы обязаны совпадать: numpy молча растянул бы срез (1, 85) на полный отпечаток
    (7, 85) и вернул осмысленное на вид, но неверное число.

    Сверяется **только форма**. Два среза, взятые `select_types` с разными списками
    типов, имеют одинаковую форму и проходят проверку, хотя общих взаимодействий
    у них нет вовсе; состав типов лежит на вызывающем. В самом проекте
    такого вызова нет: `ifp_score` строит оба среза одним списком
    `SCORING_INTERACTION_TYPES`, метрики работают с полными отпечатками.
    """
    first = _as_bits(a, "a")
    second = _as_bits(b, "b")
    if first.shape != second.shape:
        raise ValueError(
            f"Отпечатки разной формы: {first.shape} и {second.shape}. Сравнивать можно "
            "только срезы одинакового размера"
        )
    return first, second


def _as_bits(ifp: np.ndarray, name: str) -> np.ndarray:
    """Проверяет форму и значения отпечатка, возвращает булев массив.

    Форма не выводится из данных, а сверяется с конфигурацией: перепутанные оси дают
    массив (85, 7), с которым арифметика проходит без ошибки, — ровно тем и был опасен
    отменённый вариант раскладки.
    """
    array = np.asarray(ifp)
    if array.ndim != 2 or array.shape[1] != N_KLIFS_POSITIONS:
        raise ValueError(
            f"Отпечаток {name}: ожидалась форма (число типов, {N_KLIFS_POSITIONS}), "
            f"получена {array.shape}"
        )
    if not 1 <= array.shape[0] <= N_KLIFS_INTERACTION_TYPES:
        raise ValueError(
            f"Отпечаток {name}: число типов {array.shape[0]} вне диапазона "
            f"1..{N_KLIFS_INTERACTION_TYPES}"
        )
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"Отпечаток {name}: допустимы только значения 0 и 1")
    return array.astype(bool)
