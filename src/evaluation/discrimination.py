"""Метрики плана Б: различает ли скор верное размещение лиганда.

Проверяется **мера**, а не модель: набор поз строится из кристаллического
лиганда, отклонение каждой позы известно точно, и вопрос ставится так — растёт ли скор
по мере приближения к кристаллической позе. Генерация для этого не нужна, поэтому
измерение выполняется на любой мишени, для которой собран пакет, и не зависит ни
от GPU, ни от клона DiffSBDD.

Три числа и по какому набору считается каждое:

- **ROC-AUC** — по двум классам поз: ближе `NEAR_POSE_RMSD_A` против дальше
  `CORRECT_POSE_RMSD_A`. Полоса между границами в сравнение не входит;
- **ранг нативной позы** — по всему набору;
- **Спирмен скора с `rmsd_to_ref`** — по всему набору, должен быть отрицательным.

**Ранг здесь не берётся из `ranking.csv` и это не дублирование.** Колонка `rank`
формата таблицы ранжирования разрешает ничьи порядком строк в файле, и ранг 1 у нативной позы может
означать «скор выше всех», а может — «скор такой же, как ещё у четырёх, но она записана
первой». Утверждение о качестве меры не может держаться на порядке
записи, поэтому ранг считается заново от числа **строго** лучших поз, а число ничьих
возвращается отдельным полем и печатается рядом.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Final

from evaluation.stats import StatsError, mann_whitney_u, spearman_rho
from experiments.layout import (
    CSV_EOL,
    DISCRIMINATION_SUMMARY_CSV,
    DISCRIMINATION_SUMMARY_MD,
    PLAN_B_METRICS_JSON,
    SOURCE_LOCAL_POSES,
)
from experiments.ranking import read_ranking
from experiments.run_io import read_molecule_props
from experiments.runs import RunError, read_passport, target_for_run
from kinase_ifp.config import CORRECT_POSE_RMSD_A, NEAR_POSE_RMSD_A
from kinase_ifp.poses import POSE_KIND_NATIVE

#: Имя SD-свойства, которым набор поз помечает способ построения позы.
POSE_KIND_PROP: Final[str] = "pose_kind"

#: Колонки сводки по прогонам. Объявлены здесь, а не в скрипте: писатель и читатель
#: обязаны видеть один состав — то же правило, что у `calibration.STRUCTURE_COLUMNS`.
DISCRIMINATION_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "pdb_id",
    "kinase",
    "seed",
    "n_poses",
    "n_near",
    "n_far",
    "roc_auc",
    "spearman_rho",
    "native_rank",
    "native_ties",
)


class DiscriminationError(RuntimeError):
    """Метрики плана Б по этому прогону посчитать нельзя; в тексте — почему."""


@dataclass(frozen=True)
class PoseDiscrimination:
    """Различительная способность скора на одном наборе поз.

    `native_rank` — сколько поз набора имеют **строго** больший скор, плюс единица;
    `native_ties` — сколько других поз имеют ровно такой же скор. Ранг 1 при ненулевых
    ничьих означает «скор нативной позы максимален», а не «нативная поза выделена
    среди прочих», и эти два утверждения в тексте различаются.
    """

    run_id: str
    pdb_id: str
    kinase: str
    seed: int
    n_poses: int
    n_near: int
    n_far: int
    roc_auc: float
    spearman_rho: float
    native_rank: int
    native_ties: int

    @property
    def native_is_best(self) -> bool:
        """Скор нативной позы строго выше всех прочих — то, что меряет top-1."""
        return self.native_rank == 1 and self.native_ties == 0

    def as_row(self) -> dict[str, Any]:
        """Строка сводки: поля в порядке `DISCRIMINATION_COLUMNS`."""
        return {
            "run_id": self.run_id,
            "pdb_id": self.pdb_id,
            "kinase": self.kinase,
            "seed": self.seed,
            "n_poses": self.n_poses,
            "n_near": self.n_near,
            "n_far": self.n_far,
            "roc_auc": f"{self.roc_auc:.4f}",
            "spearman_rho": f"{self.spearman_rho:.4f}",
            "native_rank": self.native_rank,
            "native_ties": self.native_ties,
        }


def roc_auc(near: Sequence[float], far: Sequence[float]) -> float:
    """Доля пар «близкая поза против дальней», где у близкой скор выше; ничья — половина.

    Считается из U-статистики Манна–Уитни (`evaluation.stats`), а не отдельным двойным
    циклом: U с поправкой на связи — это ровно число выигранных пар плюс половина ничьих,
    и вторая реализация того же счёта разошлась бы с первой на связках, которых у скора
    большинство.
    """
    if not near or not far:
        raise DiscriminationError(
            f"ROC-AUC не определён: поз ближе {NEAR_POSE_RMSD_A} Å — {len(near)}, "
            f"дальше {CORRECT_POSE_RMSD_A} Å — {len(far)}. Нужны обе группы"
        )
    u, _ = mann_whitney_u(near, far)
    return u / (len(near) * len(far))


def discriminate_run(run_dir: Path, targets_dir: Path | None = None) -> PoseDiscrimination:
    """Считает метрики плана Б по одному прогону набора поз.

    Читает паспорт, `ranking.csv` и свойства молекул `rmsd_to_ref`
    и `pose_kind`. Прогон генерации отвергается: эталонной позы у него нет,
    и все три метрики лишены смысла — это отказ с причиной, а не пустые числа.

    `targets_dir` нужен только ради имени киназы в сводке: в паспорте прогона его нет,
    а `pdb_id` для человека, читающего таблицу по одиннадцати киназам, не то же самое,
    что «CK2a2». Пакет ищется по паспорту (`runs.target_for_run`), а не по имени папки.
    """
    _require_pose_set(run_dir)
    скоры = {
        строка["mol_id"]: float(строка["ifp_score"])
        for строка in read_ranking(run_dir)
        if строка.get("ifp_score", "") != ""
    }
    return discriminate_scores(run_dir, скоры, targets_dir)


def _require_pose_set(run_dir: Path) -> dict[str, Any]:
    """Паспорт прогона, если это набор поз; иначе отказ с причиной."""
    паспорт = read_passport(run_dir)
    if str(паспорт.get("source")) != SOURCE_LOCAL_POSES:
        raise DiscriminationError(
            f"{run_dir.name}: source={паспорт.get('source')}, а метрики плана Б считаются "
            f"по набору поз ({SOURCE_LOCAL_POSES}): у сгенерированных молекул нет "
            f"эталонной позы, и отклонение от неё не определено"
        )
    return паспорт


def pose_points(
    run_dir: Path,
    скоры: dict[str, float],
    свойства: dict[str, dict[str, str]] | None = None,
) -> list[tuple[str, float, float]]:
    """Точки набора поз: `(mol_id, скор, отклонение от кристаллической позы)`.

    Принимает папку набора и скоры молекул; возвращает список троек, из которого
    считаются все метрики различения. Выделена из `discriminate_scores` 18.09 ради
    рисунка «скор против отклонения позы»: до этого пары существовали только внутри
    расчёта, и нарисовать облако точек было не из чего — в файлах выхода этапа лежат
    одни агрегаты по наборам.

    Молекулы без `rmsd_to_ref` пропускаются: отклонение известно не у всех поз,
    а точка без одной координаты не точка.
    """
    свойства_поз = (
        свойства
        if свойства is not None
        else read_molecule_props(run_dir, ("rmsd_to_ref", POSE_KIND_PROP))
    )
    пары: list[tuple[str, float, float]] = []
    for mol_id, скор in скоры.items():
        сырое = свойства_поз.get(mol_id, {}).get("rmsd_to_ref", "")
        if сырое:
            пары.append((mol_id, скор, float(сырое)))
    if not пары:
        raise DiscriminationError(
            f"{run_dir.name}: ни у одной молекулы с посчитанным скором нет rmsd_to_ref"
        )
    return пары


def discriminate_scores(
    run_dir: Path, скоры: dict[str, float], targets_dir: Path | None = None
) -> PoseDiscrimination:
    """Те же метрики плана Б, но по скорам, переданным явно, а не из `ranking.csv`.

    Принимает папку набора поз и словарь `{mol_id: скор}`; возвращает
    `PoseDiscrimination`. Нужна пересчёту по правилам KLIFS
    (`experiments.klifs_rescore`): `ranking.csv` посчитан через ProLIF, а подмена
    источника скора не должна менять ни одного определения — ROC-AUC, ранг нативной
    позы и Спирмен считаются здесь одним кодом для обоих путей.
    """
    паспорт = _require_pose_set(run_dir)
    свойства = read_molecule_props(run_dir, ("rmsd_to_ref", POSE_KIND_PROP))
    пары = pose_points(run_dir, скоры, свойства)
    нативная = native_mol_id(свойства, пары)
    скор_нативной = next(скор for mol_id, скор, _ in пары if mol_id == нативная)

    near = [скор for _, скор, rmsd in пары if rmsd < NEAR_POSE_RMSD_A]
    far = [скор for _, скор, rmsd in пары if rmsd > CORRECT_POSE_RMSD_A]

    try:
        ро = spearman_rho([скор for _, скор, _ in пары], [rmsd for _, _, rmsd in пары])
    except StatsError as ошибка:
        raise DiscriminationError(f"{run_dir.name}: Спирмен не считается — {ошибка}") from ошибка

    цель = паспорт.get("target", {})
    return PoseDiscrimination(
        run_id=str(паспорт["run_id"]),
        pdb_id=str(цель.get("pdb_id", "")),
        kinase=_kinase_name(run_dir, targets_dir),
        seed=int(паспорт.get("sampling", {}).get("seed", 0)),
        n_poses=len(пары),
        n_near=len(near),
        n_far=len(far),
        roc_auc=roc_auc(near, far),
        spearman_rho=ро,
        native_rank=1 + sum(1 for _, скор, _ in пары if скор > скор_нативной),
        native_ties=sum(
            1 for mol_id, скор, _ in пары if скор == скор_нативной and mol_id != нативная
        ),
    )


def _kinase_name(run_dir: Path, targets_dir: Path | None) -> str:
    """Имя киназы из пакета мишени прогона; пусто, если пакет не задан или не найден.

    Отсутствие имени не отказ: метрики считаются по прогону, а имя нужно только
    подписи в сводке — терять из-за него измерение было бы неверным разменом.
    """
    if targets_dir is None:
        return ""
    try:
        пакет = target_for_run(run_dir, targets_dir)
    except RunError:
        return ""
    return str(json.loads(пакет.read_text(encoding="utf-8")).get("kinase_name", ""))


def native_mol_id(
    свойства: dict[str, dict[str, str]], пары: Sequence[tuple[str, float, float]]
) -> str:
    """`mol_id` нативной позы: по метке `pose_kind`, а при её отсутствии — по нулевому RMSD.

    Запасной путь нужен не для красоты: прогоны, записанные до появления метки, лежат
    в `runs/` и переписаны быть не могут. Если не нашлось ни метки, ни позы
    с нулевым отклонением, вызов падает: набор без нативной позы не даёт ни top-1,
    ни ранга, и молчаливая подстановка «лучшей из имеющихся» подменила бы утверждение.
    """
    помеченные = [
        mol_id
        for mol_id, _, _ in пары
        if свойства.get(mol_id, {}).get(POSE_KIND_PROP) == POSE_KIND_NATIVE
    ]
    if len(помеченные) > 1:
        raise DiscriminationError(
            f"в наборе {len(помеченные)} поз помечены как {POSE_KIND_NATIVE}: "
            f"нативная поза обязана быть одна"
        )
    if помеченные:
        return помеченные[0]

    нулевые = [mol_id for mol_id, _, rmsd in пары if rmsd == 0.0]
    if len(нулевые) == 1:
        return нулевые[0]
    raise DiscriminationError(
        f"нативная поза не найдена: меток {POSE_KIND_PROP}={POSE_KIND_NATIVE} нет, "
        f"поз с нулевым rmsd_to_ref — {len(нулевые)}"
    )


#: Колонки сводки по мишеням — производные числа для текста.
TARGET_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "pdb_id",
    "kinase",
    "n_runs",
    "roc_auc_median",
    "roc_auc_min",
    "roc_auc_max",
    "spearman_median",
    "spearman_min",
    "spearman_max",
    "native_rank_max",
    "native_ties_median",
    "top1_strict",
)


@dataclass(frozen=True)
class TargetSummary:
    """Сводка по мишени: медиана и размах каждой метрики по её наборам поз.

    В текст идут именно эти числа, а не значения отдельных наборов: один набор —
    одно испытание, и медиана с размахом по десяти сидам говорят о мере, тогда как
    одиночное значение говорит ещё и о том, какой сид взяли.
    """

    pdb_id: str
    kinase: str
    n_runs: int
    roc_auc: tuple[float, float, float]
    spearman: tuple[float, float, float]
    native_rank_max: int
    native_ties_median: float
    top1_strict: int

    def as_row(self) -> dict[str, Any]:
        """Строка сводки в порядке `TARGET_SUMMARY_COLUMNS`."""
        медиана, минимум, максимум = self.roc_auc
        ро_медиана, ро_мин, ро_макс = self.spearman
        return {
            "pdb_id": self.pdb_id,
            "kinase": self.kinase,
            "n_runs": self.n_runs,
            "roc_auc_median": f"{медиана:.4f}",
            "roc_auc_min": f"{минимум:.4f}",
            "roc_auc_max": f"{максимум:.4f}",
            "spearman_median": f"{ро_медиана:.4f}",
            "spearman_min": f"{ро_мин:.4f}",
            "spearman_max": f"{ро_макс:.4f}",
            "native_rank_max": self.native_rank_max,
            "native_ties_median": f"{self.native_ties_median:.1f}",
            "top1_strict": self.top1_strict,
        }


def summarize_by_target(measurements: Sequence[PoseDiscrimination]) -> list[TargetSummary]:
    """Сводит измерения по мишеням: медиана и размах по наборам поз каждой из них.

    Мишени не смешиваются: сходство с эталоном считается к своему эталону, и общая
    медиана по двум мишеням не отвечала бы ни на один вопрос.
    """
    if not measurements:
        raise DiscriminationError("сводить нечего: список измерений пуст")

    по_мишеням: dict[str, list[PoseDiscrimination]] = {}
    for измерение in measurements:
        по_мишеням.setdefault(измерение.pdb_id, []).append(измерение)

    сводка: list[TargetSummary] = []
    for pdb_id, группа in sorted(по_мишеням.items()):
        auc = [м.roc_auc for м in группа]
        ро = [м.spearman_rho for м in группа]
        выиграно, _ = top1_rate(группа)
        сводка.append(
            TargetSummary(
                pdb_id=pdb_id,
                kinase=next((м.kinase for м in группа if м.kinase), ""),
                n_runs=len(группа),
                roc_auc=(median(auc), min(auc), max(auc)),
                spearman=(median(ро), min(ро), max(ро)),
                native_rank_max=max(м.native_rank for м in группа),
                native_ties_median=median([float(м.native_ties) for м in группа]),
                top1_strict=выиграно,
            )
        )
    return сводка


def write_target_summary(summaries: Sequence[TargetSummary], out_dir: Path) -> tuple[Path, Path]:
    """Пишет сводку по мишеням в `results/` парой файлов.

    Markdown — для чтения, CSV — те же числа машинно: сверка чисел текста
    (`scripts/check_numbers.py`) читает второй, человек — первый.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_путь = out_dir / DISCRIMINATION_SUMMARY_CSV
    with csv_путь.open("w", encoding="utf-8", newline=CSV_EOL) as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(TARGET_SUMMARY_COLUMNS))
        писатель.writeheader()
        for сводка in summaries:
            писатель.writerow(сводка.as_row())

    строки = [
        "# Различительная способность скора на наборах поз",
        "",
        "Собрано `scripts/plan_b_metrics.py`. Те же числа машинно — "
        f"в `{DISCRIMINATION_SUMMARY_CSV}`.",
        "",
        "| Мишень | Киназа | Наборов | ROC-AUC, медиана (размах) | "
        "Спирмен, медиана (размах) | Худшее место нативной позы | Ничьих, медиана | "
        "Строгих побед |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for с in summaries:
        # Числа берутся из той же строки, что уходит в CSV, а не форматируются заново:
        # читаемый двойник обязан совпадать с машинным до последнего знака. Иначе
        # на половинных значениях они расходятся (0.74845 даёт 0.7485 в CSV и 0.748
        # при печати), и сверка чисел текста подтверждает одно, а глаз читает другое.
        строка = с.as_row()
        строки.append(
            f"| {с.pdb_id} | {с.kinase} | {с.n_runs} | "
            f"{строка['roc_auc_median']} "
            f"({строка['roc_auc_min']}–{строка['roc_auc_max']}) | "
            f"{строка['spearman_median']} "
            f"(от {строка['spearman_max']} до {строка['spearman_min']}) | "
            f"{с.native_rank_max} | {строка['native_ties_median']} | "
            f"{с.top1_strict} из {с.n_runs} |"
        )
    строки.append("")
    md_путь = out_dir / DISCRIMINATION_SUMMARY_MD
    md_путь.write_text(CSV_EOL.join(строки), encoding="utf-8")
    return md_путь, csv_путь


def top1_rate(measurements: Sequence[PoseDiscrimination]) -> tuple[int, int]:
    """Сколько наборов из скольких дали нативной позе строго лучший скор.

    Доля, а не одно испытание: top-1 по единственному набору — это бросок монеты,
    и разные сиды набора для того и строятся (`make_local_poses.py --seed`).
    """
    if not measurements:
        raise DiscriminationError("top-1 по пустому списку наборов не определена")
    return sum(1 for m in measurements if m.native_is_best), len(measurements)


def write_plan_b_metrics(run_dir: Path, measurement: PoseDiscrimination) -> Path:
    """Пишет `plan_b_metrics.json` в папку прогона и возвращает путь."""
    путь = run_dir / PLAN_B_METRICS_JSON
    путь.write_text(
        json.dumps(measurement.as_row(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return путь


def write_discrimination_csv(measurements: Sequence[PoseDiscrimination], path: Path) -> Path:
    """Пишет сводку по наборам поз: строка на прогон, колонки `DISCRIMINATION_COLUMNS`.

    Перевод строки `\\n` (`CSV_EOL`): модуль `csv` по умолчанию ставит
    CRLF, и файл, записанный кодом, отличался бы в diff от того же файла, сохранённого
    git, во всех строках сразу.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline=CSV_EOL) as файл:
        писатель = csv.DictWriter(файл, fieldnames=list(DISCRIMINATION_COLUMNS))
        писатель.writeheader()
        for измерение in measurements:
            писатель.writerow(измерение.as_row())
    return path
