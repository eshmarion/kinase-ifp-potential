"""Пять проверок, усиливающих обоснование результатов текста.

Что считается и какой абзац текста этим держится:

* **Ключевая связь** — доля молекул с водородной связью до и после уточнения позы,
  критерий Макнемара и интервал на разность долей. Держит пункт 4 подраздела 5.5,
  где сейчас чисел нет вовсе. Колонки `hbond_before` и `hbond_after` таблиц уточнения
  для этого не годятся: в них записано «есть ли у молекулы хоть одна водородная связь
  с карманом», а ключевые позиции — те, где у эталона стоит донор или акцептор
, поэтому доля считается заново по отпечаткам поз.
* **Смещение** — смещение положения в опыте и в отрицательном контроле. Если смещения
  одинаковы, возражение «прирост даёт само шевеление позы» закрывается устройством
  опыта, а не только отсутствием прироста в контроле.
* **Отбор по размеру** — сравнение отобранных и отбракованных молекул при выравнивании по числу
  тяжёлых атомов. Отвечает на возражение «отобрали молекулы покрупнее» к подразделу 5.4.
  Сравниваются Танимото и доля с ключевой связью, но **не скор**: по скору топ выше
  тавтологически, потому что по нему и отбирался.
* **Прирост и старт** — связь прироста со стартовым скором помолекулярно. Превращает наблюдение
  по четырём прогонам в измерение по сотням молекул.
* **Поправка Холма** — поправка внутри семейств критериев.

Все числа — потенциала работы М1 (таблицы с суффиксом `-clash`).

    docker compose run --rm dev python scripts/thesis_checks.py \\
        --runs runs/2026-08-23-6tgu-s0-n100-prep ...
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np

from evaluation.paired_stats import (
    PairedBinary,
    exact_binomial_p,
    holm,
    mcnemar,
    sign_test,
)
from evaluation.size_matching import SizedItem, match_by_size
from evaluation.stats import StatsError, bootstrap_ci, mann_whitney_u, spearman_rho
from evaluation.target import hinge_hbond, key_hbond_positions
from experiments.klifs_rescore import reference_ifp, rescore_run
from experiments.results_io import Строка, записать
from experiments.runs import target_for_run
from kinase_ifp.config import KLIFS_INTERACTION_TYPES, RESULTS_DIR, TARGETS_DIR
from kinase_ifp.fingerprint_klifs import ifp_from_groups, load_pocket_rules
from kinase_ifp.ligand_flags import groups_from_mol
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.scoring import DEFAULT_TOP_FRACTION, ScoredMolecule, rank_molecules
from kinase_ifp.similarity import select_types, tversky

РЕЗУЛЬТАТЫ = RESULTS_DIR
ИМЯ = "thesis_checks"
КОМАНДА = "python scripts/thesis_checks.py --runs <прогоны>"

# Имена таблиц уточнения позы в `results/`. Суффикс `-clash` — потенциал работы
# М1; таблицы без него посчитаны при нулевом весе физического члена и сюда не входят.
ОПЫТ = "pose_optimization-{run}-all7-clash"
КОНТРОЛЬ = "pose_optimization-{run}-all7-control-clash"

# Расхождение скора, посчитанного здесь, со скором в таблице уточнения. Обе величины
# считаны по одним правилам на одних молекулах, поэтому совпадать должны до округления
# таблицы (четыре знака); больший разброс означает, что пары построены неверно.
ДОПУСК_СВЕРКИ = 1e-4


class CheckError(RuntimeError):
    """Проверку выполнить нельзя: нет входного файла или пары не строятся."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="прогоны М1")
    parser.add_argument(
        "--selection",
        type=Path,
        default=РЕЗУЛЬТАТЫ / "klifs_rules_rescore_molecules.csv",
        help="помолекулярная таблица пересчёта по правилам KLIFS",
    )
    parser.add_argument("--targets-dir", type=Path, default=TARGETS_DIR)
    return parser.parse_args()


def _таблица(имя: str) -> list[dict[str, str]]:
    путь = РЕЗУЛЬТАТЫ / f"{имя}.csv"
    if not путь.is_file():
        raise CheckError(f"нет таблицы {путь.name}")
    with путь.open(encoding="utf-8", newline="") as файл:
        return list(csv.DictReader(файл))


def _позы_после(sdf: Path, пакет: Path, mol_ids: list[str]) -> dict[str, tuple[int, float]]:
    """Ключевая связь и скор каждой уточнённой позы, в порядке строк таблицы.

    Метки молекулы в SDF оптимизации нет, поэтому позы сопоставляются со строками
    таблицы по порядку — и это сопоставление проверяется сверкой скора (`ДОПУСК_СВЕРКИ`),
    а не принимается на веру.
    """
    if not sdf.is_file():
        raise CheckError(f"нет файла поз {sdf.name}")
    карман = load_pocket_rules(пакет)
    эталон = reference_ifp(пакет)
    ключевые = key_hbond_positions(эталон)
    эталон_все = select_types(эталон, KLIFS_INTERACTION_TYPES)

    итог: dict[str, tuple[int, float]] = {}
    with open_sdf(sdf) as supplier:
        позы = [mol for mol in supplier]
    if len(позы) != len(mol_ids):
        raise CheckError(f"{sdf.name}: поз {len(позы)}, строк таблицы {len(mol_ids)}")
    for mol_id, mol in zip(mol_ids, позы, strict=True):
        if mol is None:
            raise CheckError(f"{sdf.name}: поза {mol_id} не прочиталась")
        отпечаток = ifp_from_groups(groups_from_mol(mol), карман)
        итог[mol_id] = (
            hinge_hbond(отпечаток, ключевые),
            tversky(эталон_все, select_types(отпечаток, KLIFS_INTERACTION_TYPES), 1.0, 0.0),
        )
    return итог


def п1_ключевая_связь(
    run_dir: Path, targets_dir: Path
) -> tuple[PairedBinary, float]:
    """Доля молекул с ключевой водородной связью до и после уточнения позы."""
    пакет = target_for_run(run_dir, targets_dir)
    строки = _таблица(ОПЫТ.format(run=run_dir.name))
    mol_ids = [строка["mol_id"] for строка in строки]

    до_по_id = {m.mol_id: m for m in rescore_run(run_dir, пакет).molecules}
    пропущенные = [mol_id for mol_id in mol_ids if mol_id not in до_по_id]
    if пропущенные:
        raise CheckError(f"{run_dir.name}: в прогоне нет молекул {пропущенные[:3]}")
    после = _позы_после(РЕЗУЛЬТАТЫ / f"{ОПЫТ.format(run=run_dir.name)}.sdf", пакет, mol_ids)

    расхождение = max(
        max(
            abs(до_по_id[строка["mol_id"]].score_all7 - float(строка["score_before"])),
            abs(после[строка["mol_id"]][1] - float(строка["score_after"])),
        )
        for строка in строки
    )
    if расхождение > ДОПУСК_СВЕРКИ:
        raise CheckError(
            f"{run_dir.name}: скор расходится с таблицей на {расхождение:.2e} — "
            "позы сопоставлены неверно"
        )
    return (
        mcnemar(
            [до_по_id[mol_id].key_hbond for mol_id in mol_ids],
            [после[mol_id][0] for mol_id in mol_ids],
        ),
        расхождение,
    )


def п15_смещение(run: str) -> tuple[float, float, float]:
    """Медианное смещение позы в опыте и в контроле и уровень значимости различия."""
    опыт = [float(с["rmsd_shift"]) for с in _таблица(ОПЫТ.format(run=run))]
    контроль = [float(с["rmsd_shift"]) for с in _таблица(КОНТРОЛЬ.format(run=run))]
    return median(опыт), median(контроль), mann_whitney_u(опыт, контроль)[1]


def п16_выравнивание_по_размеру(
    молекулы: list[dict[str, str]],
) -> tuple[int, int, tuple[int, int, float], tuple[float, float], PairedBinary]:
    """Сравнение топа с отбракованными при равном числе тяжёлых атомов."""
    ранжированные = rank_molecules(
        [
            ScoredMolecule(
                с["mol_id"],
                float(с["score_all7"]),
                float(с["score_all7"]) / int(с["n_heavy"]),
            )
            for с in молекулы
        ],
        top_fraction=DEFAULT_TOP_FRACTION,
    )
    выбрано = {с["mol_id"] for с in ранжированные if с["selected"]}

    def предметы(величина: str, только_топ: bool) -> list[SizedItem]:
        return [
            SizedItem(с["mol_id"], int(с["n_heavy"]), float(с[величина]))
            for с in молекулы
            if (с["mol_id"] in выбрано) == только_топ
        ]

    подбор = match_by_size(предметы("tanimoto", True), предметы("tanimoto", False))
    ключ = match_by_size(предметы("key_hbond", True), предметы("key_hbond", False))
    if not подбор.pairs:
        raise CheckError("пар одинакового размера не нашлось")
    return (
        len(подбор.pairs),
        подбор.unmatched,
        sign_test(подбор.differences),
        bootstrap_ci(подбор.differences, aggregate="mean"),
        mcnemar(
            [int(низ.value) for _, низ in ключ.pairs],
            [int(верх.value) for верх, _ in ключ.pairs],
        ),
    )


def п17_прирост_и_старт(run: str) -> tuple[float, float]:
    """Связь прироста скора со стартовым уровнем и сам медианный прирост."""
    строки = _таблица(ОПЫТ.format(run=run))
    до = np.array([float(с["score_before"]) for с in строки])
    после = np.array([float(с["score_after"]) for с in строки])
    return spearman_rho(до, после - до), float(median(после - до))


def main() -> None:
    args = parse_args()
    строки: list[Строка] = []
    семейства: dict[str, dict[str, float]] = {}

    print("=== Ключевая связь: доля молекул с водородной связью ===")
    всего_появилось = всего_исчезло = 0
    for run_dir in args.runs:
        итог, расхождение = п1_ключевая_связь(run_dir, args.targets_dir)
        всего_появилось += итог.gained
        всего_исчезло += итог.lost
        префикс = f"{run_dir.name}: ключевая связь, "
        строки += [
            Строка(префикс + "молекул", итог.n, "молекул"),
            Строка(префикс + "доля до", итог.before, "доля"),
            Строка(префикс + "доля после", итог.after, "доля"),
            Строка(префикс + "появилась", итог.gained, "молекул"),
            Строка(префикс + "исчезла", итог.lost, "молекул"),
            Строка(префикс + "p Макнемара", итог.p_value, "p"),
            Строка(префикс + "разность долей, нижняя граница", итог.ci[0], "доля"),
            Строка(префикс + "разность долей, верхняя граница", итог.ci[1], "доля"),
            Строка(префикс + "расхождение скора с таблицей", расхождение, "скор"),
        ]
        семейства.setdefault("5.5 ключевая связь", {})[
            f"{run_dir.name}: ключевая связь"
        ] = итог.p_value
        print(
            f"{run_dir.name}: {итог.before:.3f} -> {итог.after:.3f} "
            f"(+{итог.gained} / -{итог.lost}), p = {итог.p_value:.2e}, "
            f"ДИ [{итог.ci[0]:+.3f}; {итог.ci[1]:+.3f}]"
        )

    # Объединение по прогонам даёт самое сильное одно число, но это отдельное
    # утверждение — «по всем прогонам разом», — и приводить его надо вместе
    # с помолекулярными долями, а не вместо них.
    p_объединённый = exact_binomial_p(всего_появилось, всего_появилось + всего_исчезло)
    строки += [
        Строка("ключевая связь, появилась по четырём прогонам", всего_появилось, "молекул"),
        Строка("ключевая связь, исчезла по четырём прогонам", всего_исчезло, "молекул"),
        Строка("ключевая связь, p по четырём прогонам", p_объединённый, "p"),
    ]
    print(
        f"по четырём прогонам: появилась у {всего_появилось}, исчезла у {всего_исчезло}, "
        f"p = {p_объединённый:.2e}"
    )

    print("\n=== Смещение положения в опыте и в контроле ===")
    for run_dir in args.runs:
        опыт, контроль, p = п15_смещение(run_dir.name)
        префикс = f"{run_dir.name}: смещение позы, "
        строки += [
            Строка(префикс + "медиана в опыте", опыт, "ангстрем"),
            Строка(префикс + "медиана в контроле", контроль, "ангстрем"),
            Строка(префикс + "p опыт против контроля", p, "p"),
        ]
        семейства.setdefault("5.5 смещение позы", {})[
            f"{run_dir.name}: смещение"
        ] = p
        print(f"{run_dir.name}: опыт {опыт:.3f} A, контроль {контроль:.3f} A, p = {p:.3f}")

    print("\n=== Отбор при выравнивании по числу тяжёлых атомов ===")
    все_молекулы = list(csv.DictReader(args.selection.open(encoding="utf-8", newline="")))
    прогоны = sorted({с["run_id"] for с in все_молекулы})
    for прогон in прогоны:
        пар, без_пары, (плюс, минус, p_знак), ди, ключ = п16_выравнивание_по_размеру(
            [с for с in все_молекулы if с["run_id"] == прогон]
        )
        префикс = f"{прогон}: выравнивание по размеру, "
        строки += [
            Строка(префикс + "пар", пар, "пар"),
            Строка(префикс + "топ без пары", без_пары, "молекул"),
            Строка(префикс + "Танимото выше у топа", плюс, "пар"),
            Строка(префикс + "Танимото ниже у топа", минус, "пар"),
            Строка(префикс + "p знакового критерия по Танимото", p_знак, "p"),
            Строка(префикс + "разность Танимото, нижняя граница", ди[0], "Танимото"),
            Строка(префикс + "разность Танимото, верхняя граница", ди[1], "Танимото"),
            Строка(префикс + "ключевая связь, доля у пар", ключ.before, "доля"),
            Строка(префикс + "ключевая связь, доля у топа", ключ.after, "доля"),
            Строка(префикс + "p Макнемара по ключевой связи", ключ.p_value, "p"),
        ]
        семейства.setdefault("5.4 отбор", {}).update(
            {
                f"{прогон}: Танимото при равном размере": p_знак,
                f"{прогон}: ключевая связь при равном размере": ключ.p_value,
            }
        )
        print(
            f"{прогон}: пар {пар} (без пары {без_пары}); Танимото выше у топа "
            f"в {плюс} из {плюс + минус}, p = {p_знак:.2e}, ДИ [{ди[0]:+.4f}; {ди[1]:+.4f}]; "
            f"ключевая связь {ключ.before:.3f} -> {ключ.after:.3f}, p = {ключ.p_value:.2e}"
        )

    print("\n=== Прирост против стартового скора ===")
    for run_dir in args.runs:
        ро, прирост = п17_прирост_и_старт(run_dir.name)
        строки += [
            Строка(f"{run_dir.name}: Спирмен прироста со стартовым скором", ро, "ро"),
            Строка(f"{run_dir.name}: медианный прирост скора", прирост, "скор"),
        ]
        print(f"{run_dir.name}: ро = {ро:+.3f}, медианный прирост {прирост:+.4f}")

    print("\n=== Поправка Холма внутри семейств ===")
    for семейство, значения in sorted(семейства.items()):
        исправленные = holm(значения)
        порог = sum(1 for значение in исправленные.values() if значение < 0.05)
        строки.append(
            Строка(f"{семейство}: критериев в семействе", len(значения), "критериев")
        )
        строки.append(
            Строка(f"{семейство}: значимо после поправки Холма", порог, "критериев")
        )
        print(f"{семейство}: {порог} из {len(значения)} значимы после поправки")
        for имя, значение in исправленные.items():
            строки.append(Строка(f"{имя}, p по Холму", значение, "p"))
            print(f"  {имя}: {значения[имя]:.2e} -> {значение:.2e}")

    md, csv_путь = записать(
        строки,
        РЕЗУЛЬТАТЫ,
        имя_md=f"{ИМЯ}.md",
        имя_csv=f"{ИМЯ}.csv",
        заголовок="Усиливающие проверки",
        откуда=КОМАНДА,
    )
    print(f"\nсводка: {md} и {csv_путь}")


if __name__ == "__main__":
    try:
        main()
    except (CheckError, StatsError) as ошибка:
        raise SystemExit(f"проверка не выполнена: {ошибка}") from ошибка
