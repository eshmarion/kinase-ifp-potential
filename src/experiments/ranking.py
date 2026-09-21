"""Переранжирование прогона по IFP-скору: `ranking.csv`.

Здесь только работа с прогоном — чтение молекул, запись файла, отметка параметров
отбора в паспорте. Сам скор живёт в `kinase_ifp.scoring` и ни от чего, кроме двух
отпечатков, не зависит: на уровне B он станет потенциалом, и тянуть за собой чтение
папок ему незачем.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Final

from experiments.layout import CSV_EOL, MOLECULES_SDF, RANKING_CSV, RUN_JSON
from experiments.run_io import append_failures
from experiments.runs import RunError, ensure_target_matches
from kinase_ifp.fingerprint import IfpError, compute_ifp
from kinase_ifp.molecule_io import open_sdf
from kinase_ifp.pocket import load_pocket
from kinase_ifp.scoring import (
    DEFAULT_TOP_FRACTION,
    ScoredMolecule,
    ScoringError,
    ifp_score,
    rank_molecules,
    score_norm,
)

RANKING_COLUMNS: Final[tuple[str, ...]] = (
    "mol_id",
    "ifp_score",
    "ifp_score_norm",
    "rank",
    "selected",
)

# Этап, под которым в `failures.csv` записываются молекулы, у которых не посчитался
# отпечаток: производитель их принял, а скор — нет, и это разные стадии.
SCORING_STAGE: Final[str] = "scoring"


def score_run(
    run_dir: Path,
    target_json: Path,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> Path:
    """Считает скор для всех молекул прогона и пишет `ranking.csv`. Возвращает путь к нему.

    Молекулу, для которой отпечаток не считается, функция не пропускает молча: причина
    уходит в `failures.csv` со стадией `scoring`, а расчёт продолжается. Уронить прогон
    из-за одной молекулы так же неверно, как потерять её без следа.

    Поднимает `ScoringError`, если у мишени нет эталонного отпечатка или если ни одна
    молекула не посчиталась — второе означает поломку, а не свойство данных.
    """
    pocket = load_pocket(target_json)
    if pocket.reference_ifp is None:
        raise ScoringError(
            f"{target_json}: у мишени нет эталонного отпечатка KLIFS, скор считать не с чем"
        )

    sdf = run_dir / MOLECULES_SDF
    if not sdf.is_file():
        raise ScoringError(f"В прогоне нет {MOLECULES_SDF}: {run_dir}")

    # После проверки состава папки, но до расчёта: считать по чужому пакету нельзя,
    # а сообщать об этом раньше, чем о пустой папке, значит путать причину со следствием.
    try:
        ensure_target_matches(run_dir, target_json)
    except RunError as ошибка:
        raise ScoringError(str(ошибка)) from ошибка

    посчитанные: list[ScoredMolecule] = []
    отказы: list[tuple[str, str, str]] = []

    # Поток, а не путь строкой: путь RDKit передаёт в C++ через ANSI и на Windows
    # не открывает ничего с кириллицей — обёртка и разбор в `kinase_ifp.molecule_io`.
    # sanitize=False здесь не годится: скор считается по отпечатку, а тот требует
    # разобранной молекулы. removeHs=False обязателен — водороды поставил производитель.
    with open_sdf(sdf) as supplier:
        for номер, mol in enumerate(supplier):
            mol_id = f"{run_dir.name}-{номер:04d}"
            if mol is None:
                отказы.append((mol_id, SCORING_STAGE, "RDKit не разобрал запись SDF"))
                continue
            if mol.HasProp("mol_id"):
                mol_id = mol.GetProp("mol_id")

            try:
                отпечаток = compute_ifp(pocket, mol)
                значение = ifp_score(отпечаток, pocket.reference_ifp)
                посчитанные.append(
                    ScoredMolecule(
                        mol_id=mol_id,
                        ifp_score=значение,
                        ifp_score_norm=score_norm(значение, mol.GetNumHeavyAtoms()),
                    )
                )
            except (IfpError, ScoringError) as ошибка:
                отказы.append((mol_id, SCORING_STAGE, str(ошибка)))

    if отказы:
        append_failures(run_dir, отказы)
    if not посчитанные:
        raise ScoringError(
            f"В прогоне {run_dir.name} не посчиталась ни одна молекула из "
            f"{len(посчитанные) + len(отказы)}: причины в failures.csv"
        )

    строки = rank_molecules(посчитанные, top_fraction=top_fraction)
    путь = write_ranking(run_dir, строки)
    _record_selection(run_dir, top_fraction=top_fraction, selected=sum(
        int(строка["selected"]) for строка in строки
    ))
    return путь


def write_ranking(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Пишет `ranking.csv` по формату таблицы ранжирования и возвращает путь к нему."""
    путь = run_dir / RANKING_CSV
    with путь.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RANKING_COLUMNS, lineterminator=CSV_EOL)
        writer.writeheader()
        writer.writerows(rows)
    return путь


def read_ranking(run_dir: Path) -> list[dict[str, Any]]:
    """Читает `ranking.csv` прогона."""
    путь = run_dir / RANKING_CSV
    if not путь.is_file():
        raise ScoringError(f"В прогоне нет {RANKING_CSV}: {run_dir}")
    with путь.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _record_selection(run_dir: Path, *, top_fraction: float, selected: int) -> None:
    """Дописывает параметры отбора в `run.json`: размер топа живёт там.

    Единственное место, где переранжирование трогает уже записанный паспорт: добавляется
    ключ `ranking`, ничего другого не меняется.

    Повторный запуск с другой долей топа разрешён намеренно — на нём стоит перебор долей,
    которому нужна кривая по нескольким размерам топа для одного и того же прогона.
    Потери данных здесь нет: `ifp_score` и `rank` от доли не зависят вовсе, меняется
    только колонка `selected`, и пересчёт воспроизводим по тем же входным файлам.
    Запрет №2 защищает сами прогоны — молекулы и паспорт, — а не запрещает считать заново.
    """
    паспорт_путь = run_dir / RUN_JSON
    if not паспорт_путь.is_file():
        raise ScoringError(f"В прогоне нет {RUN_JSON}: {run_dir}")

    паспорт = json.loads(паспорт_путь.read_text(encoding="utf-8"))
    паспорт["ranking"] = {"top_fraction": top_fraction, "n_selected": selected}
    паспорт_путь.write_text(json.dumps(паспорт, indent=2, ensure_ascii=False), encoding="utf-8")
