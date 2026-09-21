"""Режим протонирования белка мишени и его фиксация в пакете мишени.

ProLIF определяет донора и акцептора водородной связи по положению самого атома
водорода. Нет водородов — водородные связи не находятся в принципе, и от отпечатка
остаются одни ван-дер-ваальсовы касания; именно так сломался черновик
(`source_files/мяу.py:2369-2371`), и дефект дожил до конца, потому что код не падал.
Поэтому режим не подразумевается, а записывается полем `protonation` в `target.json`:
любой прогон сообщает, в каком состоянии был белок.

Достраивать водороды не потребовалось: KLIFS отдаёт белок уже протонированным (см.
`KLIFS_PROTONATION_TOOL`). Задача модуля — это проверить, зафиксировать в данных
и приготовить вход для второго режима, чтобы выбор между режимами делало измерение
измерением, а не суждением.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from kinase_ifp.klifs import KlifsDataError, canonical_target_json, validate_target_json
from kinase_ifp.structure_io import PROTEIN_PDB, count_hydrogens, rebuild_pdb_files

# Чем протонирован белок. KLIFS обрабатывает структуры в MOE и по Chemical Component
# Dictionary исправляет типы атомов и связей именно ради «proper protonation of the
# molecules and accurate analysis of protein-ligand interactions, including H-bond
# interaction networks» (Kooistra et al., Nucleic Acids Research 44:D365, 2016,
# doi:10.1093/nar/gkv1082, раздел «Data collection and preparation»).
#
# Версию MOE база не публикует, поэтому в поле её нет: воспроизводимость обеспечивают
# `klifs_structure_id` и дата выгрузки, уже лежащие в пакете. Записывать сюда выдуманный
# номер версии было бы хуже, чем не записывать ничего.
KLIFS_PROTONATION_TOOL: Final[str] = (
    "KLIFS/MOE (Chemical Computing Group): водороды из конвейера подготовки KLIFS, "
    "версия MOE базой не публикуется"
)

PROTONATION_EXPLICIT: Final[str] = "explicit"
PROTONATION_IMPLICIT: Final[str] = "implicit-prolif"

# Что делать со структурой, у которой водородов нет вовсе.
ON_MISSING_H_FAIL: Final[str] = "fail"
ON_MISSING_H_IMPLICIT: Final[str] = "implicit"
ON_MISSING_HYDROGENS: Final[frozenset[str]] = frozenset({ON_MISSING_H_FAIL, ON_MISSING_H_IMPLICIT})


def annotate_protonation(
    target_json: Path, *, on_missing_hydrogens: str = ON_MISSING_H_FAIL
) -> dict[str, Any]:
    """Фиксирует режим протонирования в пакете мишени и перестраивает файлы структур.

    Принимает путь к `target.json`, возвращает обновлённый пакет. Побочный эффект:
    перезаписывает `protein.pdb` и `pocket.pdb` из лежащих рядом mol2 с восстановленной
    идентичностью остатка, создаёт `protein_noh.pdb` и правит сам `target.json`.

    `on_missing_hydrogens` решает судьбу структуры без единого водорода:

    - `'fail'` (по умолчанию) — остановиться. Годится, когда мишень обязана прийти
      протонированной, и её отсутствие водородов означает поломку выгрузки;
    - `'implicit'` — записать режим `implicit-prolif`. Не «водороды достроены»,
      а «работаем без них по правилам ProLIF»: типы взаимодействий берутся из
      `KLIFS_TO_PROLIF_IMPLICIT_H`. Выбор для `select_target.py` — терять
      структуру целиком хуже, чем считать её слабее.

    Своих водородов функция не достраивает ни в каком режиме: в окружении
    нет инструмента, который делает это корректно, а RDKit расставил бы их по валентности
    — с протонированными Asp и Glu, произвольным таутомером His и произвольной
    ориентацией OH, то есть с правдоподобным и неверным отпечатком. Цена режима
    без водородов измерена сверкой: Танимото по типам скора
    0.400 против 1.000 у `explicit`.
    """
    if on_missing_hydrogens not in ON_MISSING_HYDROGENS:
        raise ValueError(
            f"on_missing_hydrogens={on_missing_hydrogens!r}, "
            f"допустимы {sorted(ON_MISSING_HYDROGENS)}"
        )

    if not target_json.is_file():
        raise KlifsDataError(f"Пакет мишени не найден: {target_json}")

    package: dict[str, Any] = json.loads(target_json.read_text(encoding="utf-8"))
    target_dir = target_json.parent

    chain = str(package.get("chain", "")).strip()
    if not chain:
        raise KlifsDataError(
            f"{target_json}: поле chain пусто, восстановить нотацию остатков ProLIF нельзя"
        )

    package.update(rebuild_pdb_files(target_dir, chain))

    hydrogens = count_hydrogens(target_dir / PROTEIN_PDB)
    if hydrogens == 0 and on_missing_hydrogens == ON_MISSING_H_FAIL:
        raise KlifsDataError(
            f"{target_dir / PROTEIN_PDB}: атомов водорода 0, режим {PROTONATION_EXPLICIT!r} "
            f"поставить нельзя. Переход на 'implicit-prolif' задаётся явно, "
            f"а не этим кодом"
        )

    if hydrogens == 0:
        # Инструмент не назван намеренно: водороды никто не достраивал, их просто нет.
        # По формату пакета мишени `protonation_tool` при этом режиме и обязан быть null.
        package["protonation"] = PROTONATION_IMPLICIT
        package["protonation_tool"] = None
    else:
        package["protonation"] = PROTONATION_EXPLICIT
        package["protonation_tool"] = KLIFS_PROTONATION_TOOL

    # Порядок ключей приводится к каноническому перед записью: `package.update`
    # выше дописывает новые ключи в конец, а хэш пакета считается по байтам файла.
    target_json.write_text(
        json.dumps(canonical_target_json(package), indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    validate_target_json(target_json)
    return package
