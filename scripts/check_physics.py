"""Физичность поз по PoseBusters — стандартный набор проверок.

    docker compose run --rm dev python scripts/check_physics.py --sdf <файл> [...]
    docker compose run --rm dev python scripts/check_physics.py --sdf <файл> \
        --target data/targets/6tgu/target.json

Зачем нужен отдельный вход. Порог «ближе 2.0 Å — наложение», которым пользуется
оптимизация позы, назначен нами и одинаков для всех пар атомов. PoseBusters
(Buttenschoen et al., Chem Sci 2024) вместо этого берёт отношение расстояния к сумме
ван-дер-ваальсовых радиусов пары с порогом 0.75 и отдельно считает перекрытие объёмов
с порогом 7.5 %. Порог 0.75 калиброван так, что почти все кристаллические структуры
его проходят, поэтому проверка сравнима с опубликованными работами, а не только
с нашими прежними числами.

Считаются только тесты, применимые без эталонной молекулы и кофакторов: остальные
дают ложные отказы, и на нашем же кристаллическом лиганде это видно сразу — с полным
набором он «не проходит» ровно потому, что эталона для него не передано.

**С ключом `--target` доли попадают в реестр измерений** (вид `physics`, проверка
). Без него скрипт печатает числа и ничего не сохраняет — ровно
это и стоило подраздела текста 18.09: числа были верные, а машинного места у них
не было, и сверка `scripts/check_numbers.py` их не подтверждала. Ключ сделан
необязательным, чтобы прежние вызовы продолжали работать.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from posebusters import PoseBusters

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.measurements import (  # noqa: E402
    Measurement,
    upsert_measurements,
    variant_from_poses,
)
from kinase_ifp.config import MEASUREMENTS_CSV  # noqa: E402
from kinase_ifp.molecule_io import read_mol  # noqa: E402
from kinase_ifp.protonate import prepare_ligand  # noqa: E402

# Атомный номер углерода: радиус берётся из таблицы RDKit по номеру, а не по имени.
УГЛЕРОД: Final[int] = 6

# Метка варианта для условий измерения, а не для набора молекул. Отдельная от мишеней
# и наборов: порог инструмента общий для всех, но записывается при каждом расчёте,
# чтобы смена калибровки PoseBusters была видна в реестре, а не только в версии пакета.
ПОРОГИ: Final[str] = "posebusters-dock"

# Тесты PoseBusters, которым довольно позы и белка. Первые восемь — свойства самой
# молекулы (при твёрдотельном движении не меняются вовсе), последние два — контакт
# с белком, и меняются только они.
ТЕСТЫ: tuple[str, ...] = (
    "sanitization",
    "all_atoms_connected",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "double_bond_flatness",
    "internal_energy",
    "minimum_distance_to_protein",
    "volume_overlap_with_protein",
)

# Величина «прошли все применимые тесты» — та, что идёт в текст как доля физичных поз.
ИТОГ: str = "valid_all"

# Условие для кристаллического лиганда мишени. Ключ реестра — четвёрка с `variant`,
# и лиганд не принадлежит никакому прогону, поэтому у него своё значение: это верхняя
# планка, с которой сравниваются доли порождённых молекул.
КРИСТАЛЛ: str = "crystal"


def доли_по_тестам(итог: object, есть: list[str]) -> dict[str, float]:
    """Доля молекул, прошедших каждый тест, плюс доля прошедших все применимые."""
    доли = {тест: float(итог[тест].mean()) for тест in есть}  # type: ignore[index]
    доли[ИТОГ] = float(итог[есть].all(axis=1).mean())  # type: ignore[index]
    return доли


def порог_наложения(buster: PoseBusters, мишень: str, источник: str) -> list[Measurement]:
    """Порог, ниже которого PoseBusters считает контакт наложением, для пары углеродов.

    Считается, а не переписывается. `clash_cutoff` и `radius_scale` берутся из набора
    `dock` самого инструмента, радиус углерода — из таблицы RDKit; число выходит как
    их произведение на сумму радиусов. «Материалы и методы» называют этот порог прозой
    («при пороге наложения 2.55 Å»), и переписанная константа разошлась бы с инструментом
    молча, если тот сменит калибровку. Пара углеродов взята потому, что о ней говорит
    текст: это самый частый контакт лиганда с белком.

    Порог пишется рядом с долями, а не отдельно, по той же причине, по какой в паспорт
    прогона пишут версии пакетов: он часть условий измерения, а не свойство мишени.
    """
    from rdkit.Chem import GetPeriodicTable

    модуль = next(
        м
        for м in buster.config["modules"]
        if м["function"] == "intermolecular_distance" and "protein" in м["name"].lower()
    )
    параметры = модуль["parameters"]
    радиус = GetPeriodicTable().GetRvdw(УГЛЕРОД)
    порог = параметры["clash_cutoff"] * параметры["radius_scale"] * 2 * радиус
    return [
        Measurement(
            target=мишень,
            measurement="physics",
            variant=ПОРОГИ,
            metric=имя,
            value=f"{значение:.4f}",
            unit=единица,
            measured_utc=datetime.now(timezone.utc).date().isoformat(),
            source=источник,
            notes="порог наложения для пары углеродов: clash_cutoff × radius_scale × 2r",
        )
        for имя, значение, единица in (
            ("clash_threshold_carbon", порог, "A"),
            ("clash_cutoff", float(параметры["clash_cutoff"]), ""),
            ("vdw_radius_carbon", float(радиус), "A"),
        )
    ]


def измерения(
    мишень: str, вариант: str, доли: dict[str, float], молекул: int, источник: str
) -> list[Measurement]:
    """Строки реестра по одному файлу поз: доля на тест плюс число молекул."""
    дата = datetime.now(timezone.utc).date().isoformat()
    строки = [
        Measurement(
            target=мишень,
            measurement="physics",
            variant=вариант,
            metric=имя,
            value=f"{доля:.3f}",
            unit="доля",
            measured_utc=дата,
            source=источник,
        )
        for имя, доля in доли.items()
    ]
    строки.append(
        Measurement(
            target=мишень,
            measurement="physics",
            variant=вариант,
            metric="molecules",
            value=str(молекул),
            unit="",
            measured_utc=дата,
            source=источник,
        )
    )
    return строки


def main() -> int:
    разбор = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Список поз необязателен: с одним `--target` скрипт записывает только условия
    # измерения (порог наложения инструмента), не считая ничего по молекулам. Это
    # дёшево и нужно затем, чтобы порог попал в реестр без часового прогона PoseBusters.
    разбор.add_argument("--sdf", type=Path, nargs="+", default=[], help="файлы поз")
    разбор.add_argument("--protein", type=Path, help="белок мишени; по умолчанию из --target")
    разбор.add_argument(
        "--target",
        type=Path,
        help="target.json пакета мишени: включает запись долей в реестр измерений",
    )
    разбор.add_argument("--measurements", type=Path, default=MEASUREMENTS_CSV)
    разбор.add_argument(
        "--with-crystal",
        action="store_true",
        help="добавить строку кристаллического лиганда мишени как верхнюю планку",
    )
    аргументы = разбор.parse_args()
    if not аргументы.sdf and аргументы.target is None:
        разбор.error("нужен хотя бы один из ключей: --sdf (что считать) или --target (условия)")

    белок = аргументы.protein
    мишень = ""
    if аргументы.target is not None:
        мишень = str(json.loads(аргументы.target.read_text(encoding="utf-8"))["pdb_id"])
        белок = белок or аргументы.target.parent / "protein.pdb"
    if белок is None:
        print("нужен --protein или --target", file=sys.stderr)
        return 1

    buster = PoseBusters(config="dock")
    накопленные: list[Measurement] = []

    if аргументы.with_crystal:
        if аргументы.target is None:
            print("--with-crystal требует --target", file=sys.stderr)
            return 1
        # Кристаллический лиганд лежит в пакете без явных водородов; PoseBusters
        # проверяет геометрию, и достройка водородов её не меняет, но без подготовки
        # часть тестов не считается.
        лиганд = prepare_ligand(read_mol(аргументы.target.parent / "ligand.sdf"))
        временный = аргументы.target.parent / "_crystal_posebusters.sdf"
        try:
            from rdkit import Chem

            with временный.open("w", encoding="utf-8", newline="") as поток:
                писатель = Chem.SDWriter(поток)
                писатель.write(лиганд)
                писатель.close()
            итог = buster.bust([временный], None, белок, full_report=True)
            есть = [тест for тест in ТЕСТЫ if тест in итог.columns]
            доли = доли_по_тестам(итог, есть)
            print(f"\n=== кристаллический лиганд {мишень} ===")
            print(f"прошли все применимые тесты: {доли[ИТОГ]:.3f}")
            накопленные += измерения(
                мишень,
                КРИСТАЛЛ,
                доли,
                len(итог),
                f"python scripts/check_physics.py --target {аргументы.target} --with-crystal",
            )
        finally:
            временный.unlink(missing_ok=True)

    for путь in аргументы.sdf:
        итог = buster.bust([путь], None, белок, full_report=True)
        есть = [тест for тест in ТЕСТЫ if тест in итог.columns]
        доли = доли_по_тестам(итог, есть)
        print(f"\n=== {путь.name}: молекул {len(итог)} ===")
        print(f"прошли все применимые тесты: {доли[ИТОГ]:.3f}")
        for тест in есть:
            отметка = "  " if доли[тест] == 1.0 else "* "
            print(f"{отметка}{тест}: {доли[тест]:.3f}")
        if мишень:
            накопленные += измерения(
                мишень,
                variant_from_poses(путь),
                доли,
                len(итог),
                f"python scripts/check_physics.py --sdf {путь} --target {аргументы.target}",
            )

    if мишень:
        накопленные += порог_наложения(
            buster, мишень, f"python scripts/check_physics.py --target {аргументы.target}"
        )

    if накопленные:
        upsert_measurements(накопленные, аргументы.measurements)
        print(f"\nв реестр измерений записано строк: {len(накопленные)}")
    elif аргументы.target is None:
        print("\n--target не задан: доли напечатаны, но в реестр не записаны")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
