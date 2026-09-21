"""Тесты сверки прогона с пакетом мишени и штампа задним числом.

Главное, что здесь проверяется, — **штамп не ставится без основания**. Функция
`runs.stamp_target` сама по себе доверчива: она пишет в паспорт то, что ей сказали.
Основание даёт только пересчёт, и если он разошёлся или не состоялся, паспорт обязан
остаться нетронутым — иначе прогон получает утверждение, которого никто не проверял.

Прогон берётся настоящий: `build_local_pose_run` строит набор поз по той же фикстуре
6tgu, что и остальные тесты, так что сверка идёт по живым числам, а не по подделанным
CSV. Подделанный файл проверил бы только сравнение строк, а не то, что пересчёт
воспроизводит прогон.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.local_poses import build_local_pose_run
from experiments.metrics import compute_run_metrics
from experiments.ranking import score_run
from experiments.restamp import VERIFIED_FILES, verify_and_stamp
from experiments.runs import RunError, compare_target_stamp, read_passport


@pytest.fixture(scope="module")
def прогон(tmp_path_factory: pytest.TempPathFactory, target_json: Path) -> Path:
    """Посчитанный набор поз: молекулы, метрики и ранжирование на месте."""
    runs = tmp_path_factory.mktemp("runs")
    папка = build_local_pose_run(target_json, n=4, seed=3, runs_dir=runs, day="2026-08-24")
    compute_run_metrics(папка, target_json)
    score_run(папка, target_json)
    return папка


def копия_прогона(прогон: Path, куда: Path) -> Path:
    папка = куда / прогон.name
    shutil.copytree(прогон, папка)
    return папка


def снять_штамп(run_dir: Path) -> None:
    """Возвращает паспорт в состояние «прогон сделан до формата паспорта прогона»."""
    путь = run_dir / "run.json"
    паспорт = json.loads(путь.read_text(encoding="utf-8"))
    паспорт.get("target", {}).pop("protonation", None)
    паспорт.get("target", {}).pop("package_sha256", None)
    паспорт.pop("scoring", None)
    путь.write_text(json.dumps(паспорт, indent=2, ensure_ascii=False), encoding="utf-8")


def чужой_пакет(target_json: Path, куда: Path) -> Path:
    """Пакет с другим эталонным отпечатком: карман валиден, а числа станут другими.

    Портить `residue_to_position` бесполезно: `validate_target_json` ловит это раньше,
    и до сравнения чисел дело не доходит.
    """
    пакет = куда / "чужой"
    shutil.copytree(target_json.parent, пакет)
    путь = пакет / target_json.name
    данные = json.loads(путь.read_text(encoding="utf-8"))
    данные["klifs_ifp_bits"] = "1" * 20 + str(данные["klifs_ifp_bits"])[20:]
    путь.write_text(json.dumps(данные, indent=2, ensure_ascii=False), encoding="utf-8")
    return путь


def test_совпадение_ставит_штамп(прогон: Path, target_json: Path, tmp_path: Path) -> None:
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)

    отчёт = verify_and_stamp(папка, target_json)

    assert отчёт.stamped
    assert отчёт.verified == VERIFIED_FILES
    assert compare_target_stamp(read_passport(папка), target_json)[0] == "match"


def test_сверка_не_трогает_файлы_прогона(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Пересчёт идёт в копии: в самом прогоне меняется только паспорт."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    было = {имя: (папка / имя).read_bytes() for имя in VERIFIED_FILES}

    verify_and_stamp(папка, target_json)

    assert {имя: (папка / имя).read_bytes() for имя in VERIFIED_FILES} == было


def test_уже_заштампованный_проходит_молча(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Повтор допустим: цепочку прогона должно быть можно запустить дважды."""
    папка = копия_прогона(прогон, tmp_path)

    отчёт = verify_and_stamp(папка, target_json)

    assert not отчёт.stamped
    assert отчёт.protonation


def test_расхождение_чисел_отменяет_штамп(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Ради этого случая функция и написана: пакет другой, а сказать это некому."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    чужой = чужой_пакет(target_json, tmp_path)

    with pytest.raises(RunError, match="Пересчёт разошёлся"):
        verify_and_stamp(папка, чужой)

    assert "package_sha256" not in read_passport(папка).get("target", {})


def test_расхождение_длиной_отменяет_штамп(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Вторая ветка расхождения — разное число строк, а не разные числа в строке.

    Прежде исполнялась только первая ветка. Случай
    достижим без подделки чисел — лишняя строка в таблице метрик появляется
    от повторного расчёта, и пересчёт даст на строку
    меньше при полном совпадении всех остальных.
    """
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    метрики = папка / "metrics_per_molecule.csv"
    строки = метрики.read_text(encoding="utf-8").splitlines()
    # Дубль последней строки: первые N строк совпадут с пересчётом посимвольно,
    # и разойдётся только длина — иначе сработала бы ветка «строка и столбец».
    метрики.write_text("\n".join([*строки, строки[-1]]) + "\n", encoding="utf-8")

    with pytest.raises(RunError, match="длиной"):
        verify_and_stamp(папка, target_json)

    assert "package_sha256" not in read_passport(папка).get("target", {})


def test_сообщение_называет_место_расхождения(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Начало строки в метриках одинаково у обеих сторон — показывать надо не его."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    чужой = чужой_пакет(target_json, tmp_path)

    with pytest.raises(RunError, match=r"строка \d+, столбец \d+"):
        verify_and_stamp(папка, чужой)


def test_прогон_без_метрик_и_ранжирования_отвергается(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """Сверять нечем — значит штамп был бы записью непроверенного утверждения."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    for имя in VERIFIED_FILES:
        (папка / имя).unlink()

    with pytest.raises(RunError, match="нечем сверить"):
        verify_and_stamp(папка, target_json)


def test_одного_ranking_достаточно(прогон: Path, target_json: Path, tmp_path: Path) -> None:
    """У прогона без метрик — сверяется то, что есть."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    (папка / "metrics_per_molecule.csv").unlink()

    отчёт = verify_and_stamp(папка, target_json)

    assert отчёт.verified == ("ranking.csv",)
    assert отчёт.stamped
    assert not (папка / "metrics_per_molecule.csv").exists()


def test_несобираемый_пакет_даёт_понятный_отказ(
    прогон: Path, target_json: Path, tmp_path: Path
) -> None:
    """«Сверить не удалось» и «сверка не сошлась» — разные ответы, и путать их нельзя."""
    папка = копия_прогона(прогон, tmp_path)
    снять_штамп(папка)
    сломанный = tmp_path / "сломанный"
    shutil.copytree(target_json.parent, сломанный)
    (сломанный / "protein.pdb").unlink()

    with pytest.raises(RunError, match="не пересчитывается|хэш пакета неполон"):
        verify_and_stamp(папка, сломанный / target_json.name)
