"""Приложение Modal: окружение DiffSBDD в облаке и проверка его работоспособности.

Тонкий вход по правилу проекта: содержательная логика живёт в `src/`, здесь только
описание образа, тома с весами и двух служебных функций.

Порядок запуска (токен Modal задаётся командой `modal setup` и в репозиторий
не попадает):

    modal run scripts/modal_app.py::download_checkpoint   # один раз, веса в Volume
    modal run scripts/modal_app.py::check_env             # проверка окружения на GPU
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSBDD_DIR = PROJECT_ROOT / "external" / "DiffSBDD"

# Чекпойнт по умолчанию: README DiffSBDD называет основным
# для генерации лиганда в заданном кармане условную полноатомную модель.
CHECKPOINT_NAME = "crossdocked_fullatom_cond.ckpt"
CHECKPOINT_URL = f"https://zenodo.org/record/8183747/files/{CHECKPOINT_NAME}?download=1"
CHECKPOINTS_PATH = "/checkpoints"

# Образ описан Dockerfile'ом, а не кодом Modal: тот же файл собирается локально командой
# docker build, поэтому «окружение не собирается» выясняется на своей машине, а не
# за GPU-минуты. Код DiffSBDD не запекается в образ, а монтируется из клона в external/.
image = modal.Image.from_dockerfile(
    str(PROJECT_ROOT / "docker" / "Dockerfile.diffsbdd"),
    context_dir=str(PROJECT_ROOT),
).add_local_dir(str(DIFFSBDD_DIR), remote_path="/opt/DiffSBDD")

# Веса живут в Volume, а не в образе: иначе каждая пересборка образа тянет их заново.
checkpoints = modal.Volume.from_name("diffsbdd-checkpoints", create_if_missing=True)

app = modal.App("kinase-ifp-potential")


@app.function(image=image, volumes={CHECKPOINTS_PATH: checkpoints}, timeout=3600)
def download_checkpoint(force: bool = False) -> dict[str, object]:
    """Кладёт чекпойнт DiffSBDD в Volume и возвращает его паспорт.

    Аргумент `force` перекачивает файл, даже если он уже на месте. Возвращает имя,
    размер и sha256 — контрольная сумма нужна в `run.json`, иначе
    невозможно доказать, что все прогоны сделаны одной и той же моделью.
    """
    import hashlib
    import urllib.request

    целевой = Path(CHECKPOINTS_PATH) / CHECKPOINT_NAME

    if force or not целевой.exists():
        print(f"качаю {CHECKPOINT_URL}")
        urllib.request.urlretrieve(CHECKPOINT_URL, целевой)
        checkpoints.commit()

    сумма = hashlib.sha256(целевой.read_bytes()).hexdigest()
    паспорт: dict[str, object] = {
        "checkpoint": CHECKPOINT_NAME,
        "path": str(целевой),
        "size_bytes": целевой.stat().st_size,
        "sha256": сумма,
    }
    print(паспорт)
    return паспорт


@app.function(image=image, gpu="T4", volumes={CHECKPOINTS_PATH: checkpoints})
def check_env() -> dict[str, object]:
    """Печатает версии критичных библиотек и путь к чекпойнту внутри Volume.

    Пока эта функция не отработала, считается,
    что образ DiffSBDD в Modal не собран.
    """
    import sys

    import torch
    import torch_scatter
    from openbabel import openbabel

    checkpoints.reload()
    файл = Path(CHECKPOINTS_PATH) / CHECKPOINT_NAME

    отчёт: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_scatter": torch_scatter.__version__,
        "openbabel": openbabel.OBReleaseVersion(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint_path": str(файл),
        "checkpoint_exists": файл.exists(),
        "diffsbdd_mounted": Path("/opt/DiffSBDD/generate_ligands.py").exists(),
    }
    for ключ, значение in отчёт.items():
        print(f"{ключ:20} {значение}")
    return отчёт


@app.local_entrypoint()
def main() -> None:
    """Полная проверка: скачать чекпойнт, если его нет, и проверить окружение."""
    download_checkpoint.remote()
    check_env.remote()
