"""CLI: выгрузка прогона из тома Modal в `runs/`.

Запуск — из контейнера через сервис `cloud`, у которого примонтирован токен:

    docker compose run --rm cloud python scripts/pull_run.py --run-id 2026-08-23-6tgu-s0-n5
    docker compose run --rm cloud python scripts/pull_run.py --list

Результаты не должны оставаться только в облаке: том живёт в чужом сервисе, а `runs/`
лежит на локальном диске. Повторная выгрузка отклоняется — перезапись прогона запрещена
, новый запуск даёт новый `run_id`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modal_targets import RUNS_VOLUME_NAME, runs  # noqa: E402

from experiments.runs import RunError, check_pulled, ensure_absent  # noqa: E402
from kinase_ifp.config import RUNS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", help="какой прогон забрать из тома")
    parser.add_argument(
        "--list",
        action="store_true",
        help="показать прогоны, лежащие в томе, и выйти",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="куда выгружать (по умолчанию runs/ в корне репозитория)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        записи = runs.listdir("/")
        if not записи:
            raise SystemExit(f"Том {RUNS_VOLUME_NAME} пуст: сначала запустите modal_sample.py")
        for запись in sorted(записи, key=lambda запись: запись.path):
            print(запись.path)
        return

    if not args.run_id:
        raise SystemExit("Укажите --run-id или --list")

    назначение = args.runs_dir / args.run_id
    try:
        ensure_absent(назначение)
    except RunError as ошибка:
        raise SystemExit(str(ошибка)) from ошибка

    # Список берётся до создания каталога намеренно: неготовый прогон (сэмплер ещё
    # пишет или был вытеснен) даёт пустой список, и каталога быть не должно вовсе.
    # Прежний порядок «создать → скопировать → проверить» оставлял пустую папку после
    # каждой неудачной попытки, а она закрывала повтор навсегда.
    записи = list(runs.listdir(f"/{args.run_id}", recursive=True))
    if not записи:
        raise SystemExit(
            f"В томе {RUNS_VOLUME_NAME} нет файлов прогона {args.run_id}: прогон ещё "
            "не дописан или не существует. Список готовых — pull_run.py --list"
        )

    # `exist_ok`: пустая папка от прежней неудачной попытки отказа больше не вызывает,
    # и системная ошибка здесь была бы хуже внятного сообщения выше.
    назначение.mkdir(parents=True, exist_ok=True)
    for запись in записи:
        # В томе лежат только файлы прогона; каталоги внутри прогона форматом
        # не предусмотрены, поэтому вложенность не разбирается.
        имя = Path(запись.path).name
        (назначение / имя).write_bytes(b"".join(runs.read_file(запись.path)))
        print(f"{запись.size:>10}  {имя}")

    try:
        паспорт = check_pulled(назначение)
    except RunError as ошибка:
        raise SystemExit(f"{ошибка}\nПапка {назначение} осталась на месте для разбора") from ошибка

    print(f"прогон {паспорт['run_id']} выгружен в {назначение}")
    print(f"источник: {паспорт['source']}, молекул: {паспорт['sampling']['n_returned']}")


if __name__ == "__main__":
    main()
