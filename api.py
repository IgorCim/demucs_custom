# -*- coding: utf-8 -*-
"""
demucs_custom / api.py
======================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    HTTP-интерфейс к нашему разделению музыки. Через него сервис
    общается с внешним миром: сайтом, мобильным приложением, другим сервисом.

    Проще говоря: клиент отправляет POST с аудиофайлом — получает в ответ
    4 ссылки на скачивание дорожек.

ТОЧКИ (endpoints):
    GET  /                 — что за сервис, какая версия
    GET  /health           — жив ли сервер (его дёргают мониторинги)
    POST /separate         — загрузить песню и получить id задачи
    GET  /status/{job_id}  — готова ли задача и что получилось
    GET  /download/{job_id}/{source} — скачать одну дорожку

ЗАПУСК (для разработки):
    pip install -r requirements.txt
    python api.py
    затем открой в браузере http://127.0.0.1:8000/docs
    (FastAPI сам рисует удобную страницу для тестов)

ЗАПУСК (в проде, как положено):
    uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
    ВАЖНО: workers = 1, пока идёт обучение/разделение на CPU.
    Каждый worker — это отдельный процесс со своей копией модели в памяти
    (а модель весит сотни мегабайт), поэтому 8 workers = 8 копий модели
    и быстро закончится память.

БЕЗОПАСНОСТЬ (обязательно прочитай):
    - НИКОГДА не подставляй имя файла от пользователя прямо в путь.
      Иначе через "/download/../../secrets" можно прочитать чужие файлы.
      Здесь имена файлов формируются ТОЛЬКО из безопасного job_id.
    - Токен бота/ключи API хранятся в переменных окружения, а не в коде.
    - Ограничивай размер загружаемого файла (config.max_upload_mb).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config

# FastAPI нужен только для запуска сервера. Если его нет — не падаем
# с невнятным Traceback, а даём понятное сообщение.
try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    FASTAPI_INSTALLED = True
except ImportError:  # pragma: no cover - зависит от окружения
    FASTAPI_INSTALLED = False


# ВРЕМЕННОЕ ХРАНИЛИЩЕ ЗАДАЧ (пока без базы данных и очередей).
# Когда появится настоящая очередь (Celery / Redis) — заменим на неё.
JOBS: Dict[str, Dict[str, Any]] = {}

# Папка, куда кладутся загруженные файлы и результаты.
WORK_DIR: Path = config.log_dir_abs / "jobs"


def _validate_upload(filename: Optional[str], size_bytes: int) -> None:
    """
    Проверяет загруженный файл ДО обработки. Бросает HTTPException(400).

    Проверяем:
      - есть ли имя файла
      - расширение из белого списка config.audio_extensions
      - размер не больше config.max_upload_mb
    """
    # TODO: реализовать проверки
    # Подсказка:
    #   if not filename: raise HTTPException(400, "Не указано имя файла")
    #   ext = Path(filename).suffix.lower()
    #   if ext not in config.audio_extensions:
    #       raise HTTPException(400, f"Неподдерживаемый формат: {ext}")
    #   if size_bytes > config.max_upload_mb * 1024 * 1024:
    #       raise HTTPException(413, "Файл слишком большой")
    raise HTTPException(status_code=501, detail="TODO: _validate_upload() ещё не написан")


def _new_job_id() -> str:
    """
    Генерирует безопасный уникальный идентификатор задачи.

    Почему uuid4, а не имя файла: идентификатор попадает в URL и в файловую
    систему. Случайная строка исключает и подделку, и «выход» за пределы папки.
    """
    return uuid.uuid4().hex


def create_app() -> "FastAPI":
    """
    Создаёт объект FastAPI со всеми маршрутами.
    Вынесено в отдельную функцию, чтобы было видно структуру, а не кашу импортов.
    """
    if not FASTAPI_INSTALLED:
        raise SystemExit(
            "[!] FastAPI не установлен.\n"
            "    Установи зависимости:  pip install -r requirements.txt"
        )

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    config.ensure_directories()

    app = FastAPI(
        title="demucs_custom API",
        description="Разделение музыки на вокал / барабаны / бас / остальное",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # GET / — общая информация
    # ------------------------------------------------------------------
    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Информация о сервисе."""
        return {
            "service": "demucs_custom",
            "version": "0.1.0",
            "model": config.demucs_model,
            "sources": config.sources,
            "device": config.device,
            "sample_rate": config.sample_rate,
            "docs": "/docs",
        }

    # ------------------------------------------------------------------
    # GET /health — проверка живости сервиса
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """
        Проверка, что сервис жив и готов принимать файлы.
        Мониторинг (monitor.py) дёргает этот адрес раз в config.monitor_interval секунд.
        """
        model_exists = config.output_model_path_abs.exists()
        return {
            # "ok" — сервис жив; "degraded" — жив, но модель не загружена
            "status": "ok" if model_exists else "degraded",
            "model_loaded": model_exists,
            "device": config.device,
            "jobs_in_memory": len(JOBS),
        }

    # ------------------------------------------------------------------
    # POST /separate — главный маршрут: загрузка песни
    # ------------------------------------------------------------------
    @app.post("/separate")
    async def separate(
        file: UploadFile = File(..., description="Аудиофайл: mp3 / wav / flac"),
    ) -> Dict[str, Any]:
        """
        Принимает файл, ставит задачу в очередь и возвращает job_id.

        Дальше клиент опрашивает /status/{job_id} и по готовности
        скачивает дорожки через /download/{job_id}/{source}.
        """
        # TODO: реализовать приём файла:
        #   1. Прочитать содержимое: data = await file.read()
        #   2. _validate_upload(file.filename, len(data))
        #   3. job_id = _new_job_id()
        #   4. Сохранить во временную папку: WORK_DIR / job_id / "input.<ext>"
        #   5. Запустить разление в фоне (fastapi.BackgroundTasks или
        #      отдельный поток), чтобы запрос не ждал минуты
        #   6. Вернуть {"job_id": job_id, "status": "queued"}
        raise HTTPException(
            status_code=501, detail="TODO: /separate ещё не реализован"
        )

    # ------------------------------------------------------------------
    # GET /status/{job_id} — статус задачи
    # ------------------------------------------------------------------
    @app.get("/status/{job_id}")
    async def status(job_id: str) -> Dict[str, Any]:
        """Возвращает состояние задачи: queued / running / done / error."""
        # TODO: реализовать:
        #   Проверить, что job_id есть в JOBS (иначе 404).
        #   ВАЖНО: проверять, что job_id состоит только из [0-9a-f] и длиной 32,
        #   иначе через "../../.." можно было бы залезть в чужие файлы.
        #   Вернуть JOBS[job_id]
        raise HTTPException(
            status_code=501, detail="TODO: /status ещё не реализован"
        )

    # ------------------------------------------------------------------
    # GET /download/{job_id}/{source} — скачать дорожку
    # ------------------------------------------------------------------
    @app.get("/download/{job_id}/{source}")
    async def download(job_id: str, source: str) -> Any:
        """Отдаёт готовую дорожку (vocals / drums / bass / other)."""
        # TODO: реализовать:
        #   1. Проверить job_id и source (source строго из config.sources)
        #   2. Собрать путь ВНУТРИ WORK_DIR
        #   3. Вернуть FileResponse(path, media_type="audio/wav", filename=...)
        raise HTTPException(
            status_code=501, detail="TODO: /download ещё не реализован"
        )

    return app


# Атрибут app нужен для команды "uvicorn api:app".
# Если FastAPI не установлен — app будет None, и uvicorn выдаст понятную ошибку.
app = create_app() if FASTAPI_INSTALLED else None


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа: запускает сервер разработки (uvicorn)."""
    print("=" * 70)
    print(" demucs_custom / API")
    print("=" * 70)
    print(f"Адрес        : http://{config.api_host}:{config.api_port}")
    print(f"Документация : http://{config.api_host}:{config.api_port}/docs")
    print(f"Модель       : {config.output_model_path_abs}")
    print(f"Устройство   : {config.device}")

    if not FASTAPI_INSTALLED:
        print()
        print("[!] FastAPI не установлен.")
        print("    Выполни:  pip install -r requirements.txt")
        return 1

    if not config.output_model_path_abs.exists():
        print()
        print(f"[!] Модель не найдена: {config.output_model_path_abs}")
        print("    Сервер запустится, но разделять песни не сможет.")
        print("    Сначала обучи модель: python train_custom.py")

    # TODO: перед долгим запуском проверить, что модель грузится ОДИН раз
    #       на весь процесс (иначе каждый запрос будет ждать загрузку весов).

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print()
        print("[!] uvicorn не установлен. Выполни: pip install -r requirements.txt")
        return 1

    # В проде лучше запускать напрямую:
    #   uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,      # True только при разработке
        workers=1,         # см. пояснение в шапке файла про память
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
