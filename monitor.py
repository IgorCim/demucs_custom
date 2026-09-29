# -*- coding: utf-8 -*-
"""
demucs_custom / monitor.py
==========================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Следит за тем, чтобы проект работал как надо, и записывает всё в логи.

    Три задачи:
      1. ЛОГИ. Пишет понятные записи в файл logs/demucs_custom.log
         и в консоль одновременно. Потом по этому файлу можно понять,
         что случилось вчера в 3 часа ночи.
      2. ЗДОРОВЬЕ. Периодически проверяет: сервис отвечает? Модель на месте?
         Папки существуют? Сколько места осталось на диске?
      3. СТАТИСТИКА. Считает, сколько задач выполнено, сколько упало,
         сколько заняло времени.

ЗАПУСК:
    python monitor.py                 # проверить всё один раз и выйти
    python monitor.py --watch         # проверять постоянно (каждую минуту)
    python monitor.py --watch --interval 30
    python monitor.py --help

ФОРМАТ ЛОГОВ:
    2026-09-26 18:30:15 | INFO  | api        | Модель загружена
    дата и время        уровень  модуль      сообщение
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config

# Имя основного лог-файла проекта.
LOG_FILENAME: str = "demucs_custom.log"

# Формат строки лога: время | уровень | модуль | сообщение
LOG_FORMAT: str = "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# ЛОГИРОВАНИЕ (это работает уже сейчас, можно пользоваться сразу)
# ---------------------------------------------------------------------------

def setup_logging(
    name: str = "demucs_custom",
    level: Optional[str] = None,
    to_file: bool = True,
) -> logging.Logger:
    """
    Настраивает и возвращает логгер.

    name    — имя логгера (обычно имя модуля: __name__)
    level   — уровень: DEBUG / INFO / WARNING / ERROR (по умолчанию config.log_level)
    to_file — писать ли в файл (True) или только в консоль (False)

    Что настраивается:
      - RotatingFileHandler: файл автоматически ротируется, когда достигает
        config.log_max_size_mb, и старые логи не съедают весь диск.
      - StreamHandler: дублирует записи в консоль.
    """
    logger = logging.getLogger(name)
    level_name = (level or config.log_level or "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    # Не вешаем обработчики дважды, если setup_logging() вызвали повторно.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if to_file:
        log_dir = config.log_dir_abs
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / LOG_FILENAME

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=config.log_max_size_mb * 1024 * 1024,
            backupCount=config.log_backup_count,
            encoding="utf-8",   # важно: в логах будет русский текст
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Общий логгер проекта. Импортируй его в других файлах:
#   from monitor import get_logger
#   log = get_logger(__name__)
log = setup_logging(__name__)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Отдаёт настроенный логгер.

    name=None — логгер самого monitor.py.
    В остальных файлах вызывай get_logger(__name__) — у каждого модуля
    будет свой "имя" в колонке лога, так проще искать проблему.
    """
    if not name or name == __name__:
        return log
    return setup_logging(name)


# ---------------------------------------------------------------------------
# ПРОВЕРКИ ЗДОРОВЬЯ
# ---------------------------------------------------------------------------

def check_api_health(api_host: Optional[str] = None,
                     api_port: Optional[int] = None) -> bool:
    """
    Проверяет, отвечает ли API-сервер (GET /health).
    Возвращает True, если сервер жив.
    """
    host = api_host or config.api_host
    port = api_port or config.api_port
    url = f"http://{host}:{port}/health"

    # TODO: реализовать проверку
    # Подсказка (без лишних библиотек):
    #   import urllib.request
    #   try:
    #       with urllib.request.urlopen(url, timeout=5) as r:
    #           return r.status == 200
    #   except Exception:
    #       return False
    return False


def check_disk_space(path: Optional[Path] = None,
                     min_free_gb: float = 1.0) -> bool:
    """
    Проверяет, сколько свободного места на диске с проектом.
    Если места мало (< min_free_gb) — обучение упадёт с непонятной ошибкой.

    Возвращает True, если места достаточно.
    """
    # TODO: реализовать
    # Подсказка:
    #   import shutil
    #   free_gb = shutil.disk_usage(path).free / (1024**3)
    #   return free_gb >= min_free_gb
    return True


def check_model_available() -> bool:
    """Проверяет, что файл модели существует."""
    return config.output_model_path_abs.exists()


def check_dirs() -> List[str]:
    """
    Проверяет, что все рабочие папки существуют.
    Возвращает список недостающих папок (пустой список = всё в порядке).
    """
    # TODO: реализовать
    # Подсказка:
    #   need = [config.custom_data_path_abs, config.output_dir_abs, config.log_dir_abs]
    #   return [str(p) for p in need if not p.exists()]
    return []


# ---------------------------------------------------------------------------
# ОТЧЁТ И ЦИКЛ ПРОВЕРОК
# ---------------------------------------------------------------------------

def build_report() -> Dict[str, Any]:
    """
    Собирает общий отчёт о состоянии проекта.
    Этот же отчёт уходит в лог при каждой проверке.
    """
    report: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_ok": not config.validate(),
        "model": check_model_available(),
        "api": check_api_health(),
        "disk_ok": check_disk_space(),
    }
    missing_dirs = check_dirs()
    report["missing_dirs"] = missing_dirs
    return report


def format_report(report: Dict[str, Any]) -> str:
    """Превращает отчёт в читаемый текст для консоли."""
    lines = [
        "-" * 70,
        f"ОТЧЁТ {report['time']}",
        "-" * 70,
        f"  Конфигурация : {'OK' if report['config_ok'] else 'ПРОБЛЕМЫ'}",
        f"  Модель       : {'есть' if report['model'] else 'НЕТ'}"
        f" ({config.output_model_path_abs.name})",
        f"  API-сервер   : {'отвечает' if report['api'] else 'не отвечает'}"
        f" ({config.api_host}:{config.api_port})",
        f"  Место на диске: {'OK' if report['disk_ok'] else 'МАЛО - закончится!'}",
    ]
    if report["missing_dirs"]:
        lines.append("  Нет папок   :")
        lines.extend(f"      - {d}" for d in report["missing_dirs"])
    else:
        lines.append("  Папки       : все на месте")
    return "\n".join(lines)


def check_once() -> Dict[str, Any]:
    """
    Делает одну проверку и печатает отчёт. Используется при запуске без --watch.
    """
    report = build_report()
    print(format_report(report))

    problems = config.validate()
    if problems:
        log.warning("Проблемы в конфигурации: %s", "; ".join(problems))
    if not report["model"]:
        log.warning("Файл модели не найден: %s", config.output_model_path_abs)
    if not report["api"]:
        log.info("API-сервер не отвечает (если он ещё не запущен — это нормально)")

    return report


def watch(interval: Optional[int] = None) -> int:
    """
    Постоянно проверяет состояние проекта с заданным интервалом
    (по умолчанию config.monitor_interval = 60 секунд).

    Остановить: Ctrl+C
    """
    seconds = interval or config.monitor_interval
    log.info("Мониторинг запущен: проверка каждые %d сек. Остановка: Ctrl+C", seconds)

    try:
        while True:
            report = check_once()
            log.info("Проверка выполнена")
            time.sleep(seconds)
    except KeyboardInterrupt:
        log.info("Мониторинг остановлен пользователем")
        return 0


def show_stats() -> int:
    """
    Показывает статистику по лог-файлу: сколько ошибок, предупреждений,
    сколько всего записей. Помогает понять, часто ли что-то падает.
    """
    log_path = config.log_dir_abs / LOG_FILENAME
    print("-" * 70)
    print(" СТАТИСТИКА ЛОГОВ")
    print("-" * 70)

    if not log_path.exists():
        print(f"Лог-файл ещё не создан: {log_path}")
        return 1

    # TODO: подсчитать количество записей по уровням
    # Подсказка:
    #   counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    #   for line in log_path.read_text(encoding="utf-8").splitlines():
    #       for level in counts:
    #           if f"| {level}" in line:
    #               counts[level] += 1
    #               break
    #   for level, count in counts.items():
    #       print(f"  {level:<8} {count}")
    print(f"Лог-файл: {log_path}")
    print(f"Размер  : {log_path.stat().st_size / 1024:.1f} КБ")
    return 0


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: мониторинг и логи",
    )
    parser.add_argument("--watch", action="store_true",
                        help="проверять постоянно, а не один раз")
    parser.add_argument("--interval", type=int, default=None,
                        help="интервал проверки в секундах (по умолчанию 60)")
    parser.add_argument("--stats", action="store_true",
                        help="показать статистику по логам")
    parser.add_argument("--level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="уровень логирования")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа: разбор аргументов и запуск мониторинга."""
    args = parse_args(argv)

    if args.level:
        log = setup_logging(__name__, level=args.level, to_file=True)

    print("=" * 70)
    print(" demucs_custom / МОНИТОРИНГ")
    print("=" * 70)
    print(f"Лог-файл : {config.log_dir_abs / LOG_FILENAME}")
    print(f"Уровень  : {args.level or config.log_level}")

    if args.stats:
        return show_stats()

    if args.watch:
        return watch(interval=args.interval)

    check_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
