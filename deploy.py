# -*- coding: utf-8 -*-
"""
demucs_custom / deploy.py
=========================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Помогает развернуть (запустить) проект на компьютере или на сервере.
    Никакой магии — только готовые команды, которые надо выполнить,
    и проверки, что всё настроено правильно.

ЧТО ВНУТРИ (пока заглушки, помечены # TODO):
    - check_dependencies()   : проверяет, что библиотеки установлены
    - check_config()         : проверяет настройки из config.py
    - check_model()          : проверяет, что модель на месте
    - make_dirs()            : создаёт рабочие папки
    - run_train()            : запускает обучение
    - run_api()              : запускает API-сервер
    - run_bot()              : запускает Telegram-бота
    - run_monitor()          : запускает мониторинг
    - show_plan()            : печатает все команды списком

ЗАПУСК:
    python deploy.py plan              # показать план запуска (безопасно)
    python deploy.py check             # проверить окружение
    python deploy.py dirs              # создать рабочие папки
    python deploy.py api               # запустить API-сервер
    python deploy.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import config

# Библиотеки, без которых проект не запустится: (имя пакета, имя для pip).
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("demucs", "demucs"),
    ("numpy", "numpy"),
    ("librosa", "librosa"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("aiofiles", "aiofiles"),
    ("telegram", "python-telegram-bot"),
]


# ---------------------------------------------------------------------------
# ПРОВЕРКИ
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    """
    Проверяет, установлены ли нужные библиотеки.
    Печатает таблицу: [OK] или [НЕТ] + подсказку, что доустановить.
    Возвращает True, если всё на месте.
    """
    print("-" * 70)
    print(" ПРОВЕРКА ЗАВИСИМОСТЕЙ")
    print("-" * 70)

    missing: List[str] = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
            print(f"  [OK]  {module_name}")
        except ImportError:
            print(f"  [НЕТ] {module_name}  ->  pip install {pip_name}")
            missing.append(pip_name)

    print("-" * 70)
    if missing:
        print(f"Не хватает {len(missing)} пакет(ов). Установи одной командой:")
        print("    pip install -r requirements.txt")
        return False

    print("Все зависимости установлены.")
    return True


def check_config() -> bool:
    """
    Проверяет настройки из config.py (config.validate()).
    Возвращает True, если проблем нет.
    """
    print("-" * 70)
    print(" ПРОВЕРКА КОНФИГУРАЦИИ")
    print("-" * 70)
    print(f"  {config.summary()}")

    problems = config.validate()
    if problems:
        print(f"  Проблем: {len(problems)}")
        for p in problems:
            print(f"    [!] {p}")
        return False

    print("  Конфигурация корректна.")
    return True


def check_model() -> bool:
    """
    Проверяет, что файл модели существует и читается.
    Возвращает True, если модель готова к работе.
    """
    print("-" * 70)
    print(" ПРОВЕРКА МОДЕЛИ")
    print("-" * 70)

    path = config.output_model_path_abs
    if not path.exists():
        print(f"  [НЕТ] Модель не найдена: {path}")
        print("        Обучи модель:  python train_custom.py")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [OK]  {path}")
    print(f"        размер: {size_mb:.1f} МБ")
    return True


def check_data() -> bool:
    """Проверяет, что папка с обучающими данными есть и не пустая."""
    print("-" * 70)
    print(" ПРОВЕРКА ДАННЫХ")
    print("-" * 70)

    data_dir = config.custom_data_path_abs
    if not data_dir.exists():
        print(f"  [НЕТ] Папка не найдена: {data_dir}")
        print("        Создать и подготовить:  python data_prep.py")
        return False

    for source in config.sources:
        sub = data_dir / source
        count = len([p for p in sub.glob("*") if p.is_file()]) if sub.exists() else 0
        flag = "OK" if count else "НЕТ"
        print(f"  [{flag:^3}] {source:<8} файлов: {count}")

    return True


# ---------------------------------------------------------------------------
# ДЕЙСТВИЯ
# ---------------------------------------------------------------------------

def make_dirs() -> int:
    """
    Создаёт все рабочие папки проекта
    (данные, модель, результаты, логи). Печатает, что создано.
    """
    print("-" * 70)
    print(" СОЗДАНИЕ ПАПОК")
    print("-" * 70)
    created = config.ensure_directories()
    if created:
        for path in created:
            print(f"  создана: {path}")
    else:
        print("  Все папки уже существуют.")
    return 0


def run_script(script: str, extra_args: Optional[List[str]] = None) -> int:
    """
    Запускает другой скрипт проекта в текущем интерпретаторе Python.

    Пример: run_script("api.py") -> запустит api.py
    """
    script_path = config.project_root / script
    if not script_path.exists():
        print(f"[!] Файл не найден: {script_path}")
        return 1

    command = [sys.executable, script] + list(extra_args or [])
    print(f"Запуск: {' '.join(command)}")
    print("-" * 70)
    try:
        # returncode: 0 = успех. Если скрипт упал — вернём его код.
        completed = subprocess.run(command, cwd=str(config.project_root))
        return completed.returncode
    except KeyboardInterrupt:
        print()
        print("Остановлено пользователем (Ctrl+C).")
        return 130


def run_train(args: argparse.Namespace) -> int:
    """Запускает обучение: train_custom.py"""
    extra: List[str] = []
    if getattr(args, "dry_run", False):
        extra.append("--dry-run")
    return run_script("train_custom.py", extra)


def run_api(args: argparse.Namespace) -> int:
    """Запускает API-сервер: api.py"""
    return run_script("api.py")


def run_bot(args: argparse.Namespace) -> int:
    """Запускает Telegram-бота: bot.py"""
    extra: List[str] = []
    if getattr(args, "check", False):
        extra.append("--check")
    return run_script("bot.py", extra)


def run_monitor(args: argparse.Namespace) -> int:
    """Запускает мониторинг: monitor.py"""
    return run_script("monitor.py")


def show_plan() -> int:
    """
    Печатает ПОЛНЫЙ план запуска проекта по шагам.
    Это самая полезная команда, если ты забыл порядок действий.
    """
    print("=" * 70)
    print(" demucs_custom / ПЛАН ЗАПУСКА")
    print("=" * 70)
    print("""
 ШАГ 0. Установить зависимости (один раз):

     cd путь_к_проекту\\demucs_custom
     pip install -r requirements.txt

     Если pip ругается, что torch не ставится — поставь сначала
     только torch с официальной страницы pytorch.org для своей системы.

 ШАГ 1. Положить данные:

     Скопируй свои аудиофайлы в custom_dataset\\
     (для обучения нужны папки vocals, drums, bass, other)

 ШАГ 2. Подготовить данные:

     python data_prep.py

 ШАГ 3. Обучить модель:

     python train_custom.py
     (Сначала можно проверить всё с --dry-run, обучение не запустится)

 ШАГ 4. Проверить качество:

     python evaluate.py

 ШАГ 5. Попробовать на своей песне:

     python inference.py моя_песня.mp3

 ШАГ 6. Запустить сервисы:

     python api.py          — веб-интерфейс (http://127.0.0.1:8000/docs)
     python bot.py          — Telegram-бот
     python monitor.py      — мониторинг и логи

 ВСЕ КОМАНДЫ СВЕДЕНЫ (то же самое через deploy.py):

     python deploy.py check     — проверить окружение
     python deploy.py dirs      — создать папки
     python deploy.py api       — запустить API
     python deploy.py bot       — запустить бота
""")
    return 0


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def check_all() -> int:
    """Прогоняет все проверки и печатает общий вердикт."""
    print("=" * 70)
    print(" demucs_custom / ПРОВЕРКА ОКРУЖЕНИЯ")
    print("=" * 70)

    results = {
        "зависимости": check_dependencies(),
        "конфигурация": check_config(),
        "данные": check_data(),
        "модель": check_model(),
    }

    print("=" * 70)
    print(" ИТОГ:")
    for name, ok in results.items():
        print(f"  {'[OK] ' if ok else '[НЕТ]'} {name}")

    # Модель может отсутствовать — это не катастрофа, а просто следующий шаг.
    if not results["зависимости"]:
        print()
        print("Сначала установи зависимости: pip install -r requirements.txt")
    elif not results["данные"]:
        print()
        print("Следующий шаг: положи данные и запусти python data_prep.py")
    elif not results["модель"]:
        print()
        print("Следующий шаг: обучи модель — python train_custom.py")
    else:
        print()
        print("Всё готово! Можно разделять песни: python inference.py песня.mp3")
    print("=" * 70)
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: запуск и проверка проекта",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=["plan", "check", "dirs", "train", "api", "bot", "monitor"],
        help="что сделать (по умолчанию plan)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="для train: только проверить, не обучать")
    parser.add_argument("--check", action="store_true",
                        help="для bot: только проверить настройки")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа: разбор действия и его выполнение."""
    args = parse_args(argv)

    if args.action == "plan":
        return show_plan()
    if args.action == "check":
        return check_all()
    if args.action == "dirs":
        return make_dirs()
    if args.action == "train":
        return run_train(args)
    if args.action == "api":
        return run_api(args)
    if args.action == "bot":
        return run_bot(args)
    if args.action == "monitor":
        return run_monitor(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
