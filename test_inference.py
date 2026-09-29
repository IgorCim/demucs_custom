# -*- coding: utf-8 -*-
"""
demucs_custom / test_inference.py
================================

БЫСТРЫЙ ТЕСТ РАЗДЕЛЕНИЯ на настоящей песне.

Что делает:
    Берёт аудиофайл, прогоняет его через дообученную модель
    (models/quick_test.pth) и раскладывает результат в output/.

ЗАПУСК:
    python test_inference.py song.mp3
    python test_inference.py song.mp3 --model models/quick_test.pth
    python test_inference.py song.mp3 --out ./output --device cuda

ЧТО ДОЛЖНО БЫТЬ:
    1) models/quick_test.pth — веса, скачанные из Google Drive
       (куда их положило обучение в Colab).
    2) output/ — сюда появятся vocals.wav, drums.wav, bass.wav, other.wav.

ПОЧЕМУ ЭТО ОТДЕЛЬНЫЙ ФАЙЛ, А НЕ ПРОСТО inference.py:
    inference.py — это рабочая лошадка для сервиса: там много ключей,
    дефолты берутся из config.py, ошибки возвращаются как коды для
    api.py/bot.py. Здесь же всё настроено ровно под проверку " положил
    mp3 — получил 4 дорожки", и любой шаг подсказывает, что делать дальше.
    Само разделение делает inference.separate() — логика одна и та же.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from config import config

# Модель по умолчанию. Путь относительный, но config умеет превращать
# его в абсолютный от папки проекта — значит скрипт работает из любой
# директории, а не только из корня.
DEFAULT_MODEL = "./models/quick_test.pth"

# Сколько примерно весит наша модель (state_dict). Слишком маленький файл —
# почти наверняка недокачанный обрывок, а не веса.
EXPECTED_SIZE_MB = 140.0


def _fail(text: str) -> int:
    print()
    print("[!] " + text)
    return 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="demucs_custom: тест разделения одной песни",
    )
    parser.add_argument("input", help="путь к песне (mp3 / wav / flac)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="путь к весам (по умолчанию %s)" % DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="куда класть дорожки")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps")
    parser.add_argument("--source", action="append", dest="sources",
                        default=None,
                        help="сохранить только эту дорожку (можно повторять)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    config.output_model_path = args.model
    if args.out:
        config.output_dir = args.out
    if args.device:
        config.device = args.device
    if args.sources:
        config.keep_sources = args.sources

    # --- печать шапки -------------------------------------------------------
    print("=" * 70)
    print(" demucs_custom / ТЕСТ РАЗДЕЛЕНИЯ")
    print("=" * 70)

    # --- 1. проверяем песню -------------------------------------------------
    song = Path(args.input)
    if not song.exists():
        return _fail(
            "Файл не найден: %s\n"
            "    Положи mp3 в папку проекта и повтори:\n"
            "    python test_inference.py song.mp3" % song
        )

    if not song.is_file():
        return _fail("Это не файл, а папка: %s" % song)

    # --- 2. проверяем модель ------------------------------------------------
    model_path = config.output_model_path_abs
    if not model_path.exists():
        return _fail(
            "Модель не найдена: %s\n"
            "    Сначала скачай модель: python download_from_colab.py\n"
            "    (или скопируй quick_test.pth из Google Drive в папку models/)"
            % model_path
        )

    size_mb = model_path.stat().st_size / (1024 * 1024)
    if size_mb < EXPECTED_SIZE_MB:
        return _fail(
            "Модель выглядит повреждённой: %s весит всего %.1f МБ,\n"
            "    а полные веса занимают около 160 МБ.\n"
            "    Похоже, файл докачался не до конца — удали его и скачай заново."
            % (model_path.name, size_mb)
        )

    print(f"Песня     : {song}")
    print(f"Модель    : {model_path}  ({size_mb:.0f} МБ)")
    print(f"Результаты: {config.output_dir_abs}")
    print(f"Устройство: {config.device}")

    # --- 3. грузим модель и разделяем --------------------------------------
    import inference

    print()
    print("Загружаю модель...")
    started = time.time()
    try:
        model = inference.load_model(str(model_path), args.device)
    except FileNotFoundError as exc:
        return _fail(str(exc))
    except RuntimeError as exc:
        return _fail("Модель не загрузилась: %s" % exc)
    print(f"  модель готова за {time.time() - started:.1f} сек")

    print()
    print("Разделяю песню (это самая долгая часть, особенно на CPU)...")
    started = time.time()
    try:
        paths = inference.separate(model, song, config.output_dir_abs,
                                   config.keep_sources or None)
    except FileNotFoundError as exc:
        return _fail(str(exc))
    except ValueError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 — здесь важно показать всё
        return _fail(
            "Разделение упало: %s: %s\n"
            "    Проверь, что модель скачалась целиком и песня читается."
            % (type(exc).__name__, exc)
        )
    elapsed = time.time() - started

    # --- 4. показываем результат -------------------------------------------
    print()
    print(f"Готово за {elapsed:.1f} сек:")
    for name in sorted(paths):
        path = paths[name]
        try:
            size_kb = path.stat().st_size / 1024
            print(f"  {name:<8} -> {path}  ({size_kb:.0f} КБ)")
        except OSError:
            print(f"  {name:<8} -> {path}")

    print()
    print("✅ Разделение завершено! Файлы в output/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
