# -*- coding: utf-8 -*-
"""
demucs_custom / inference.py
============================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Разделяет новую песню на дорожки. Это «рабочая лошадка» сервиса —
    именно её вызывают api.py и bot.py.

    Вход : одна песня (mp3 / wav / flac)
    Выход: 4 файла — вокал, барабаны, бас, остальное

КАК ЭТО РАБОТАЕТ (внутри Demucs):
    1. Читаем файл, приводим к 44100 Гц (config.sample_rate).
    2. Нарезаем длинную песню на короткие окна (~10.9 сек) с перекрытием
       (config.overlap = 0.25), чтобы не было артефактов на стыках.
    3. Каждое окно прогоняем через нейросеть — получаем 4 дорожки.
    4. Склеиваем окна обратно с перекрытием и усреднением.
    5. Сохраняем 4 файла в output_dir.

    config.shifts = 0 означает "ровно один раз". Если поставить 1 или 2 —
    модель сделает 2-3 прогона со случайным сдвигом и усреднит результат.
    Качество выше, но время работы растёт в 2-3 раза.

ЗАПУСК:
    python inference.py song.mp3
    python inference.py song.mp3 --out ./output --source vocals
    python inference.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config


# ---------------------------------------------------------------------------
# МОДЕЛЬ
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[str] = None, device: Optional[str] = None):
    """
    Загружает нашу дообученную модель (или базовую htdemucs, если нашей нет).

    model_path — путь к .pth (по умолчанию config.output_model_path).
    device     — "cpu" / "cuda" / "mps" (по умолчанию config.device).

    Что происходит внутри:
      1. Проверяем, что файл модели существует.
         Если нет — честно говорим об этом и предлагаем скачать базовую htdemucs.
      2. Создаём модель через demucs.pretrained.get_model(config.demucs_model).
      3. Загружаем наши веса: torch.load(...) + model.load_state_dict(...)
      4. Переносим на устройство и переводим в режим оценки
         (model.eval()), чтобы не считались градиенты и не менялись
         батч-нормализации.

    ВАЖНО: модель грузится ОДИН раз на весь процесс, а не на каждый файл.
    Иначе каждый запрос будет тратить секунды на загрузку весов.
    """
    # TODO: реализовать загрузку модели
    # Подсказка:
    #   from demucs.pretrained import get_model
    #   model = get_model(config.demucs_model)
    #   state = torch.load(model_path, map_location=device)
    #   model.load_state_dict(state)
    #   model.eval().to(device)
    raise NotImplementedError("TODO: load_model() ещё не написан")


# ---------------------------------------------------------------------------
# РАЗДЕЛЕНИЕ
# ---------------------------------------------------------------------------

def separate(
    model,
    input_path: Path,
    output_dir: Optional[Path] = None,
    keep_sources: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Разделяет ОДИН файл на дорожки.

    input_path   — путь к исходной песне
    output_dir   — куда сохранять (по умолчанию config.output_dir)
    keep_sources — какие дорожки нужны (по умолчанию все из config.sources)

    Возвращает словарь {"vocals": путь_к_файлу, "drums": ..., ...}
    """
    # TODO: реализовать разделение
    # Подсказка: основной вызов делает библиотека demucs:
    #   from demucs.apply import apply_model
    #   sources = apply_model(model, wav_tensor_or_path,
    #                         device=device, split=True,
    #                         overlap=config.overlap,
    #                         shifts=config.shifts,
    #                         progress=False)
    raise NotImplementedError("TODO: separate() ещё не написан")


def separate_file(
    input_path: str,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Удобная обёртка для «одного файла от начала до конца»:
    загрузить модель (или взять уже загруженную) -> разделить -> сохранить.
    Возвращает словарь с путями к готовым дорожкам.
    """
    # TODO: собрать из load_model() + separate() + проверки существования файла
    raise NotImplementedError("TODO: separate_file() ещё не написан")


# ---------------------------------------------------------------------------
# ПРОВЕРКИ
# ---------------------------------------------------------------------------

def check_input_file(path: Path) -> None:
    """
    Проверяет входной файл ПЕРЕД обработкой, чтобы не тратить время на
    заведомо бесполезную работу. Бросает исключение с понятным текстом.

    Проверяем:
      - файл существует
      - расширение поддерживается (config.audio_extensions)
      - размер не больше config.max_file_size_mb
    """
    # TODO: реализовать проверки
    # Бросай понятные исключения: FileNotFoundError / ValueError
    raise NotImplementedError("TODO: check_input_file() ещё не написан")


def save_stems(
    sources: Dict[str, Any],
    output_dir: Path,
    keep_sources: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Сохраняет дорожки в файлы.

    Имена файлов: "<имя_трека>_vocals.wav", "<имя_трека>_drums.wav" и т.д.
    Формат берём из config.output_format.
    """
    # TODO: реализовать сохранение
    # Подсказка: soundfile.write(path, data.T, config.sample_rate) — data в виде [N, 2]
    raise NotImplementedError("TODO: save_stems() ещё не написан")


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: разделение песни на дорожки",
    )
    parser.add_argument("input", help="путь к песне (mp3 / wav / flac)")
    parser.add_argument("--out", default=None, help="папка для результатов")
    parser.add_argument("--model", default=None, help="путь к модели (.pth)")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps")
    parser.add_argument("--source", action="append", dest="sources",
                        default=None,
                        help="сохранить только эту дорожку (можно повторять)")
    parser.add_argument("--format", default=None,
                        choices=["wav", "mp3", "flac"],
                        help="формат выходных файлов")
    parser.add_argument("--shifts", type=int, default=None,
                        help="сколько раз прогнать со сдвигом (0-2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="только проверить файл, не разделять")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа: разбор аргументов и разделение одного файла."""
    args = parse_args(argv)

    if args.out:
        config.output_dir = args.out
    if args.model:
        config.output_model_path = args.model
    if args.device:
        config.device = args.device
    if args.format:
        config.output_format = args.format
    if args.shifts is not None:
        config.shifts = args.shifts
    if args.sources:
        config.keep_sources = args.sources

    input_path = Path(args.input)
    print("=" * 70)
    print(" demucs_custom / РАЗДЕЛЕНИЕ ПЕСНИ")
    print("=" * 70)
    print(f"Файл        : {input_path}")
    print(f"Результаты  : {config.output_dir_abs}")
    print(f"Устройство  : {config.device}")
    print(f"Дорожки     : {', '.join(config.keep_sources or config.sources)}")
    print(f"Модель      : {config.output_model_path_abs}")

    if not input_path.exists():
        print()
        print(f"[!] Файл не найден: {input_path}")
        return 1

    if args.dry_run:
        print()
        print("Режим --dry-run: файл проверен, разделение не запускалось.")
        return 0

    if not config.output_model_path_abs.exists():
        print()
        print(f"[!] Модель не найдена: {config.output_model_path_abs}")
        print("    Обучи модель (python train_custom.py) или укажи пульт через --model.")
        return 1

    # TODO: заменить на реальный вызов
    # started = time.time()
    # paths = separate_file(input_path)
    # print(f"Готово за {time.time() - started:.1f} сек:")
    # for name, path in paths.items():
    #     print(f"  {name:<8} -> {path}")
    print()
    print("[!] Это заглушка: разделение ещё не реализовано.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
