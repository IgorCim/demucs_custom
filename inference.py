# -*- coding: utf-8 -*-
"""
demucs_custom / inference.py
============================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ:
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

ПРО МОДЕЛЬ (важно!):
    Файл весов — это ЧИСТЫЙ state_dict, ровно того же вида, что отдаёт
    demucs.pretrained.get_model(). Поэтому load_model() сначала собирает
    обычную htdemucs (она нужна ради архитектуры), а затем накладывает на
    неё наши веса. Та же схема работает и для чекпоинта с состоянием
    оптимизатора (ключ "model_state" внутри) — например, quick_test_last.pth.

    ПЕРЕУЧИТЬ state_dict НЕЛЬЗЯ: имена дорожек зашиты в сеть. Наша
    htdemucs выдаёт дорожки в порядке drums, bass, other, vocals
    (проверено: get_model("htdemucs").sources). Названия берём у самой
    модели (model.sources), а НЕ из config.sources, где порядок другой
    (vocals, drums, bass, other) — иначе вокал сохранился бы как "drums".

ЗАПУСК:
    python inference.py song.mp3
    python inference.py song.mp3 --out ./output --source vocals
    python inference.py song.mp3 --model models/quick_test.pth
    python inference.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config, resolve_device


# ---------------------------------------------------------------------------
# МОДЕЛЬ
# ---------------------------------------------------------------------------

def _load_state_dict(path: Path) -> Dict[str, Any]:
    """
    Читает .pth и достаёт из него именно веса.

    Что там может лежать:
      * чистый state_dict (OrderedDict из тензоров) — так сохраняет
        train_custom.py для лучшей модели;
      * словарь с ключом "model_state" — так сохраняется полный
        чекпоинт (эпоха + состояние оптимизатора).

    Сначала пробуем weights_only=True: он запрещает pickle выполнить
    произвольный код, то есть защищает от чужого .pth. Наши данные —
    словари и тензоры, так что должно открыться. Если нет — читаем
    обычным способом и честно предупреждаем, что это небезопасно.
    """
    import torch

    try:
        loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        print("  [!] Безопасное чтение не вышло (%s: %s)" % (
            type(exc).__name__, str(exc).splitlines()[0][:120]))
        print("      Читаем обычным способом — это небезопасно для чужих .pth.")
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)

    # Полный чекпоинт: достаём из него веса.
    if isinstance(loaded, dict):
        for key in ("model_state", "state_dict", "model"):
            inner = loaded.get(key)
            if isinstance(inner, dict):
                return inner

    if not isinstance(loaded, dict):
        raise RuntimeError(
            "Файл %s не похож на веса модели: получили %s.\n"
            "  Ожидался state_dict (словарь «имя тензора -> тензор»)."
            % (path, type(loaded).__name__)
        )
    return loaded


def load_model(model_path: Optional[str] = None, device: Optional[str] = None):
    """
    Загружает нашу дообученную модель (или базовую htdemucs, если нашей нет).

    model_path — путь к .pth (по умолчанию config.output_model_path).
    device     — "cpu" / "cuda" / "mps" (по умолчанию config.device).

    Что происходит внутри:
      1. Проверяем, что файл модели существует.
         Если нет — честно говорим об этом и предлагаем скачать базовую htdemucs.
      2. Создаём модель через demucs.pretrained.get_model(config.demucs_model).
         Это НИЖЕ: сеть нужна ради архитектуры, веса всё равно затираются
         нашими. Сама библиотека скачает базовые веса (≈80 МБ) — это
         делается один раз и кэшируется.
      3. Загружаем наши веса: torch.load(...) + model.load_state_dict(...)
      4. Переносим на устройство и переводим в режим оценки
         (model.eval()), чтобы не считались градиенты и не менялись
         батч-нормализации.

    ВАЖНО: модель грузится ОДИН раз на весь процесс, а не на каждый файл.
    Иначе каждый запрос будет тратить секунды на загрузку весов.
    """
    import torch
    from demucs.pretrained import get_model

    path = Path(model_path) if model_path else config.output_model_path_abs
    wanted = device or config.device
    target = torch.device(resolve_device(wanted))

    if not path.exists():
        raise FileNotFoundError(
            "Модель не найдена: %s\n"
            "  Файл весов должен лежать в папке models/ проекта.\n"
            "  Скачай его из Google Drive (где обучал в Colab) и положи сюда,\n"
            "  либо укажи другой путь: --model путь/к/модели.pth"
            % path
        )

    if target.type == "cuda" and not torch.cuda.is_available():
        print("  [!] CUDA просят, но её нет — работаем на CPU")

    print("  Собираю архитектуру %s..." % config.demucs_model)
    model = get_model(config.demucs_model)

    state = _load_state_dict(path)
    try:
        # strict=True: если хоть одно имя или форма не совпали — падаем.
        # Молча грузить "почти те же" веса опаснее: сеть будет работать,
        # но выдавать мусор, и мы не поймём почему.
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        got = len(state)
        want = len(model.state_dict())
        raise RuntimeError(
            "Веса из %s не подходят к %s.\n"
            "  В файле %d ключей, у модели ожидается %d.\n"
            "  Похоже, веса от другой архитектуры или другой версии demucs.\n"
            "  Подробности:\n%s"
            % (path, config.demucs_model, got, want, exc)
        ) from exc

    model.eval()
    model.to(target)
    return model


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
    from demucs.apply import apply_model

    from convert_audio import read_audio, resample_audio, to_stereo

    input_path = Path(input_path)
    check_input_file(input_path)

    out_dir = Path(output_dir) if output_dir else config.output_dir_abs
    keep = list(keep_sources) if keep_sources else list(config.keep_sources)

    audio, file_rate = read_audio(input_path)
    audio = to_stereo(audio)
    if int(file_rate) != int(config.sample_rate):
        audio = resample_audio(audio, int(file_rate), int(config.sample_rate))

    # apply_model ждёт (батч, каналы, отсчёты) и сам режет длинное на окна.
    mix = audio.unsqueeze(0)

    # Устройство берём у самой модели, а не задаём по умолчанию: иначе
    # на GPU свёртки получат тензор с CPU и упадут.
    device = next(model.parameters()).device

    estimates = apply_model(
        model,
        mix,
        device=device,
        split=True,
        overlap=config.overlap,
        shifts=config.shifts,
        progress=False,
    )
    # estimates: (батч, дорожки, каналы, отсчёты)

    # ПОРЯДОК ДОРОЖЕК — у модели, а не из config. У htdemucs он
    # drums, bass, other, vocals, и менять его нельзя: это зашито в сеть.
    names = list(getattr(model, "sources", []) or config.sources)
    if len(names) != estimates.shape[1]:
        names = list(config.sources)[:estimates.shape[1]]

    stems = {name: estimates[0, index] for index, name in enumerate(names)}
    return save_stems(stems, out_dir, keep or None)


def separate_file(
    input_path: str,
    output_dir: Optional[Path] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    model: Optional[Any] = None,
) -> Dict[str, Path]:
    """
    Удобная обёртка для «одного файла от начала до конца»:
    загрузить модель (или взять уже загруженную) -> разделить -> сохранить.
    Возвращает словарь с путями к готовым дорожкам.

    model — если модель уже загружена, передаём её сюда, чтобы не грузить
    веса второй раз (важно для api.py/bot.py, где много файлов подряд).
    """
    if model is None:
        model = load_model(model_path, device)
    return separate(model, input_path, output_dir, config.keep_sources or None)


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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError("Файл не найден: %s" % path)
    if not path.is_file():
        raise ValueError("Это не файл, а папка: %s" % path)

    ext = path.suffix.lower()
    allowed = {e.lower() for e in config.audio_extensions}
    if ext not in allowed:
        raise ValueError(
            "Не понимаю формат %r: %s\n"
            "  Понимаю: %s"
            % (ext, path.name, ", ".join(sorted(allowed)))
        )

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_file_size_mb:
        raise ValueError(
            "Файл слишком большой: %.1f МБ (максимум %d МБ).\n"
            "  Подними предел в config.py: max_file_size_mb"
            % (size_mb, config.max_file_size_mb)
        )


def _write_stem(path: Path, audio: Any) -> None:
    """
    Пишет одну дорожку в файл нужного формата.

    wav идёт через convert_audio.write_wav — там уже есть надёжный выбор
    способа записи (в этой версии torchaudio.save требует torchcodec).
    Остальные форматы (mp3/flac) пишем через soundfile: он умеет их
    напрямую, без ffmpeg.
    """
    import numpy as np

    fmt = (config.output_format or "wav").lower()

    if fmt == "wav":
        from convert_audio import write_wav
        write_wav(path, audio, int(config.sample_rate), "16")
        return

    try:
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "Чтобы сохранять в %s, нужен пакет soundfile:\n"
            "    pip install soundfile\n"
            "Либо оставь wav (обычно этого хватает):\n"
            "    python inference.py song.mp3 --format wav"
            % fmt
        ) from None

    # soundfile ждёт (отсчёты, каналы), у нас (каналы, отсчёты).
    data = audio.detach().to("cpu").float().numpy()
    try:
        sf.write(str(path), np.ascontiguousarray(data.T), int(config.sample_rate),
                 format=fmt.upper())
    except Exception as exc:
        raise RuntimeError(
            "Не удалось записать %s: %s\n"
            "  Сборка soundfile не умеет этот формат. Сохрани в wav:\n"
            "    python inference.py song.mp3 --format wav"
            % (path.name, exc)
        ) from exc


def save_stems(
    sources: Dict[str, Any],
    output_dir: Path,
    keep_sources: Optional[List[str]] = None,
) -> Dict[str, Path]:
    """
    Сохраняет дорожки в файлы.

    Имена файлов: "vocals.wav", "drums.wav" и т.д. — то есть ровно
    названия дорожек, без префикса с именем трека: так просил сценарий
    проверки (в output/ должны появиться 4 файла с этими именами).
    Формат берём из config.output_format.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = set(keep_sources) if keep_sources else None
    fmt = (config.output_format or "wav").lower()

    result: Dict[str, Path] = {}
    for name, data in sources.items():
        if keep and name not in keep:
            continue
        path = out_dir / ("%s.%s" % (name, fmt))
        _write_stem(path, data)
        result[name] = path
    return result


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
        print("    Скачай её из Google Drive и положи в models/,")
        print("    либо укажи пульт через --model.")
        return 1

    print()
    print("Загружаю модель (это может занять минуту)...")
    started = time.time()
    try:
        paths = separate_file(input_path)
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        return 1
    except ValueError as exc:
        print(f"[!] {exc}")
        return 1
    except RuntimeError as exc:
        print(f"[!] Не получилось разделить: {exc}")
        return 1

    print(f"Готово за {time.time() - started:.1f} сек:")
    for name, path in paths.items():
        print(f"  {name:<8} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
