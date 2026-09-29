# -*- coding: utf-8 -*-
"""
demucs_custom / convert_audio.py
================================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Переводит любые исходники (MP3, FLAC, OGG, WAV и другие) в единый
    формат, который любит Demucs: WAV, 44100 Гц, стерео.

    Зачем это нужно: Demucs обучен на WAV 44100 Гц. Если подать ему MP3
    с другой частотой, звук будет сдвинут по высоте и модель сработает
    заметно хуже.

ПОЧЕМУ ЗДЕСЬ НЕ ТОЛЬКО torchaudio (важно, проверено на этой машине):
    В torchaudio 2.9+ функции `load` и `save` больше не работают сами -
    им нужен пакет `torchcodec`, а с ним ещё и ffmpeg. Без них вызов
    падает с ошибкой ImportError.

    Поэтому скрипт пробует сначала torchaudio (как положено), а если он
    недоступен - soundfile, а если и он не справился - внешний ffmpeg.
    В логе всегда видно, чем именно прочитали файл.

    Чтобы гарантированно работать с MP3/FLAC/OGG, поставь soundfile:
        pip install soundfile

ЗАПУСК:
    python convert_audio.py input/ output/
    python convert_audio.py input/ output/ --sample-rate 44100
    python convert_audio.py input/ output/ --recursive
    python convert_audio.py --help

ПРО ЛИЦЕНЗИЮ:
    Конвертируй только свои файлы, на которые у тебя есть права.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# НАСТРОЙКИ
# ---------------------------------------------------------------------------

# Какие расширения считаем аудиофайлами.
AUDIO_EXTENSIONS = (
    ".mp3", ".flac", ".ogg", ".wav",     # основные, как в задании
    ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma",  # вдруг попадутся
)

# Расширения, которые не нужно переписывать: они уже подходят.
ALREADY_GOOD = (".wav",)

# 16 = обычный CD-качество, 32 = float (без потерь, файл в 2 раза больше).
BIT_DEPTHS = {"16": 2, "32": 4}

# Запоминаем, чем именно читаем файлы (для отчёта в конце).
_LOADER: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# КОНСОЛЬ
# ---------------------------------------------------------------------------

def setup_console() -> None:
    """
    Делает так, чтобы русский текст печатался без "кракозябр".

    На русской Windows консоль по умолчанию живёт в кодировке cp866/cp1251,
    а в файлах у нас UTF-8. Без этой настройки вместо букв пользователь
    увидит непонятные символы.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ЧТЕНИЕ ИСХОДНИКОВ
# ---------------------------------------------------------------------------

def _read_with_torchaudio(path: Path) -> Tuple[torch.Tensor, int]:
    """Читает файл через torchaudio (нужен пакет torchcodec)."""
    import torchaudio

    tensor, sample_rate = torchaudio.load(str(path))
    return tensor.float(), int(sample_rate)


def _read_with_soundfile(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Читает файл через soundfile (внутри libsndfile).

    libsndfile понимает WAV, FLAC, OGG, AIFF и MP3.
    """
    import soundfile as sf

    # always_2d=True -> всегда возвращает (кадры, 2) даже для моно
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)


def _read_with_ffmpeg(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Читает файл внешней программой ffmpeg (универсальный вариант).

    ffmpeg понимает буквально всё: MP3, FLAC, OGG, AAC, M4A, WMA.
    Нужен только если не сработали torchaudio и soundfile.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg не найден в PATH")

    result = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"ffmpeg вернул код {result.returncode}")

    raw = np.frombuffer(result.stdout, dtype="<f4")
    if raw.size < 2:
        raise RuntimeError("ffmpeg не вернул звука (пустой файл?)")

    # ffmpeg отдаёт моно одним рядом - догадываемся о частоте и каналах
    # иначе, чем в soundfile: спрашиваем у ffmpeg через -show_streams.
    info = subprocess.run(
        [exe, "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
    )
    rate = 44100
    text = (info.stderr or b"").decode("utf-8", "replace")
    for line in text.splitlines():
        marker = "Hz, "
        if "Stream #0:0" in line and marker in line:
            try:
                rate = int(line.split(marker)[1].split()[0])
            except (IndexError, ValueError):
                pass
            break

    stereo = torch.from_numpy(np.ascontiguousarray(raw[0::2]))
    right = torch.from_numpy(np.ascontiguousarray(raw[1::2])) if raw.size >= 4 else stereo
    return torch.stack([stereo, right]), rate


def _can_read_with_torchaudio() -> bool:
    """
    Проверяем, читает ли torchaudio что-нибудь на этой машине.

    Проверять надо по-настоящему (на временном файле), а не только
    по факту `import torchaudio`: модуль импортируется, а вот внутри
    ему не хватает пакета torchcodec. Без такой проверки мы бы выбрали
    torchaudio, а потом на КАЖДОМ файле ловили бы ImportError и
    молча падали бы на soundfile - медленно и с врущей отчётностью.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        probe = Path(tmp_dir) / "probe.wav"
        _write_wav_stdlib(probe, torch.zeros(2, 1000), 44100, "16")
        _read_with_torchaudio(probe)
    return True


def _can_read_with_soundfile() -> bool:
    """Проверяет, читает ли soundfile что-нибудь на этой машине."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        probe = Path(tmp_dir) / "probe.wav"
        _write_wav_stdlib(probe, torch.zeros(2, 1000), 44100, "16")
        _read_with_soundfile(probe)
    return True


def _can_read_with_ffmpeg() -> bool:
    """Проверяет, что ffmpeg есть в PATH и он рабочий."""
    return shutil.which("ffmpeg") is not None


# Порядок важен: сначала то, что положено по уставу, потом - что реально есть.
READERS = (
    ("torchaudio", _read_with_torchaudio, _can_read_with_torchaudio),
    ("soundfile", _read_with_soundfile, _can_read_with_soundfile),
    ("ffmpeg", _read_with_ffmpeg, _can_read_with_ffmpeg),
)


def read_audio(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Читает любой аудиофайл. Возвращает (аудио (каналы, N), частота).

    При первом вызове выбираем ЧИТАЮЩИЙ модуль: каждый кандидат
    проверяется на настоящем файле, берётся первый рабочий.
    Дальше этот выбор не пересматривается, пока не пригодится запасной
    вариант (например, soundfile не тянет какой-то mp3, но его тянет ffmpeg).

    Если не сработал ни один - поднимаем исключение с понятным текстом.
    """
    if "reader" not in _LOADER:
        working: List[Tuple[str, Any]] = []
        for name, reader, probe in READERS:
            try:
                probe()
            except Exception:
                _LOADER.setdefault("unavailable", []).append(name)
                continue
            working.append((name, reader))
        # Запоминаем ТОЛЬКО рабочие способы. Иначе на каждом битом файле
        # мы бы зря дёргали torchaudio (который не работает) и получали
        # в сообщении путаницу вида "ошибка из-за TorchCodec", хотя
        # настоящая причина - битый файл.
        _LOADER["working"] = working
        if not working:
            raise RuntimeError(
                "Нечем читать аудио: не работает ни один способ.\n"
                "  Поставь soundfile (этого хватает для mp3, flac, ogg, wav):\n"
                "    pip install soundfile\n"
                "  или установи ffmpeg и добавь его в PATH."
            )
        _LOADER["reader"] = working[0][1]
        _LOADER["name"] = working[0][0]

    reader = _LOADER["reader"]
    try:
        return reader(path)
    except Exception as exc:
        # Выбранный способ не справился с ИМЕННО ЭТИМ файлом.
        # Пробуем остальные рабочие (например, mp3 не тянет soundfile,
        # но его тянет ffmpeg).
        errors = [f"{_LOADER.get('name')}: {type(exc).__name__}: {exc}"]
        for name, fallback in _LOADER.get("working", []):
            if fallback is reader:
                continue
            try:
                result = fallback(path)
                _LOADER["fallback"] = name
                return result
            except Exception as fallback_exc:
                errors.append(f"{name}: {type(fallback_exc).__name__}: {fallback_exc}")
        raise RuntimeError(
            "не удалось прочитать файл.\n"
            + "\n".join(f"  {line}" for line in errors)
        ) from exc


# ---------------------------------------------------------------------------
# ПРИВЕДЕНИЕ К НУЖНОМУ ВИДУ
# ---------------------------------------------------------------------------

def to_stereo(audio: torch.Tensor) -> torch.Tensor:
    """
    Делает аудио строго двухканальным, как требует Demucs.

      - 1 канал (моно)  -> дублируем в оба уха
      - 2 канала         -> ничего не делаем
      - больше 2         -> берём первые два (левый и правый)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(
            f"ожидался звук формы (каналы, N), а получилось {tuple(audio.shape)}"
        )

    channels = audio.shape[0]
    if channels == 1:
        return audio.repeat(2, 1)
    if channels == 2:
        return audio
    return audio[:2].contiguous()


def resample_audio(audio: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """
    Меняет частоту дискретизации через torchaudio.transforms.Resample.

    Если частота уже нужная - просто возвращает как есть, без затрат.
    """
    if int(orig_freq) == int(new_freq):
        return audio

    from torchaudio.transforms import Resample

    resampler = Resample(orig_freq=int(orig_freq), new_freq=int(new_freq))
    return resampler(audio)


# ---------------------------------------------------------------------------
# ЗАПИСЬ WAV
# ---------------------------------------------------------------------------

def _to_pcm16(audio: torch.Tensor) -> np.ndarray:
    """Переводит звук -1..1 в целые числа -32768..32767 (16 бит)."""
    data = np.clip(audio.detach().cpu().numpy().astype(np.float64), -1.0, 1.0)
    return (data * 32767.0).astype("<i2")


def _to_pcm32float(audio: torch.Tensor) -> np.ndarray:
    """Оставляет звук как float32 (32-битный WAV без потерь)."""
    data = np.clip(audio.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
    return data.astype("<f4")


def _write_wav_stdlib(path: Path, audio: torch.Tensor, sample_rate: int,
                      bit_depth: str = "16") -> None:
    """
    Сохраняет WAV стандартной библиотекой Python (модуль wave).

    Работает везде, где есть Python, и не зависит ни от чего.
    Умеет только 16 бит - модуль wave не поддерживает формат IEEE float.
    """
    if bit_depth != "16":
        raise ValueError(
            "модуль wave из стандартной библиотеки умеет только 16 бит"
        )

    pcm = _to_pcm16(audio)

    # WAV хранит звук вперемешку (левый, правый, левый, правый...),
    # поэтому транспонируем: (2, N) -> (N, 2) -> один ровный массив.
    interleaved = np.ascontiguousarray(pcm.T)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.tobytes())


def _write_wav_soundfile(path: Path, audio: torch.Tensor, sample_rate: int,
                         bit_depth: str = "16") -> None:
    """
    Сохраняет WAV через soundfile.

    Нужен для 32 бит: там формат IEEE float, а модуль wave из
    стандартной библиотеки помечает файл как обычный целочисленный PCM.
    Из-за этого игроки прочитали бы звук как шум. soundfile ставит
    правильную метку, и всё читается верно.
    """
    import soundfile as sf

    data = np.clip(
        audio.detach().cpu().numpy().astype(np.float32), -1.0, 1.0
    )
    subtype = {"16": "PCM_16", "32": "FLOAT"}.get(bit_depth)
    if subtype is None:
        raise ValueError(f"неподдерживаемая глубина: {bit_depth} бит")
    # soundfile ждёт (отсчёты, каналы), у нас (каналы, отсчёты).
    sf.write(str(path), np.ascontiguousarray(data.T), int(sample_rate),
             format="WAV", subtype=subtype)


def _write_wav_torchaudio(path: Path, audio: torch.Tensor, sample_rate: int,
                          bit_depth: str = "16") -> None:
    """Сохраняет WAV через torchaudio (нужен пакет torchcodec)."""
    import torchaudio

    tensor = torch.from_numpy(
        _to_pcm16(audio) if bit_depth == "16" else _to_pcm32float(audio)
    )
    torchaudio.save(str(path), tensor, sample_rate)


def write_wav(path: Path, audio: torch.Tensor, sample_rate: int,
              bit_depth: str = "16") -> str:
    """
    Сохраняет звук в WAV. Возвращает, каким способом записали.

    Для 16 бит: сначала пробуем torchaudio (как просили в задании),
    если он недоступен - пишем стандартной библиотекой, результат тот же.

    Для 32 бит - только soundfile: там формат IEEE float, который
    модуль wave умеет испортить (помечает как целый PCM).
    """
    if bit_depth == "32":
        try:
            import soundfile  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Для 32-битного WAV нужен пакет soundfile:\n"
                "    pip install soundfile\n"
                "Либо оставь 16 бит (обычно этого хватает):\n"
                "    --bit-depth 16"
            ) from None
        _LOADER["writer"] = "soundfile"
        _write_wav_soundfile(path, audio, sample_rate, bit_depth)
        return _LOADER["writer"]

    if "writer" not in _LOADER:
        _LOADER["writer_ok"] = False
        try:
            import torchaudio  # noqa: F401

            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                probe = Path(tmp) / "probe.wav"
                _write_wav_torchaudio(probe, torch.zeros(2, 64), 44100, "16")
            _LOADER["writer_ok"] = True
        except Exception:
            pass
        _LOADER["writer"] = (
            "torchaudio" if _LOADER["writer_ok"] else "wave (стандартная библиотека)"
        )

    if _LOADER["writer_ok"]:
        _write_wav_torchaudio(path, audio, sample_rate, bit_depth)
    else:
        _write_wav_stdlib(path, audio, sample_rate, bit_depth)
    return _LOADER["writer"]


# ---------------------------------------------------------------------------
# ОСНОВНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------

def convert_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int = 44100,
    bit_depth: str = "16",
) -> float:
    """
    Конвертирует один файл в WAV нужной частоты и стерео.

    Что происходит по шагам:
        1. читаем файл (mp3 / flac / ogg / wav / что угодно)
        2. делаем строго 2 канала
        3. пересчитываем частоту, если она не равна нужной
        4. сохраняем WAV
        5. печатаем, сколько это секунд

    ВОЗВРАЩАЕТ: длительность получившегося файла в секундах (float).

    Если файл битый - поднимает исключение с понятным текстом
    (его поймает batch_convert и пропустит файл).
    """
    src = Path(input_path)
    dst = Path(output_path)

    if not src.is_file():
        raise FileNotFoundError(f"файл не найден: {src}")

    # Папку для результата создаём: без неё будет ошибка при записи.
    dst.parent.mkdir(parents=True, exist_ok=True)

    bit_depth = str(bit_depth)
    if bit_depth not in BIT_DEPTHS:
        raise ValueError(
            f"глубина {bit_depth} бит не поддерживается, выбери 16 или 32"
        )

    # 1. Читаем
    audio, file_rate = read_audio(src)

    # 2. Строго стерео
    audio = to_stereo(audio)

    # 3. Нужная частота дискретизации
    audio = resample_audio(audio, file_rate, int(sample_rate))

    if audio.shape[1] == 0:
        raise ValueError("в файле нет ни одного отсчёта (пустой?)")

    # 4. Сохраняем
    write_wav(dst, audio, int(sample_rate), bit_depth)

    # 5. Длительность
    return audio.shape[1] / float(sample_rate)


def peek_wav_info(path: Path) -> Optional[Dict[str, int]]:
    """
    Быстро читает ТОЛЬКО заголовок wav, не читая сам звук.

    Нужно, чтобы понять: «а может, этот wav уже в нужном формате
    и конвертировать его незачем». Файл целиком при этом не открывается,
    поэтому проверка мгновенная.

    Основной способ - soundfile: он понимает все подтипы wav, включая
    IEEE float. Модуль wave из стандартной библиотеки на 32-битном
    float-WAV спотыкается об ошибку "unknown format", поэтому он тут
    только запасной вариант.

    Возвращает {"channels", "sample_rate", "frames", "bit_depth"}
    или None, если это не wav (или файл битый).
    """
    # сначала soundfile: умеет и PCM, и float
    try:
        import soundfile as sf

        info = sf.info(str(path))
        bit_depth = {
            "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24,
            "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64,
        }.get(info.subtype, 0)
        return {
            "channels": int(info.channels),
            "sample_rate": int(info.samplerate),
            "frames": int(info.frames),
            "bit_depth": bit_depth,
        }
    except Exception:
        pass

    # запасной путь: обычный PCM-wav разбираем модулем wave
    try:
        with wave.open(str(path), "rb") as wav_file:
            return {
                "channels": int(wav_file.getnchannels()),
                "sample_rate": int(wav_file.getframerate()),
                "frames": int(wav_file.getnframes()),
                "bit_depth": int(wav_file.getsampwidth()) * 8,
            }
    except Exception:
        return None


def find_audio_files(directory: Path, recursive: bool = False) -> List[Path]:
    """
    Находит все аудиофайлы в папке.

    recursive=True - искать ещё и во вложенных папках.
    Порядок стабильный (по имени), чтобы прогоны были одинаковыми.
    """
    pattern = "**/*" if recursive else "*"
    found = [
        p for p in sorted(directory.glob(pattern))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return found


def batch_convert(
    input_dir: str,
    output_dir: str,
    sample_rate: int = 44100,
    bit_depth: str = "16",
    recursive: bool = False,
    force: bool = False,
    skip_wav: bool = True,
) -> Dict[str, Any]:
    """
    Конвертирует все аудиофайлы из одной папки в другую.

    Параметры:
        input_dir   - откуда брать (создаётся, если её нет)
        output_dir  - куда складывать (создаётся, если её нет)
        sample_rate - частота на выходе (по умолчанию 44100)
        bit_depth   - "16" или "32"
        recursive   - искать во вложенных папках тоже
        force       - перезаписывать уже существующие .wav
        skip_wav    - не трогать wav, которые уже в нужном формате

    Возвращает словарь со статистикой:
        {"converted": N, "skipped": N, "failed": N, "seconds": X, "files": [...]}
    """
    setup_console()

    src_dir = Path(input_dir)
    dst_dir = Path(output_dir)

    print("=" * 72)
    print(" demucs_custom / КОНВЕРТАЦИЯ АУДИО")
    print("=" * 72)
    print(f"  Откуда   : {src_dir}")
    print(f"  Куда     : {dst_dir}")
    print(f"  Формат   : WAV, {sample_rate} Гц, стерео, {bit_depth} бит")

    # --- Папки: создаём, если их нет --------------------------------
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [!] Папки с исходниками не было - создал пустую: {src_dir}")
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Создал папку для результатов: {dst_dir}")

    # --- Находим файлы ------------------------------------------------
    files = find_audio_files(src_dir, recursive=recursive)
    total = len(files)

    print(f"  Найдено  : {total} аудиофайлов "
          f"({', '.join(e.lstrip('.') for e in AUDIO_EXTENSIONS[:4])} и другие)")
    print()

    if total == 0:
        print("  [!] Аудиофайлов не найдено.")
        print(f"      Положи mp3 / flac / ogg / wav в папку {src_dir}")
        print("      и запусти скрипт ещё раз.")
        return {"converted": 0, "skipped": 0, "failed": 0, "seconds": 0.0,
                "files": [], "reader": _LOADER.get("name", "?")}

    converted = 0
    skipped = 0
    failed = 0
    total_seconds = 0.0
    # Кто из исходников уже занял какое-то имя на выходе.
    produced_by: Dict[str, Path] = {}
    results: List[Dict[str, Any]] = []  # дадут одно и то же имя .wav

    for index, src in enumerate(files, start=1):
        # Имя на выходе: то же, но с .wav
        dst = dst_dir / (src.stem + ".wav")
        key = dst.name.lower()

        # --- ВАЖНЫЙ ПОРЯДОК ПРОВЕРОК ---------------------------------
        # Сначала ловим столкновение имён, и только потом - "уже есть".
        # Иначе второй файл молча пропустится под предлогом "уже есть",
        # а на самом деле его просто затер бы первый: трек.mp3 и трек.flac
        # оба хотят стать трек.wav.
        if key in produced_by:
            skipped += 1
            reason = (f"столкновение имён: и {produced_by[key].name}, и "
                      f"{src.name} стали бы {dst.name}. "
                      f"Оставлен {produced_by[key].name}. "
                      f"Переименуй один из них или запусти с --force")
            print(f"  [{index}/{total}] {src.name}: пропущен, {reason}")
            results.append({"source": str(src), "output": str(dst),
                            "status": "skipped", "reason": reason, "seconds": 0.0})
            continue

        if dst.exists() and not force:
            skipped += 1
            reason = "уже есть" + (
                " (а он уже WAV)" if src.suffix.lower() in ALREADY_GOOD else ""
            )
            print(f"  [{index}/{total}] {src.name}: пропущен, {reason}")
            results.append({"source": str(src), "output": str(dst),
                            "status": "skipped", "reason": reason, "seconds": 0.0})
            continue

        # wav, который УЖЕ в нужном формате, переписывать незачем.
        # Смотрим только заголовок файла - это мгновенно.
        if skip_wav and src.suffix.lower() in ALREADY_GOOD and not force:
            info = peek_wav_info(src)
            if info and info["sample_rate"] == int(sample_rate) \
                    and info["channels"] == 2 and info["bit_depth"] == 16:
                skipped += 1
                reason = "уже WAV 44100 Гц стерео, конвертировать незачем"
                produced_by[key] = src
                print(f"  [{index}/{total}] {src.name}: пропущен, {reason}")
                results.append({"source": str(src), "output": str(dst),
                                "status": "skipped", "reason": reason, "seconds": 0.0})
                continue

        try:
            seconds = convert_to_wav(
                str(src), str(dst),
                sample_rate=sample_rate, bit_depth=bit_depth,
            )
        except Exception as exc:
            # Один битый файл не должен останавливать всю конвертацию.
            failed += 1
            # Сообщение многострочное - печатаем как есть, но с отступом,
            # чтобы было видно, где кончился прогресс и началась причина.
            detail = " ".join(str(exc).split())
            print(f"  [{index}/{total}] {src.name}: [!] ПРОПУЩЕН, файл не читается")
            for line in str(exc).strip().splitlines():
                print(f"        {line.strip()}")
            if not str(exc).strip():
                print(f"        {detail}")
            # Не оставляем после себя обрезок файла.
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink()
            results.append({"source": str(src), "output": str(dst),
                            "status": "failed", "reason": detail, "seconds": 0.0})
            continue

        produced_by[key] = src
        converted += 1
        total_seconds += seconds
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  [{index}/{total}] {src.name} -> {dst.name}: "
              f"{seconds:.2f} сек, {size_mb:.2f} МБ")
        results.append({"source": str(src), "output": str(dst),
                        "status": "ok", "seconds": seconds})

    # --- Итог ----------------------------------------------------------
    minutes = int(total_seconds // 60)
    print()
    print("-" * 72)
    print(f"  Обработано {converted}/{total} файлов")
    if skipped:
        print(f"  Пропущено (без изменений): {skipped}")
    if failed:
        print(f"  С ошибками: {failed}")
    print(f"  Всего звука: {int(total_seconds)} сек ({minutes} мин {total_seconds % 60:.0f} сек)")
    print(f"  Читал через: {_LOADER.get('name', '?')}")
    print(f"  Писал через: {_LOADER.get('writer', '?')}")
    print("-" * 72)

    return {
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "seconds": total_seconds,
        "files": results,
        "reader": _LOADER.get("name", "?"),
        "writer": _LOADER.get("writer", "?"),
        "output_dir": str(dst_dir),
    }


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: конвертация MP3/FLAC/OGG в WAV 44100 Гц стерео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python convert_audio.py input/ output/\n"
            "  python convert_audio.py input/ output/ --recursive\n"
            "  python convert_audio.py input/ output/ --force\n"
        ),
    )
    parser.add_argument("input_dir", help="папка с исходниками (mp3, flac, ogg, wav)")
    parser.add_argument("output_dir", help="папка для результатов (wav)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="частота на выходе (по умолчанию 44100)")
    parser.add_argument("--bit-depth", default="16", choices=sorted(BIT_DEPTHS),
                        help="16 = компактно, 32 = без потерь (по умолчанию 16)")
    parser.add_argument("--recursive", action="store_true",
                        help="искать файлы во вложенных папках")
    parser.add_argument("--force", action="store_true",
                        help="перезаписывать уже готовые файлы")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа."""
    setup_console()
    args = parse_args(argv)

    result = batch_convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        recursive=args.recursive,
        force=args.force,
    )

    print()
    if result["converted"] == 0 and result["failed"] == 0:
        print("Конвертация завершена. Файлы в: " f"{result['output_dir']}")
        return 0

    if result["failed"] and result["converted"] == 0:
        print("Конвертация завершена. Файлы в: " f"{result['output_dir']}")
        return 1

    print("Конвертация завершена. Файлы в: " f"{result['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
