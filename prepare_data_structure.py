# -*- coding: utf-8 -*-
"""
demucs_custom / prepare_data_structure.py
=========================================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Создаёт ПРАВИЛЬНУЮ структуру папок для обучения и наполняет её
    примерами-треками, чтобы можно было сразу проверить всю дальнейшую
    цепочку (чтение данных -> обучение -> разделение), не имея под рукой
    настоящих записей.

СТРУКТУРА, КОТОРУЮ ОН СОЗДАЁТ:
    custom_dataset/
      train/                     <- для обучения
        track_001/
          mixture.wav            <- сумма всех источников (что подаём на вход)
          vocals.wav
          drums.wav
          bass.wav
          other.wav
        track_002/
        ...
      valid/                     <- для проверки качества (модель на них не учится)
        track_101/
        track_102/
        ...

    Почему так, а не "все вокалы в одной папке":
      - дорожки одного трека лежат рядом и не теряются;
      - train и valid РАЗНЫЕ треки, поэтому проверочная выборка
        не протекает в обучение (это главная ошибка новичков -
        если в train и valid попадут одни и те же треки, метрики
        будут красивыми, а модель бесполезной).

ПАРАМЕТРЫ ПРИМЕРОВ:
      44100 Гц, стерео (2 канала), 480000 семплов = 10.88 секунды
      16 бит PCM (обычный wav, открывается везде)
      mixture.wav = ПОБАЙТОВО (с точностью до квантования) сумма 4 дорожек

СКОЛЬКО МЕСТА НУЖНО:
      1 файл  = 480000 * 2 канала * 2 байта = 1.83 МБ
      1 трек  = 5 файлов = 9.2 МБ
      100 train + 20 valid = 120 треков = примерно 1.1 ГБ
    Нужно мало места? Создай меньше примеров:
        python prepare_data_structure.py --num-train 3 --num-valid 2

БЕЗОПАСНОСТЬ (важно):
    - Скрипт НИКОГДА не удаляет и не перезаписывает твои настоящие треки.
    - Каждая созданная папка помечается файлом-меткой ".demucs_example".
    - Папки без метки считаются "чужими" (реальными данными) и не трогаются.
    - Удалить примеры можно только так:
          python prepare_data_structure.py --clean

ЗАПУСК:
    python prepare_data_structure.py                  # 100 train + 20 valid
    python prepare_data_structure.py --num-train 3 --num-valid 2
    python prepare_data_structure.py --dry-run        # только показать план
    python prepare_data_structure.py --clean          # удалить примеры
    python prepare_data_structure.py --help
"""

from __future__ import annotations

import argparse
import shutil
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import config

# ---------------------------------------------------------------------------
# КОНСТАНТЫ
# ---------------------------------------------------------------------------

# Имя файла с полной смесью (то, что реально подаётся на вход модели).
MIXTURE_NAME = "mixture.wav"

# Файл-метка: по нему мы узнаём "наши" папки с примерами.
# Настоящие треки никогда не создавались этим скриптом, метки в них не будет.
EXAMPLE_MARKER = ".demucs_example"

# Амплитуда одной дорожки-примера (0.2). Сумма четырёх = 0.8 - не клиппит.
SOURCE_AMPLITUDE = 0.2

# Байт на один отсчёт: 16 бит = 2 байта.
BYTES_PER_SAMPLE = 2

# Сколько треков читать обратно при проверке (полная проверка 120 треков
# читает 1 ГБ - это лишние секунды, а 2-3 трека уже всё доказывают).
DEFAULT_VERIFY_LIMIT = 2


# ---------------------------------------------------------------------------
# КОНСОЛЬ
# ---------------------------------------------------------------------------

def setup_console() -> None:
    """
    Делает так, чтобы русский текст печатался без "кракозябр".

    На русской Windows консоль по умолчанию живёт в кодировке cp866/cp1251,
    а в файлах у нас UTF-8. Если не переключить вывод, пользователь увидит
    вместо букв непонятные символы. Поэтому явно просим Python печатать в UTF-8.

    Если консоль не умеет UTF-8 (редкий случай), ничего не ломаем:
    скрипт продолжит работать, просто буквы могут отобразиться как "????".
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# ИМЕНА ПАПОК
# ---------------------------------------------------------------------------

def track_dir_name(index: int) -> str:
    """
    Имя папки трека по его номеру: 1 -> "track_001", 120 -> "track_120".

    Номер сквозной: сначала идут train (1, 2, 3...), потом valid (101, 102...).
    Так сразу видно, к какому набору относится трек.
    """
    return f"track_{index:03d}"


# ---------------------------------------------------------------------------
# ЗАПИСЬ WAV
# ---------------------------------------------------------------------------

def _write_wav_stdlib(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Записывает WAV стандартной библиотекой Python (модуль wave).

    Никаких зависимостей: работает везде, где есть Python. Подходит, потому
    что 16-битный PCM WAV - самый простой и самый совместимый формат.

    Принимает звук в нормальном виде (диапазон -1..1) и сам переводит
    его в целые числа. Так нельзя случайно записать мусор, передав
    в эту функцию что-то не то.
    """
    pcm = _to_pcm16(audio)
    # WAV хранит звук вперемешку (левый, правый, левый, правый...),
    # поэтому транспонируем: (2, N) -> (N, 2) -> один ровный массив.
    interleaved = np.ascontiguousarray(pcm.T)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(pcm.shape[0]))
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.tobytes())


def _write_wav_torchaudio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    Записывает WAV через torchaudio (тот способ, который просили в задании).

    ВНИМАНИЕ (проверено на этой машине 26.09.2026):
        torchaudio версии 2.9+ больше не умеет писать файлы сам -
        внутри требуется пакет torchcodec. Если его нет, будет ошибка
        ImportError. Поэтому мы пробуем этот способ, а если не вышло -
        молча откатываемся на стандартную библиотеку (см. write_wav).
    """
    import torch
    import torchaudio

    tensor = torch.from_numpy(_to_pcm16(audio))
    torchaudio.save(str(path), tensor, sample_rate)


# Запоминаем, каким способом реально удалось записать (для отчёта).
_WRITER_USED: Dict[str, Any] = {}


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> str:
    """
    Записывает один WAV-файл и возвращает название способа записи.

    Принимает звук в нормальном виде (диапазон -1..1, форма (каналы, N)).

    Сначала пробуем torchaudio (как просили в задании). Если он недоступен
    или требует torchcodec - откатываемся на модуль wave из стандартной
    библиотеки. Результат в обоих случаях одинаковый: 16-битный WAV.

    Проверка способа делается один раз, на первом файле: дальше уже известно,
    чем писать, и лишних исключений не ловим.
    """
    if "method" not in _WRITER_USED:
        _WRITER_USED["ok"] = False
        try:
            _write_wav_torchaudio(path, audio, sample_rate)
            _WRITER_USED["ok"] = True
            _WRITER_USED["method"] = "torchaudio"
        except Exception:
            # Причину запомним и покажем в конце одной строкой.
            _WRITER_USED["method"] = "wave (стандартная библиотека)"
            _WRITER_USED["reason"] = "torchaudio требует пакет torchcodec"

    if _WRITER_USED.get("ok"):
        _write_wav_torchaudio(path, audio, sample_rate)
    else:
        _write_wav_stdlib(path, audio, sample_rate)
    return _WRITER_USED["method"]


def _to_pcm16(audio_float: np.ndarray) -> np.ndarray:
    """
    Переводит звук из диапазона -1..1 в целые числа -32768..32767
    и заодно обрезает всё, что вылезает за границы (иначе будет щелчок).
    """
    clipped = np.clip(np.asarray(audio_float, dtype=np.float64), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2")


def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """
    Читает WAV обратно. Возвращает (аудио float32 формы (каналы, N), частота).

    Сделан на модуле wave, а не на torchaudio, специально: чтение должно
    работать всегда, независимо от того, установлен ли torchcodec.
    """
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if width != BYTES_PER_SAMPLE:
        raise ValueError(
            f"{path.name}: ожидали {BYTES_PER_SAMPLE * 8} бит, а в файле {width * 8}"
        )

    pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    return (pcm.astype(np.float32) / 32768.0).T, sample_rate


# ---------------------------------------------------------------------------
# ГЕНЕРАЦИЯ ПРИМЕРОВ
# ---------------------------------------------------------------------------

def generate_track(
    index: int,
    length: int,
    sample_rate: int,
    seed: int,
    silence: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Делает содержимое одного трека-примера.

    Возвращает словарь {"vocals": массив, "drums": ..., "bass": ..., "other": ...}
    где каждый массив имеет форму (2, length).

    Про содержание:
      - silence=False (по умолчанию) - тихий случайный шум. Он удобен тем,
        что сразу видно: файл не пустой, дорожки разные, а сумма сходится.
      - silence=True - настоящая цифровая тишина (все нули).
    Ни то, ни другое не является музыкой: это ЗАГЛУШКИ для проверки
    структуры и форматов.
    """
    rng = np.random.default_rng(seed + index)   # одинаковый index -> одинаковый трек
    channels = 2

    stems: Dict[str, np.ndarray] = {}
    for name in config.sources:
        if silence:
            stems[name] = np.zeros((channels, length), dtype=np.float32)
            continue

        # Равномерный шум -> делим на максимум, чтобы пик был ровно
        # SOURCE_AMPLITUDE, и сумма четырёх дорожек гарантированно
        # не вышла за пределы -1..1 (не будет искажений-щелчков).
        noise = rng.uniform(-1.0, 1.0, size=(channels, length))
        peak = float(np.max(np.abs(noise)))
        if peak > 0:
            noise = noise / peak * SOURCE_AMPLITUDE
        stems[name] = noise.astype(np.float32)

    return stems


def mix_stems(stems: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Складывает все дорожки в одну смесь - ровно так же, как это делает
    любой звуковой редактор при сведении.

    ВАЖНО для обучения: mixture = vocals + drums + bass + other.
    Если правило нарушится, модель будет учиться на неправильных данных.
    """
    total = np.zeros_like(next(iter(stems.values())))
    for array in stems.values():
        total = total + array
    return np.clip(total, -1.0, 1.0)


# ---------------------------------------------------------------------------
# ОДНОТРЕКОВАЯ ПАПКА
# ---------------------------------------------------------------------------

def is_foreign_folder(path: Path) -> bool:
    """
    Папка с файлами, которые созданы НЕ нами (то есть настоящие данные).

    Такие папки скрипт не трогает и не перезаписывает - это защита
    от потери реальной работы.
    """
    if not path.is_dir():
        return False
    if (path / EXAMPLE_MARKER).exists():
        return False
    return any(child.is_file() for child in path.iterdir())


def create_track_folder(
    split_dir: Path,
    index: int,
    length: int,
    sample_rate: int,
    seed: int,
    silence: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Создаёт папку одного трека и кладёт в неё 5 WAV-файлов.

    Возвращает словарь с результатом:
        {"status": "created" | "skipped" | "exists" | "foreign",
         "path": путь, "files": [имена], "bytes": сколько заняли}
    """
    track_dir = split_dir / track_dir_name(index)
    result: Dict[str, Any] = {"path": track_dir, "files": [], "bytes": 0}

    # --- Защита чужих данных -------------------------------------------
    if track_dir.exists() and is_foreign_folder(track_dir):
        result["status"] = "foreign"
        return result

    # --- Уже созданный наш трек -----------------------------------------
    already_done = track_dir.exists() and (track_dir / MIXTURE_NAME).exists()
    if already_done and not force:
        result["status"] = "exists"
        return result

    track_dir.mkdir(parents=True, exist_ok=True)

    # --- Содержимое -------------------------------------------------------
    stems = generate_track(index, length, sample_rate, seed, silence)
    mixture = mix_stems(stems)

    written: List[str] = []
    for name, audio in list(stems.items()) + [(None, mixture)]:
        filename = MIXTURE_NAME if name is None else f"{name}.wav"
        path = track_dir / filename
        writer = write_wav(path, audio, sample_rate)
        written.append(filename)
        result["bytes"] += path.stat().st_size
    result["writer"] = _WRITER_USED.get("method", writer)

    # --- Метка: "эту папку можно безопасно удалить" ----------------------
    (track_dir / EXAMPLE_MARKER).write_text(
        "Это папка-пример, созданная prepare_data_structure.py.\n"
        "Можно удалить целиком: python prepare_data_structure.py --clean\n"
        f"Трек №{index}, {sample_rate} Гц, {length} семплов, "
        f"сид={seed + index}\n",
        encoding="utf-8",
    )

    result["status"] = "created"
    result["files"] = written
    return result


# ---------------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ (та, что просили в задании)
# ---------------------------------------------------------------------------

def create_example_structure(
    num_train: int = 100,
    num_valid: int = 20,
    length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    seed: Optional[int] = None,
    silence: bool = False,
    force: bool = False,
    dry_run: bool = False,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Создаёт всю структуру custom_dataset/train и custom_dataset/valid
    с примерами-треками.

    Параметры:
        num_train  - сколько треков в train (по умолчанию 100)
        num_valid  - сколько треков в valid (по умолчанию 20)
        length     - длина примера в семплах (по умолчанию config.segment_length)
        sample_rate- частота (по умолчанию config.sample_rate)
        seed       - зерно случайности (по умолчанию config.seed)
        silence    - True = писать настоящую тишину вместо шума
        force      - True = перезаписать уже созданные примеры
        dry_run    - True = только показать план, ничего не писать
        root       - куда создавать (по умолчанию config.custom_data_path_abs)

    Нумерация сквозная: train = 001..100, valid = 101..120.
    """
    length = length or config.segment_length
    sample_rate = sample_rate or config.sample_rate
    seed = config.seed if seed is None else seed
    root = root or config.custom_data_path_abs

    setup_console()

    total_tracks = num_train + num_valid
    bytes_per_file = length * 2 * BYTES_PER_SAMPLE      # 2 канала, 16 бит
    bytes_per_track = bytes_per_file * (len(config.sources) + 1)
    total_mb = total_tracks * bytes_per_track / 1024 / 1024

    print("=" * 74)
    print(" demucs_custom / ПОДГОТОВКА СТРУКТУРЫ ДАННЫХ")
    print("=" * 74)
    print(f"  Куда          : {root}")
    print(f"  Треков train  : {num_train}")
    print(f"  Треков valid  : {num_valid}")
    print(f"  Длина трека   : {length} семплов = {length / sample_rate:.2f} сек")
    print(f"  Формат        : WAV, {sample_rate} Гц, стерео, 16 бит")
    print(f"  Дорожки       : {', '.join(config.sources)} + {MIXTURE_NAME}")
    print(f"  Содержимое    : {'тишина' if silence else 'тихий шум (mixture = сумма)'}")
    print(f"  Место нужно   : примерно {total_mb:.0f} МБ ({total_tracks} треков по 5 файлов)")

    if not check_disk_space(root, total_mb):
        print()
        print("[!] Мало места на диске. Освободи место или уменьши количество:")
        print(f"    python {Path(__file__).name} --num-train 3 --num-valid 2")
        return {"status": "error", "reason": "not enough disk space"}

    if dry_run:
        print()
        print("Режим --dry-run: файлы не создавались, показан только план.")
        print()
        print("Пример будущей структуры:")
        print_example_tree(root, num_train)
        return {"status": "dry-run", "tracks": total_tracks}

    # --- Создаём ---------------------------------------------------------
    print()
    print(" Создаю треки...")

    train_dir = root / "train"
    valid_dir = root / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    foreign = 0
    used_bytes = 0

    # Если треков мало - показываем каждый (так нагляднее в примере).
    # Если много - каждый 20-й, иначе вывод будет на 120 строк.
    list_every = 1 if total_tracks <= 20 else 20

    for offset, count in enumerate((num_train, num_valid)):
        # Сквозная нумерация: train -> 1..num_train, valid -> num_train+1..
        is_train = offset == 0
        split_dir = train_dir if is_train else valid_dir
        split_name = "train" if is_train else "valid"
        first_index = 1 if is_train else num_train + 1
        print()
        print(f"  {split_name}/ ({count} треков):")

        for position in range(count):
            index = first_index + position
            result = create_track_folder(
                split_dir=split_dir,
                index=index,
                length=length,
                sample_rate=sample_rate,
                seed=seed,
                silence=silence,
                force=force,
            )
            used_bytes += result["bytes"]

            status = result["status"]
            if status == "created":
                created += 1
                if created <= 3 or created % list_every == 0:
                    print(f"    + {result['path'].name:<12} "
                          f"({result['bytes'] / 1024 / 1024:.1f} МБ)")
                elif created == 4:
                    print(f"    + ... (дальше сообщаю каждый {list_every}-й)")
            elif status == "exists":
                skipped += 1
            elif status == "foreign":
                foreign += 1
                print(f"    ! {result['path'].name} - папка с чужими данными, "
                      f"НЕ ТРОГАЮ")

    # --- Итог -------------------------------------------------------------
    print()
    print("=" * 74)
    print(f" Создано треков : {created}")
    print(f"  уже было      : {skipped}")
    if foreign:
        print(f"  чужих данных  : {foreign} (не тронуто)")
    print(f"  Занято места  : {used_bytes / 1024 / 1024:.0f} МБ")
    print(f"  Способ записи : {_WRITER_USED.get('method', '?')}")
    if _WRITER_USED.get("reason"):
        print(f"    (причина отката: {_WRITER_USED['reason']})")
    print("=" * 74)
    print()
    print(f" Структура данных создана. Теперь помести свои реальные треки "
          f"в {root.name}/train/ и {root.name}/valid/")

    return {
        "status": "ok",
        "created": created,
        "skipped": skipped,
        "foreign": foreign,
        "bytes": used_bytes,
        "root": str(root),
        "train_dir": str(train_dir),
        "valid_dir": str(valid_dir),
    }


# ---------------------------------------------------------------------------
# ПРОВЕРКА
# ---------------------------------------------------------------------------

def check_disk_space(root: Path, needed_mb: float) -> bool:
    """
    Проверяет, хватит ли места на диске.
    Возвращает True, если места достаточно.
    """
    probe = root if root.exists() else root.parent
    try:
        free_mb = shutil.disk_usage(str(probe)).free / 1024 / 1024
    except OSError:
        return True      # не смогли узнать - не блокируем работу
    print(f"  Свободно      : {free_mb:.0f} МБ")
    return free_mb > needed_mb * 1.05


def verify_track(track_dir: Path, length: int, sample_rate: int) -> List[str]:
    """
    Проверяет один созданный трек: читает файлы обратно и сравнивает.

    Проверяем:
      1. Что все 5 файлов на месте.
      2. Что у каждого 2 канала и правильная частота.
      3. Что длина равна length.
      4. Что mixture действительно равен сумме дорожек.

    Возвращает список найденных проблем (пустой список = всё отлично).
    """
    problems: List[str] = []

    # 1. Наличие файлов
    expected = [MIXTURE_NAME] + [f"{name}.wav" for name in config.sources]
    for filename in expected:
        if not (track_dir / filename).exists():
            problems.append(f"нет файла {filename}")
    if problems:
        return problems

    # 2-3. Параметры каждого файла
    loaded: Dict[str, np.ndarray] = {}
    for filename in expected:
        try:
            audio, rate = read_wav(track_dir / filename)
        except Exception as exc:
            problems.append(f"{filename}: не читается ({type(exc).__name__}: {exc})")
            continue
        if rate != sample_rate:
            problems.append(f"{filename}: частота {rate} вместо {sample_rate}")
        if audio.shape[0] != 2:
            problems.append(f"{filename}: каналов {audio.shape[0]} вместо 2")
        if audio.shape[1] != length:
            problems.append(f"{filename}: семплов {audio.shape[1]} вместо {length}")
        loaded[filename] = audio

    # 4. Смесь = сумма дорожек (с точностью до квантования 16 бит)
    if len(loaded) == len(expected):
        mixture = loaded[MIXTURE_NAME]
        stems_sum = None
        same_length = True

        for name in config.sources:
            stem = loaded[f"{name}.wav"]
            if stem.shape != mixture.shape:
                # Разные длины или число каналов: численно сравнивать нельзя.
                # Раньше тут был краш, теперь - понятное сообщение.
                problems.append(
                    f"{MIXTURE_NAME} ({mixture.shape[1]} семплов) и "
                    f"{name}.wav ({stem.shape[1]} семплов) разной длины"
                )
                same_length = False
                break
            stems_sum = stem.astype(np.float32) if stems_sum is None \
                else stems_sum + stem

        # Допуск считаем не "на глаз", а из самого формата:
        #   1 единица 16 бит = 1/32768 = 0.0000305
        #   каждая из 4 дорожек округлилась при записи  -> максимум 4 единицы
        #   плюс сама смесь округлилась при записи        -> ещё 1 единица
        #   итого до 5 единиц; берём 8 на запас и на ошибки float32.
        tolerance = 8.0 / 32768
        if same_length and stems_sum is not None:
            max_error = float(np.max(np.abs(mixture - stems_sum)))
            if max_error > tolerance:
                problems.append(
                    f"{MIXTURE_NAME} не равна сумме дорожек "
                    f"(расхождение {max_error:.6f} при допуске {tolerance:.6f})"
                )
        if float(np.max(np.abs(mixture))) > 1.0:
            problems.append(f"{MIXTURE_NAME}: есть перегруз (клиппинг)")

    return problems


def verify_structure(
    num_train: int = 100,
    num_valid: int = 20,
    limit: int = DEFAULT_VERIFY_LIMIT,
    length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    root: Optional[Path] = None,
) -> bool:
    """
    Проверяет, что созданная структура читается и данные корректны.

    limit - сколько треков проверить (по умолчанию 2: этого достаточно,
            а проверка 120 треков читала бы лишний гигабайт).
    Возвращает True, если проблем нет.
    """
    length = length or config.segment_length
    sample_rate = sample_rate or config.sample_rate
    root = root or config.custom_data_path_abs

    setup_console()

    print()
    print("=" * 74)
    print(" ПРОВЕРКА СОЗДАННОЙ СТРУКТУРЫ")
    print("=" * 74)

    all_problems: List[str] = []
    checked = 0

    for split_name in ("train", "valid"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            all_problems.append(f"нет папки {split_name}/")
            continue

        track_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        # Из всех берём первые limit (включая чужие - их тоже надо проверить)
        for track_dir in track_dirs[:limit]:
            # Чужие папки (реальные треки) не проверяем и не считаем ошибкой:
            # скрипт отвечает за примеры, а не за то, что положил пользователь.
            if is_foreign_folder(track_dir):
                print(f"  [--] {split_name}/{track_dir.name}: чужие данные, "
                      f"пропускаю (проверяю только примеры)")
                continue

            problems = verify_track(track_dir, length, sample_rate)
            if problems:
                print(f"  [!] {split_name}/{track_dir.name}:")
                for problem in problems:
                    print(f"        - {problem}")
                all_problems.extend(f"{track_dir.name}: {p}" for p in problems)
            else:
                print(f"  [OK] {split_name}/{track_dir.name}: 5 файлов, "
                      f"стерео {sample_rate} Гц, {length} семплов, "
                      f"mixture = сумма дорожек")
            checked += 1

    print("-" * 74)
    if all_problems:
        print(f" Проблем найдено: {len(all_problems)} (проверено треков: {checked})")
        return False

    print(f" Проблем нет. Проверено треков: {checked}")
    return True


# ---------------------------------------------------------------------------
# УДАЛЕНИЕ ПРИМЕРОВ
# ---------------------------------------------------------------------------

def clean_examples(root: Optional[Path] = None, dry_run: bool = False) -> int:
    """
    Удаляет ТОЛЬКО папки-примеры (те, где есть файл-метка .demucs_example).

    Настоящие треки не трогает никогда: если метки нет - папка пропускается
    и выводится предупреждение.
    """
    root = root or config.custom_data_path_abs
    setup_console()
    print("=" * 74)
    print(" demucs_custom / УДАЛЕНИЕ ПРИМЕРОВ")
    print("=" * 74)
    print(f"  Куда смотрю: {root}")

    removed = 0
    kept = 0
    freed_mb = 0.0

    for split_name in ("train", "valid"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        for track_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            if not (track_dir / EXAMPLE_MARKER).exists():
                kept += 1
                print(f"  ! {track_dir.name}: нет метки, это реальные данные - НЕ УДАЛЯЮ")
                continue

            size = sum(f.stat().st_size for f in track_dir.rglob("*") if f.is_file())
            if dry_run:
                print(f"  - было бы удалено: {track_dir.name} ({size / 1024 / 1024:.1f} МБ)")
            else:
                shutil.rmtree(track_dir)
                print(f"  - удалено: {track_dir.name} ({size / 1024 / 1024:.1f} МБ)")
            removed += 1
            freed_mb += size / 1024 / 1024

    if not dry_run:
        print()
        print(f" Удалено папок: {removed}, освобождено {freed_mb:.0f} МБ")
        print(f" Сохранено папок с реальными данными: {kept}")
    return 0


# ---------------------------------------------------------------------------
# ПОКАЗ СТРУКТУРЫ
# ---------------------------------------------------------------------------

def print_example_tree(root: Path, num_train: int = 100) -> None:
    """Печатает пример того, что получится (для --dry-run и для понимания)."""
    print()
    print(f"  {root.name}/")
    for split_name, first_index in (("train", 1), ("valid", num_train + 1)):
        print(f"    {split_name}/")
        note = "имена идут подряд" if split_name == "train" else "сразу после train"
        print(f"      {track_dir_name(first_index)}/          <- {note}")
        for filename in [MIXTURE_NAME] + [f"{n}.wav" for n in config.sources]:
            print(f"        {filename}")
        print(f"      {track_dir_name(first_index + 1)}/")
        print("        ...")


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: создание структуры папок для данных",
    )
    parser.add_argument("--num-train", type=int, default=100,
                        help="сколько треков в train (по умолчанию 100)")
    parser.add_argument("--num-valid", type=int, default=20,
                        help="сколько треков в valid (по умолчанию 20)")
    parser.add_argument("--data-path", default=None,
                        help="куда создавать (по умолчанию из config.py)")
    parser.add_argument("--length", type=int, default=None,
                        help="длина трека в семплах (по умолчанию 480000)")
    parser.add_argument("--sample-rate", type=int, default=None,
                        help="частота (по умолчанию 44100)")
    parser.add_argument("--seed", type=int, default=None,
                        help="зерно случайности (по умолчанию 42)")
    parser.add_argument("--silence", action="store_true",
                        help="писать настоящую тишину вместо шума")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать уже созданные примеры")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="не читать созданные файлы обратно (по умолчанию "
                             "проверка запускается всегда)")
    parser.add_argument("--verify-limit", type=int, default=DEFAULT_VERIFY_LIMIT,
                        help="сколько треков проверять чтением (по умолчанию 2)")
    parser.add_argument("--clean", action="store_true",
                        help="удалить созданные примеры (только свои!)")
    parser.add_argument("--dry-run", action="store_true",
                        help="только показать план, ничего не создавать")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа."""
    setup_console()
    args = parse_args(argv)

    if args.data_path:
        config.custom_data_path = args.data_path
    if args.length is not None:
        config.segment_length = args.length
    if args.sample_rate is not None:
        config.sample_rate = args.sample_rate
    if args.seed is not None:
        config.seed = args.seed

    root = config.custom_data_path_abs

    if args.clean:
        return clean_examples(root=root, dry_run=args.dry_run)

    result = create_example_structure(
        num_train=args.num_train,
        num_valid=args.num_valid,
        length=args.length,
        sample_rate=args.sample_rate,
        seed=args.seed,
        silence=args.silence,
        force=args.force,
        dry_run=args.dry_run,
        root=root,
    )

    if result.get("status") == "error":
        return 1

    if args.dry_run:
        return 0

    # Проверяем по умолчанию всегда: это дёшево (2 трека) и сразу
    # показывает, что получились настоящие читаемые wav-файлы.
    ok = True
    if args.verify:
        ok = verify_structure(
            num_train=args.num_train,
            num_valid=args.num_valid,
            limit=args.verify_limit,
            length=args.length,
            sample_rate=args.sample_rate,
            root=root,
        )

    if not ok:
        print()
        print("[!] Проверка нашла проблемы. Созданные файлы не трогаю, "
              "разбирайся по списку выше.")
        return 1

    print()
    print(" Готово. Дальше: положи настоящие треки в train/ и valid/.")
    print(" ВАЖНО: реальные треки - только те, на которые у тебя есть права.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
