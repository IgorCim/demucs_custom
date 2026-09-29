# -*- coding: utf-8 -*-
"""
demucs_custom / data_prep.py
============================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Готовит данные к обучению: читает треки с диска и отдаёт их
    в том виде, в котором Demucs их понимает.

    В проекте есть класс `CustomDataset` - это "переводчик" между
    файлами на диске и числами для нейросети.

ОЖИДАЕМАЯ СТРУКТУРА ДАННЫХ (её создаёт prepare_data_structure.py):
    custom_dataset/
        train/                     <- обучение
            track_001/
                mixture.wav        <- сумма всех дорожек (подаётся на вход)
                vocals.wav
                drums.wav
                bass.wav
                other.wav
            track_002/
        valid/                     <- проверка качества
            track_101/
                ...

ЧТО ВОЗВРАЩАЕТ ОДИН ЭЛЕМЕНТ ДАТАСЕТА (это важно):
    {
        "mixture": (2, 480000)          - что подаём на вход модели
        "sources": (4, 2, 480000)      - "правильный ответ" в порядке
                                          [vocals, drums, bass, other]
    }

ПОЧЕМУ ДАННЫЕ ЧИТАЮТСЯ ЛЕНИВО (важное решение):
    В __init__ мы только ИЩЕМ папки и БЫСТРО проверяем заголовки файлов
    (это доли секунды), а сами звуковые данные читаются в __getitem__ -
    в момент, когда они реально нужны.

    Почему так: если загрузить всё в __init__, то 100 треков по 10 секунд
    это примерно 1 ГБ оперативной памяти - а на 1000 треков уже 10 ГБ,
    и компьютер начнёт зависать. Ленивое чтение держит память на уровне
    одного батча.

ВАЖНО ПРО ПОРЯДОК ДОРОЖЕК:
    Здесь sources всегда в порядке config.sources = [vocals, drums, bass, other].
    Но модель Demucs внутри выдаёт каналы в порядке
    [drums, bass, other, vocals] (это мы проверили в check_demucs.py).
    Поэтому перед обучением надо переставить цели в порядок модели -
    это делает функция `reorder_sources_to_model_order` в этом же файле.
    Забыть про это - самая вероятная ошибка при дообучении.

ПРО ЛИЦЕНЗИЮ:
    Клади сюда только музыку, на которую у тебя есть права.

ЗАПУСК:
    python data_prep.py                       # проверить, что данные читаются
    python data_prep.py --split train         # только обучающая выборка
    python data_prep.py --make-examples       # создать примеры, если папка пуста
    python data_prep.py --help
"""

from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import config as project_config

# Имя файла со смесью. Остальные дорожки берём из config.sources.
MIXTURE_NAME = "mixture.wav"

# Префикс папки трека: track_001, track_002 и так далее.
TRACK_PREFIX = "track_"

# Метка папок-примеров (создаётся prepare_data_structure.py).
EXAMPLE_MARKER = ".demucs_example"

# Ширина отсчёта в байтах -> сколько бит.
SAMPLE_WIDTH_BITS = {1: 8, 2: 16, 3: 24, 4: 32}


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
# ЧТЕНИЕ WAV
# ---------------------------------------------------------------------------
# ВНИМАНИЕ (проверено на этой машине 26.09.2026):
#     torchaudio.load в версии 2.9+ требует пакет torchcodec. Без него
#     вызов падает с ImportError. Поэтому сначала пробуем torchaudio
#     (как положено по уставу), а если не получилось - читаем wav
#     стандартной библиотекой Python. Результат одинаковый.
# ---------------------------------------------------------------------------

_LOADER_INFO: Dict[str, Any] = {}


def _decode_pcm(raw: bytes, width: int, channels: int) -> torch.Tensor:
    """
    Превращает «сырые» байты из wav-файла в тензор torch (каналы, N).

    Поддерживает 8, 16, 24 и 32 бита, а также 32-битный float.
    Возвращает float32 в диапазоне -1..1.
    """
    import numpy as np

    if width == 1:
        # 8 бит в wav хранятся как БЕЗЗНАКОВЫЕ числа 0..255,
        # где 128 - это тишина. Поэтому вычитаем 128 и делим на 128.
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        # 24 бита - это 3 байта. Собираем их в одно 32-битное число
        # вручную: младшие 2 байта + старший байт со знаком.
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int32 = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        # Знак числа живёт в третьем байте: если он 0x80+, число отрицательное.
        negative = (data[:, 2] & 0x80) != 0
        as_int32[negative] -= 1 << 24
        data = as_int32.astype(np.float32) / 8388608.0
    elif width == 4:
        # Здесь нельзя отличить целое от float только по размеру.
        # Пробуем float32 - если значения выглядят как нормальный звук
        # (не больше единицы по модулю), значит это float-формат.
        as_float = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if float(np.max(np.abs(as_float))) <= 1.0:
            data = as_float
        else:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"неподдерживаемая ширина отсчёта: {width} байт")

    if channels <= 0:
        raise ValueError("в файле нет ни одного канала")
    return torch.from_numpy(np.ascontiguousarray(data.reshape(-1, channels).T))


def _read_wav_stdlib(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Читает wav стандартной библиотекой Python.

    Возвращает (аудио float32 формы (каналы, N), частота дискретизации).
    Работает всегда, потому что модуль wave есть в любом Python.
    """
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if width in (1, 2, 3, 4):
        audio = _decode_pcm(frames, width, channels)
    else:
        # 32-битный float-формат модуль wave опознаёт отдельно.
        if width == 4 and hasattr(wave, "IEEE_FLOAT"):
            audio = _decode_pcm(frames, 4, channels)
        else:
            raise ValueError(f"{path.name}: непонятный формат, {width} байт на отсчёт")

    return audio, sample_rate


def _read_wav_torchaudio(path: Path) -> Tuple[torch.Tensor, int]:
    """Читает wav через torchaudio (нужен пакет torchcodec)."""
    import torchaudio

    tensor, sample_rate = torchaudio.load(str(path))
    return tensor.float(), int(sample_rate)


def read_wav(path: Path) -> Tuple[torch.Tensor, int]:
    """
    Читает один wav-файл. Возвращает (аудио, частота).

    Сначала пробуем torchaudio, при неудаче - стандартную библиотеку.
    Способ запоминается, чтобы не пробовать заново на каждом файле.
    """
    if "method" not in _LOADER_INFO:
        try:
            # Сначала проверяем на временном файле, что torchaudio вообще
            # умеет читать. Иначе ошибка вылезет на первом же треке данных.
            _probe_torchaudio()
            _LOADER_INFO["method"] = "torchaudio"
            _LOADER_INFO["ok"] = True
        except Exception:
            _LOADER_INFO["ok"] = False
            _LOADER_INFO["method"] = "wave (стандартная библиотека)"
            _LOADER_INFO["reason"] = "torchaudio требует пакет torchcodec"

    if _LOADER_INFO.get("ok"):
        return _read_wav_torchaudio(path)
    return _read_wav_stdlib(path)


def _probe_torchaudio() -> None:
    """
    Проверяет, читает ли torchaudio wav на этой машине.

    Если нет - мы молча переключимся на стандартную библиотеку, но
    пользователь должен знать почему (это показывается в логе).
    """
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp_dir:
        probe = Path(tmp_dir) / "probe.wav"
        silence = np.zeros((2, 1000), dtype="<i2")
        with wave.open(str(probe), "wb") as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(silence.tobytes())
        _read_wav_torchaudio(probe)


def read_wav_info(path: Path) -> Optional[Tuple[int, int, int]]:
    """
    Быстро читает ТОЛЬКО заголовок wav, не читая сам звук.

    Возвращает (каналы, частота, число кадров) или None, если это не wav.
    Очень быстро: файлы не открываются целиком, поэтому проверка
    сотни треков занимает доли секунды.
    """
    try:
        with wave.open(str(path), "rb") as wav_file:
            return (
                int(wav_file.getnchannels()),
                int(wav_file.getframerate()),
                int(wav_file.getnframes()),
            )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ПРИВЕДЕНИЕ К ОБЩЕМУ ВИДУ
# ---------------------------------------------------------------------------

def to_stereo(audio: torch.Tensor) -> torch.Tensor:
    """
    Делает аудио строго двухканальным, как требует Demucs.

      - 1 канал (моно)  -> дублируем в оба уха
      - 2 канала         -> ничего не делаем
      - больше 2         -> берём первые два (левый и правый)

    Возвращает тензор формы (2, N).
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(f"ожидался звук формы (каналы, N), а получилось {tuple(audio.shape)}")

    channels = audio.shape[0]
    if channels == 1:
        return audio.repeat(2, 1)
    if channels == 2:
        return audio
    return audio[:2].contiguous()


def resample_audio(audio: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """
    Меняет частоту дискретизации через torchaudio.transforms.Resample.

    Зачем: записи бывают 22050, 48000, 96000 Гц, а модель жёстко заточена
    под 44100 Гц. Без пересчёта звук будет звучать выше или ниже по
    высоте (это называется «питч-шифт»).

    Если частота уже нужная - просто возвращаем как есть, без затрат.
    """
    if int(orig_freq) == int(new_freq):
        return audio

    from torchaudio.transforms import Resample

    resampler = Resample(orig_freq=int(orig_freq), new_freq=int(new_freq))
    return resampler(audio)


def fit_length(audio: torch.Tensor, length: int) -> torch.Tensor:
    """
    Делает звук ровно нужной длины.

      - короче  -> добавляет нули в конец (тишина)
      - длиннее -> обрезает лишнее
      - равно   -> ничего не делает
    """
    current = audio.shape[1]
    if current == length:
        return audio
    if current > length:
        return audio[:, :length].contiguous()
    padding = torch.zeros(
        (audio.shape[0], length - current), dtype=audio.dtype, device=audio.device
    )
    return torch.cat([audio, padding], dim=1)


def resample_length_after_fit(length: int, orig_freq: int, new_freq: int) -> int:
    """
    Сколько семплов будет после пересчёта частоты.

    Нужно, чтобы заранее понимать длину файла, прочитанного только
    по заголовку (без чтения звука).
    """
    if int(orig_freq) == int(new_freq):
        return length
    return int(math.ceil(length * int(new_freq) / float(orig_freq)))


# ---------------------------------------------------------------------------
# ОСНОВНОЙ КЛАСС
# ---------------------------------------------------------------------------

class CustomDataset(Dataset):
    """
    Датасет для дообучения Demucs.

    Умеет отдавать треки из папок, созданных prepare_data_structure.py:

        root_dir/
          train/
            track_001/
              mixture.wav, vocals.wav, drums.wav, bass.wav, other.wav
          valid/
            track_101/
              ...

    Один элемент датасета - это словарь:
        "mixture": (2, segment_length)     - вход для модели
        "sources": (4, 2, segment_length)  - эталон в порядке
                                             [vocals, drums, bass, other]
        "track":   "track_001"             - имя (для логов, модель его не ест)

    АВТОМАТИЧЕСКИ ДЕЛАЕТ:
        - читает только .wav, остальное пропускает с предупреждением
        - приводит моно к стерео
        - пересчитывает частоту, если она не равна нужной
        - если дорожки в треке разной длины - обрезает все до самой короткой
        - если трек короче segment_length - дополняет нулями
        - если длиннее - обрезает

    ПАРАМЕТРЫ:
        root_dir       - папка с данными (та, где лежат train/ и valid/)
        split          - "train" или "valid"
        segment_length - длина кусочка в семплах
        sample_rate    - нужная частота дискретизации
        sources        - какие дорожки нужны и в каком порядке
        check_mixture  - проверять ли, что mixture равна сумме дорожек
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        segment_length: int = 480000,
        sample_rate: int = 44100,
        sources: Optional[Sequence[str]] = None,
        check_mixture: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = str(split)
        self.segment_length = int(segment_length)
        self.sample_rate = int(sample_rate)
        self.sources = list(sources) if sources else list(project_config.sources)
        self.check_mixture = bool(check_mixture)
        self.verbose = bool(verbose)

        # --- Находим папку выборки -------------------------------------
        self.split_dir = self.root_dir / self.split
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Папка с данными не найдена: {self.root_dir}\n"
                f"  Создать её и структуру можно так:\n"
                f"    python prepare_data_structure.py --num-train 100 --num-valid 20"
            )
        if not self.split_dir.is_dir():
            available = [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            hint = ", ".join(sorted(available)) if available else "папок нет"
            raise FileNotFoundError(
                f"Папка выборки не найдена: {self.split_dir}\n"
                f"  Что есть в {self.root_dir.name}: {hint}\n"
                f"  Ожидалась папка '{self.split}' с подпапками track_XXX."
            )

        # --- Собираем список треков --------------------------------------
        self.tracks: List[Path] = self._scan_tracks()
        if not self.tracks:
            raise FileNotFoundError(
                f"В папке {self.split_dir} нет ни одного трека.\n"
                f"  Ожидались подпапки вида track_001 с файлами "
                f"{MIXTURE_NAME} и дорожек.\n"
                f"  Создать примеры: python prepare_data_structure.py "
                f"--num-train 3 --num-valid 2"
            )

        # Резолвер для пересчёта частоты (создаётся один раз, он тяжёлый).
        self._resamplers: Dict[Tuple[int, int], Any] = {}

        # О треках, о которых уже предупредили. Зачем: при обучении один и тот
        # же трек читается десятки раз (эпохи x батчи), и без этого счётчика
        # в консоль сыпались бы тысячи одинаковых строк про одно и то же.
        self._warned: set = set()

        if self.verbose:
            self._print_report()

    def _warn_once(self, key: Any, message: str) -> None:
        """Печатает предупреждение про конкретный трек только один раз."""
        if not self.verbose or key in self._warned:
            return
        self._warned.add(key)
        print(f"  [!] {message}")

    # -- служебное ---------------------------------------------------------

    def _scan_tracks(self) -> List[Path]:
        """
        Находит все папки треков и сразу отсеивает негодные.

        Что делаем и почему:
          - берём только папки, названные track_*  (остальное не треки)
          - пропускаем папки без нужных файлов      (предупреждение)
          - предупреждаем про файлы не-wav          (они не используются)
        """
        tracks: List[Path] = []
        warnings: List[str] = []

        candidates = sorted(
            d for d in self.split_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith(TRACK_PREFIX)
        )

        for track_dir in candidates:
            # 1. Какие файлы вообще лежат в папке трека
            try:
                entries = [p for p in track_dir.iterdir() if p.is_file()]
            except OSError as exc:
                warnings.append(f"{track_dir.name}: не смог прочитать папку ({exc})")
                continue

            wavs = {
                p.name.lower(): p for p in entries
                if p.suffix.lower() == ".wav"
            }

            # 2. Предупреждаем про файлы других форматов.
            #    Маркер .demucs_example - наш собственный, его не считаем мусором.
            foreign = [
                p.name for p in entries
                if p.suffix.lower() != ".wav" and p.name != EXAMPLE_MARKER
            ]
            if foreign:
                warnings.append(
                    f"{track_dir.name}: пропущены не-wav файлы: "
                    f"{', '.join(sorted(foreign)[:5])}"
                )

            # 3. Проверяем, что все нужные файлы на месте.
            needed = [MIXTURE_NAME] + [f"{name}.wav" for name in self.sources]
            missing = [n for n in needed if n.lower() not in wavs]
            if missing:
                warnings.append(
                    f"{track_dir.name}: пропущен трек, не хватает файлов: "
                    f"{', '.join(missing)}"
                )
                continue

            tracks.append(track_dir)

        self._scan_warnings = warnings
        return tracks

    def _get_resampler(self, orig_freq: int) -> Any:
        """Возвращает готовый Resample для нужной пары частот (с кэшем)."""
        if orig_freq == self.sample_rate:
            return None
        key = (int(orig_freq), int(self.sample_rate))
        if key not in self._resamplers:
            from torchaudio.transforms import Resample

            self._resamplers[key] = Resample(
                orig_freq=key[0], new_freq=key[1]
            )
        return self._resamplers[key]

    def _print_report(self) -> None:
        """Печатает короткую сводку о том, что нашлось в папке."""
        print(f"  Выборка      : {self.split} ({len(self.tracks)} треков)")
        print(f"  Папка        : {self.split_dir}")
        print(f"  Длина куска  : {self.segment_length} семплов "
              f"= {self.segment_length / self.sample_rate:.2f} сек")
        print(f"  Частота      : {self.sample_rate} Гц")
        print(f"  Дорожки      : {', '.join(self.sources)}")
        for line in getattr(self, "_scan_warnings", []):
            print(f"  [!] {line}")

    def describe_track(self, index: int) -> str:
        """Короткое описание трека - для сообщений об ошибках."""
        return f"{self.split}/{self.tracks[index].name}"

    # -- главное -----------------------------------------------------------

    def _load_one(self, path: Path) -> torch.Tensor:
        """
        Читает один файл и приводит его к (2, N) float32 нужной частоты.

        Порядок важен:
          1. чтение как есть          -> (каналы, N_как_есть)
          2. приведение к стерео      -> (2, ...)
          3. пересчёт частоты         -> (2, N_при_новой_частоте)
        """
        audio, file_rate = read_wav(path)
        audio = to_stereo(audio)
        resampler = self._get_resampler(file_rate)
        if resampler is not None:
            audio = resampler(audio)
        return audio.float()

    def __len__(self) -> int:
        """Сколько треков в выборке (это то, что нужно DataLoader)."""
        return len(self.tracks)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Возвращает один трек для обучения.

        Формы на выходе:
            "mixture" -> (2, segment_length)
            "sources" -> (4, 2, segment_length) в порядке self.sources
        """
        if index < 0:
            index += len(self.tracks)
        if not 0 <= index < len(self.tracks):
            raise IndexError(
                f"Нет трека с номером {index} (в выборке {self.split} "
                f"их {len(self.tracks)})"
            )

        track_dir = self.tracks[index]

        # 1. Читаем смесь и все дорожки.
        mixture = self._load_one(track_dir / MIXTURE_NAME)
        stems = [
            self._load_one(track_dir / f"{name}.wav")
            for name in self.sources
        ]

        # 2. Дорожки могут быть разной длины (даже после пересчёта частоты,
        #    если файлы были выгружены неровно). Смесь тоже своя длина.
        #    Обрезаем ВСЁ до самой короткой части - иначе сложение невозможно.
        lengths = [mixture.shape[1]] + [stem.shape[1] for stem in stems]
        target = min(lengths)
        if len(set(lengths)) > 1:
            self._warn_once(
                ("len", track_dir.name),
                f"{self.split}/{track_dir.name}: дорожки разной длины "
                f"({', '.join(str(v) for v in lengths)}), "
                f"обрезаю все до {target}",
            )

        mixture = fit_length(mixture, target)
        stems = [fit_length(stem, target) for stem in stems]

        # 3. Доводим до нужной длины куска: нули или обрезка.
        mixture = fit_length(mixture, self.segment_length)
        stems = [fit_length(stem, self.segment_length) for stem in stems]

        sources = torch.stack(stems, dim=0)      # (4, 2, segment_length)

        # 4. Незачем проверять сумму при каждом чтении во время обучения
        #    (это лишнее время на каждом батче), но при проверке полезно.
        if self.check_mixture:
            self._check_mixture(index, mixture, sources)

        return {
            "mixture": mixture,
            "sources": sources,
            "track": track_dir.name,
        }

    def _check_mixture(
        self, index: int, mixture: torch.Tensor, sources: torch.Tensor
    ) -> None:
        """Предупреждает, если mixture заметно отличается от суммы дорожек."""
        diff = (mixture - sources.sum(dim=0)).abs().max().item()
        if diff > 0.05:
            self._warn_once(
                ("mix", self.tracks[index].name),
                f"{self.describe_track(index)}: mixture не равна сумме "
                f"дорожек (расхождение {diff:.3f} при норме 1.0). "
                f"Проверь, что дорожки из одного трека.",
            )


# ---------------------------------------------------------------------------
# ПОРЯДОК ДОРОЖЕК: ДАННЫЕ vs МОДЕЛЬ
# ---------------------------------------------------------------------------

def reorder_sources_to_model_order(
    sources: torch.Tensor,
    data_order: Sequence[str],
    model_order: Sequence[str],
) -> torch.Tensor:
    """
    Переставляет дорожки из порядка данных в порядок выхода модели.

    ЗАЧЕМ ЭТО НУЖНО (мы проверили это в check_demucs.py):
        Модель htdemucs выдаёт каналы в порядке
            [drums, bass, other, vocals]
        а наши файлы и config.py лежат в порядке
            [vocals, drums, bass, other]

        Если не переставить, модель будет сравнивать «свой вокал»
        с «чужими барабанами» - обучение пойдёт не туда, и метрики
        покажут красивую, но бессмысленную цифру.

    Аргументы:
        sources     - тензор (N, 2, L) - дорожки в порядке data_order
        data_order  - как они лежат у нас: ["vocals", "drums", ...]
        model_order - как их выдаёт модель: ["drums", "bass", ...]

    Возвращает тензор (N, 2, L) в порядке model_order.
    """
    if sorted(data_order) != sorted(model_order):
        raise ValueError(
            f"Наборы дорожек не совпадают.\n"
            f"  данные:  {list(data_order)}\n"
            f"  модель:  {list(model_order)}\n"
            f"  Переставить нельзя - не хватает или лишние дорожки."
        )

    if list(data_order) == list(model_order):
        return sources

    permutation = [list(data_order).index(name) for name in model_order]
    return sources[permutation].contiguous()


# ---------------------------------------------------------------------------
# ЗАГРУЗЧИКИ ДЛЯ ОБУЧЕНИЯ
# ---------------------------------------------------------------------------

def get_custom_loaders(
    config: Any,
    segment_length: Optional[int] = None,
    sample_rate: Optional[int] = None,
    num_workers: Optional[int] = None,
    check_mixture: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Делает два DataLoader-а: для обучения и для проверки.

    Возвращает пару (train_loader, valid_loader).

    Настройки берутся из config:
        batch_size   - сколько треков за раз
        num_workers  - сколько процессов читают файлы
        seed         - для воспроизводимого перемешивания

    Про num_workers на Windows: процессы создаются заново при каждом запуске
    и заметно тормозят старт. Если видишь ошибку про multiprocessing или
    подвисание - поставь num_workers: int = 0 в config.py.
    """
    root = Path(getattr(config, "custom_data_path_abs")
                or Path(getattr(config, "custom_data_path")).resolve())
    segment_length = int(segment_length or getattr(config, "segment_length", 480000))
    sample_rate = int(sample_rate or getattr(config, "sample_rate", 44100))
    batch_size = int(getattr(config, "batch_size", 8))
    workers = int(getattr(config, "num_workers", 0) if num_workers is None else num_workers)
    sources = list(getattr(config, "sources", ["vocals", "drums", "bass", "other"]))

    # На видеокарте закреплённая память заметно ускоряет передачу данных.
    # На CPU она только мешает, поэтому там выключаем.
    device_name = str(getattr(config, "device", "cpu"))
    use_cuda = device_name.startswith("cuda")
    pin_memory = use_cuda and workers > 0

    train_set = CustomDataset(
        root_dir=str(root),
        split="train",
        segment_length=segment_length,
        sample_rate=sample_rate,
        sources=sources,
        check_mixture=check_mixture,
    )
    valid_set = CustomDataset(
        root_dir=str(root),
        split="valid",
        segment_length=segment_length,
        sample_rate=sample_rate,
        sources=sources,
        check_mixture=check_mixture,
    )

    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        # persistent_workers нельзя при workers=0 - он там не имеет смысла
        "persistent_workers": workers > 0,
    }

    # Перемешиваем только обучение: иначе каждый батч будет тем же самым,
    # и проверка станет бесполезной (а на GPU - вредно для скорости).
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=False, **common
    )
    valid_loader = DataLoader(
        valid_set, shuffle=False, drop_last=False, **common
    )

    return train_loader, valid_loader


# ---------------------------------------------------------------------------
# ПРОВЕРКА ПРИ ЗАПУСКЕ
# ---------------------------------------------------------------------------

def self_test(
    split: str = "train",
    root: Optional[str] = None,
    batch_size: int = 2,
    check_mixture: bool = True,
) -> int:
    """
    Проверяет, что данные читаются и имеют правильную форму.

    Создаёт датасет, берёт один батч и печатает формы. Именно то, что
    просят в ТЗ: mixture и sources.
    """
    setup_console()

    print("=" * 72)
    print(" demucs_custom / ПРОВЕРКА ДАННЫХ")
    print("=" * 72)
    print(f"  Папка с данными: {root or config_root()}")

    dataset = CustomDataset(
        root_dir=root or config_root(),
        split=split,
        segment_length=project_config.segment_length,
        sample_rate=project_config.sample_rate,
        sources=list(project_config.sources),
        check_mixture=check_mixture,
    )

    print()
    print("  Формы одного трека (как отдаёт __getitem__):")
    sample = dataset[0]
    print(f"    mixture : {tuple(sample['mixture'].shape)}")
    print(f"    sources : {tuple(sample['sources'].shape)}  "
          f"порядок {list(project_config.sources)}")
    print(f"    трек    : {sample['track']}")
    print(f"    длина   : {len(dataset)} треков")

    print()
    print(f"  Загружаю батч (batch_size = {batch_size}):")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"    mixture : {tuple(batch['mixture'].shape)}   "
          f"ожидалось ({batch_size}, 2, {project_config.segment_length})")
    print(f"    sources : {tuple(batch['sources'].shape)}   "
          f"ожидалось ({batch_size}, 4, 2, {project_config.segment_length})")
    print(f"    треки   : {list(batch['track'])}")

    # --- Честная проверка, а не просто "не упало" ----------------------
    problems: List[str] = []
    mixture, sources = batch["mixture"], batch["sources"]

    if tuple(mixture.shape) != (batch_size, 2, project_config.segment_length):
        problems.append(f"неверная форма mixture: {tuple(mixture.shape)}")
    if tuple(sources.shape) != (batch_size, 4, 2, project_config.segment_length):
        problems.append(f"неверная форма sources: {tuple(sources.shape)}")
    if mixture.dtype != torch.float32:
        problems.append(f"mixture имеет тип {mixture.dtype}, а не float32")
    for tensor, name in ((mixture, "mixture"), (sources, "sources")):
        if not torch.isfinite(tensor).all():
            problems.append(f"в {name} есть NaN или бесконечности")
        if float(tensor.abs().max()) > 1.0:
            problems.append(f"в {name} есть перегруз (пик "
                            f"{float(tensor.abs().max()):.2f})")

    # Порядок дорожек по умолчанию должен совпадать с config.
    expected_order = list(project_config.sources)
    if sources.shape[1] != len(expected_order):
        problems.append(f"дорожек {sources.shape[1]}, а в config "
                        f"{len(expected_order)}: {expected_order}")

    print()
    print("-" * 72)
    if problems:
        print(" Проблемы:")
        for problem in problems:
            print(f"   - {problem}")
        return 1

    print(" Всё в порядке: файлы читаются, формы верные, NaN нет,")
    print(" перегруз нет, порядок дорожек соответствует config.")
    print("-" * 72)
    return 0


def config_root() -> str:
    """Путь к папке с данными из config.py (строкой, для читаемости вывода)."""
    return str(project_config.custom_data_path_abs)


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: проверка и подготовка данных",
    )
    parser.add_argument("--data-path", default=None,
                        help="папка с данными (по умолчанию из config.py)")
    parser.add_argument("--split", default="train", choices=["train", "valid"],
                        help="какую выборку проверять (по умолчанию train)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="размер батча для проверки (по умолчанию 2)")
    parser.add_argument("--loaders", action="store_true",
                        help="собрать настоящие загрузчики через "
                             "get_custom_loaders (так же, как для обучения)")
    parser.add_argument("--no-mixture-check", dest="check_mixture",
                        action="store_false",
                        help="не проверять, что mixture равна сумме дорожек")
    parser.add_argument("--make-examples", type=int, nargs="?", const=3,
                        metavar="N",
                        help="если папка пуста, создать N примеров-треков "
                             "через prepare_data_structure.py")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа."""
    setup_console()
    args = parse_args(argv)

    if args.data_path:
        project_config.custom_data_path = args.data_path

    root = config_root()

    # Если данных нет и попросили - создаём примеры, чтобы было что проверить.
    split_dir = Path(root) / args.split
    if args.make_examples is not None and not any(split_dir.glob(TRACK_PREFIX + "*")):
        count = max(1, int(args.make_examples))
        print(f"Данных в {split_dir} нет, создаю примеры: "
              f"{count} train + {count} valid")
        try:
            from prepare_data_structure import create_example_structure
        except ImportError:
            print("[!] Не найден prepare_data_structure.py - он должен лежать "
                  "в одной папке с этим файлом.")
            return 1
        result = create_example_structure(num_train=count, num_valid=count)
        if result.get("status") == "error":
            return 1
        print()

    if args.loaders:
        print("=" * 72)
        print(" demucs_custom / ЗАГРУЗЧИКИ ДЛЯ ОБУЧЕНИЯ")
        print("=" * 72)
        try:
            train_loader, valid_loader = get_custom_loaders(
                project_config, check_mixture=args.check_mixture
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[!] {exc}")
            return 1
        batch = next(iter(train_loader))
        print(f"  train: {len(train_loader.dataset)} треков, "
              f"batch_size={train_loader.batch_size}, "
              f"перемешивание={train_loader.sampler.__class__.__name__}")
        print(f"  valid: {len(valid_loader.dataset)} треков, "
              f"batch_size={valid_loader.batch_size}")
        print(f"  батч mixture: {tuple(batch['mixture'].shape)}")
        print(f"  батч sources: {tuple(batch['sources'].shape)}")
        return 0

    try:
        return self_test(
            split=args.split,
            root=root,
            batch_size=args.batch_size,
            check_mixture=args.check_mixture,
        )
    except FileNotFoundError as exc:
        # Не показываем пользователю трассировку: сообщение в exc уже
        # написано по-человечески, с подсказкой что делать дальше.
        print()
        print(f"[!] {exc}")
        return 1
    except (ValueError, RuntimeError) as exc:
        print()
        print(f"[!] Данные не прочитались: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
