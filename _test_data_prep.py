# -*- coding: utf-8 -*-
"""
Неформальный тест data_prep.py: проверяем краевые случаи, которые ломаются
на реальных данных. Запуск:  python _test_data_prep.py
"""
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import torch

from data_prep import (
    CustomDataset,
    read_wav,
    reorder_sources_to_model_order,
    resample_audio,
    to_stereo,
    fit_length,
)

TMP = Path(r"C:\Users\User\AppData\Local\Temp\opencode\dp_test")
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


def write_wav(path, data, sr, width=2, channels=2, float_fmt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if float_fmt:
        pcm = np.ascontiguousarray(data.T.astype("<f4")).tobytes()
        width = 4
    elif width == 1:
        pcm = np.ascontiguousarray(
            ((np.clip(data, -1, 1) * 127) + 128).astype(np.uint8).T).tobytes()
    elif width == 2:
        pcm = np.ascontiguousarray(
            (np.clip(data, -1, 1) * 32767).astype("<i2").T).tobytes()
    elif width == 3:
        b = (np.clip(data, -1, 1) * 8388607).astype(np.int32)
        u = np.ascontiguousarray(b.T).view(np.uint8).reshape(b.T.shape[0], b.T.shape[1], 4)
        pcm = np.ascontiguousarray(u[:, :, :3]).tobytes()
    elif width == 4:
        pcm = np.ascontiguousarray(
            (np.clip(data, -1, 1) * 2147483647).astype("<i4").T).tobytes()
    else:
        raise ValueError(width)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(pcm)


def sine(freq, sr, seconds, channels=2, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([mono] * channels) if channels > 1 else mono[None]


def make_track(root, split, name, sr, seconds, channels=2, width=2,
               float_fmt=False, stem_seconds=None, seed=0):
    """Создаёт трек: mixture = сумма дорожек."""
    stem_seconds = seconds if stem_seconds is None else stem_seconds
    d = root / split / name
    stems = {}
    rng = np.random.default_rng(seed)
    for i, s in enumerate(["vocals", "drums", "bass", "other"]):
        n = int(sr * stem_seconds)
        t = np.arange(n) / sr
        data = (0.2 * np.sin(2 * np.pi * (300 + 100 * i) * t)).astype(np.float32)
        if channels == 1:
            data = data[None]
        else:
            data = np.stack([data, data * 0.9])
        stems[s] = data
        write_wav(d / f"{s}.wav", data, sr, width=width, channels=channels,
                  float_fmt=float_fmt)
    mix = sum(stems.values())
    write_wav(d / "mixture.wav", mix, sr, width=width, channels=channels,
              float_fmt=float_fmt)
    return d


print("=" * 72)
print("1. ЧТЕНИЕ РАЗНЫХ БИТНОСТЕЙ")
print("=" * 72)
for width, name, ff in [(1, "8 бит", False), (2, "16 бит", False),
                        (3, "24 бит", False), (4, "32 бит int", False),
                        (4, "32 бит float", True)]:
    p = TMP / f"bits_{width}_{ff}.wav"
    data = sine(440, 44100, 0.1)
    write_wav(p, data, 44100, width=width, float_fmt=ff)
    try:
        a, sr = read_wav(p)
        ok = a.shape == (2, 4410) and abs(float(a.abs().max()) - 0.5) < 0.05
        check(f"{name}: форма {tuple(a.shape)}, пик {float(a.abs().max()):.3f}", ok, f"sr={sr}")
    except Exception as e:
        check(f"{name}: {type(e).__name__}: {e}", False)

print()
print("=" * 72)
print("2. РЕСЕМПЛИНГ (главное: без сдвига высоты)")
print("=" * 72)
for src_sr in (22050, 48000, 96000, 8000):
    x = sine(1000, src_sr, 1.0, channels=1, amp=0.5)
    t = torch.from_numpy(x)
    y = resample_audio(t, src_sr, 44100)
    exp = int(44100)
    # Проверяем высоту: считаем, сколько раз сигнал пересёк ноль за секунду
    crossings = int(((y[0, :-1] * y[0, 1:]) < 0).sum())
    ok_len = abs(y.shape[1] - exp) <= 2
    ok_pitch = abs(crossings - 2000) < 60
    check(f"{src_sr} -> 44100 Гц: длина {y.shape[1]} (ждали ~{exp}), "
          f"пересечений нуля {crossings} (1000 Гц -> 2000)",
          ok_len and ok_pitch)
check("та же частота - данные не трогаются",
      torch.equal(resample_audio(torch.zeros(2, 100), 44100, 44100),
                  torch.zeros(2, 100)))

print()
print("=" * 72)
print("3. МОНО / МНОГОКАНАЛЬНОСТЬ")
print("=" * 72)
check("моно (1, N) -> стерео (2, N)",
      tuple(to_stereo(torch.zeros(1, 50)).shape) == (2, 50))
check("моно: оба канала одинаковые",
      torch.equal(to_stereo(torch.ones(1, 10))[0], to_stereo(torch.ones(1, 10))[1]))
check("стерео не меняется", tuple(to_stereo(torch.zeros(2, 50)).shape) == (2, 50))
check("5 каналов -> берём первые 2", tuple(to_stereo(torch.zeros(5, 50)).shape) == (2, 50))
check("одномерный (N,) -> стерео (2, N)", tuple(to_stereo(torch.zeros(50)).shape) == (2, 50))

print()
print("=" * 72)
print("4. ДЛИНЫ: КОРОЧЕ / ДЛИННЕ / РОВНО")
print("=" * 72)
check("короче -> нулями до 100", tuple(fit_length(torch.ones(2, 40), 100).shape) == (2, 100))
check("короче: хвост = нули",
      bool((fit_length(torch.ones(2, 40), 100)[:, 40:] == 0).all()))
check("короче: начало не тронуто",
      bool((fit_length(torch.ones(2, 40), 100)[:, :40] == 1).all()))
check("длиннее -> обрезка до 100", tuple(fit_length(torch.ones(2, 400), 100).shape) == (2, 100))
check("ровно -> без изменений",
      bool(torch.equal(fit_length(torch.ones(2, 100), 100), torch.ones(2, 100))))

print()
print("=" * 72)
print("5. ДАТАСЕТ: разные частоты, моно, разная длина в треке")
print("=" * 72)
if TMP.exists():
    shutil.rmtree(TMP)

# a) 22050 Гц, моно, короче сегмента -> ресемпл + стерео + нули
make_track(TMP, "train", "track_001", sr=22050, seconds=3.0, channels=1)
# b) 48000 Гц, длиннее сегмента -> ресемпл + обрезка
make_track(TMP, "train", "track_002", sr=48000, seconds=15.0)
# c) 44100, дорожки РАЗНОЙ длины -> обрезка до минимальной
d = make_track(TMP, "train", "track_003", sr=44100, seconds=5.0)
# делаем bass короче остальных
write_wav(d / "bass.wav", sine(200, 44100, 2.0), 44100)
# d) трек с лишним mp3 -> должен быть пропущен с предупреждением
d4 = make_track(TMP, "train", "track_004", sr=44100, seconds=5.0)
(d4 / "notes.txt").write_text("hello", encoding="utf-8")
# e) трек БЕЗ файла drums -> пропустить с предупреждением
d5 = make_track(TMP, "train", "track_005", sr=44100, seconds=5.0)
(d5 / "drums.wav").unlink()
# f) папка не-track -> игнорировать
(TMP / "train" / "random_folder").mkdir(parents=True, exist_ok=True)
(TMP / "train" / "random_folder" / "mixture.wav").write_bytes(b"nope")
# valid для get_custom_loaders
make_track(TMP, "valid", "track_101", sr=44100, seconds=5.0)
make_track(TMP, "valid", "track_102", sr=22050, seconds=5.0)

ds = CustomDataset(root_dir=str(TMP), split="train", segment_length=480000,
                   sample_rate=44100, check_mixture=True, verbose=True)
print()
check("найдено 4 трека из 6 папок", len(ds) == 4,
      f"найдено: {[t.name for t in ds.tracks]}")
check("track_005 (нет drums.wav) исключён", "track_005" not in [t.name for t in ds.tracks])
check("track_004 (с notes.txt) остался: пропускается файл, а не трек",
      "track_004" in [t.name for t in ds.tracks])
check("track_004: про notes.txt сказано предупреждением",
      any("notes.txt" in w for w in ds._scan_warnings))
check("папка random_folder проигнорирована",
      "random_folder" not in [t.name for t in ds.tracks])

a = ds[0]
check("getitem: mixture (2, 480000)", tuple(a["mixture"].shape) == (2, 480000),
      str(tuple(a["mixture"].shape)))
check("getitem: sources (4, 2, 480000)", tuple(a["sources"].shape) == (4, 2, 480000),
      str(tuple(a["sources"].shape)))
check("getitem: dtype float32", a["mixture"].dtype == torch.float32)
# Пик суммы 4 разных синусоид 0.744 (не 0.8): пики не совпадают по времени.
# Проверяем, что ресемплинг ничего не потерял - для сравнения считаем
# тот же пик без ресемплинга.
_t = np.arange(int(22050 * 3.0)) / 22050
_expected_peak = float(np.abs(sum(
    0.2 * np.sin(2 * np.pi * f * _t) for f in (300, 400, 500, 600))).max())
check("трек 22050 Гц моно: ресемпл + стерео + нули, громкость не потеряна",
      a["track"] == "track_001"
      and abs(float(a["mixture"].abs().max()) - _expected_peak) < 0.01,
      f"пик {float(a['mixture'].abs().max()):.4f} "
      f"(без ресемплинга было бы {_expected_peak:.4f})")
check("трек короче сегмента дополнен нулями",
      bool((a["mixture"][:, int(22050 * 3.0) * 2:] == 0).all()))

b = ds[1]
check("трек 15 сек обрезан до 480000", tuple(b["mixture"].shape) == (2, 480000))

c = ds[2]
check("дорожки разной длины: все по 480000",
      tuple(c["sources"].shape) == (4, 2, 480000), str(tuple(c["sources"].shape)))

# Предупреждение о разных длинах должно прозвучать ОДИН раз, а не на
# каждом чтении: при обучении трек читается десятки раз.
# Берём СВЕЖИЙ датасет, потому что предыдущий ds[2] уже потратил предупреждение.
import io
from contextlib import redirect_stdout

_fresh = CustomDataset(root_dir=str(TMP), split="train", segment_length=480000,
                       sample_rate=44100, check_mixture=True, verbose=True)
_buf = io.StringIO()
with redirect_stdout(_buf):
    for _ in range(5):
        _fresh[2]
_repeat_lines = [line for line in _buf.getvalue().splitlines()
                 if "дорожки разной длины" in line]
check("предупреждение о разных длинах печатается один раз, а не 5",
      len(_repeat_lines) == 1, f"строк с предупреждением: {len(_repeat_lines)}")

print()
print("=" * 72)
print("6. ПОРЯДОК ДОРОЖЕК")
print("=" * 72)
src = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
data_order = ["vocals", "drums", "bass", "other"]
model_order = ["drums", "bass", "other", "vocals"]
out = reorder_sources_to_model_order(src, data_order, model_order)
check("перестановка: drums(1) на место 0", bool(torch.equal(out[0], src[1])))
check("перестановка: vocals(0) на место 3", bool(torch.equal(out[3], src[0])))
check("одинаковые порядки - без изменений",
      bool(torch.equal(reorder_sources_to_model_order(src, data_order, data_order), src)))
try:
    reorder_sources_to_model_order(src, data_order, ["a", "b"])
    check("разные наборы -> ошибка", False)
except ValueError:
    check("разные наборы -> понятная ошибка", True)

print()
print("=" * 72)
print("7. ОШИБКИ (должны быть понятными)")
print("=" * 72)
for root, split, label in [
    (TMP / "нет_такой_папки", "train", "нет папки данных"),
    (TMP, "нет_такого_split", "нет папки выборки"),
]:
    try:
        CustomDataset(root_dir=str(root), split=split, verbose=False)
        check(f"{label}: сообщение об ошибке", False, "исключения не было!")
    except FileNotFoundError as e:
        check(f"{label}: FileNotFoundError с подсказкой", "не найдена" in str(e).lower(),
              str(e).splitlines()[0])
    except Exception as e:
        check(f"{label}: {type(e).__name__}", False, str(e))

empty = TMP / "empty"
(empty / "train").mkdir(parents=True, exist_ok=True)
try:
    CustomDataset(root_dir=str(empty), split="train", verbose=False)
    check("пустая папка: сообщение", False)
except FileNotFoundError as e:
    check("пустая папка: FileNotFoundError с подсказкой", "нет ни одного трека" in str(e))

print()
print("=" * 72)
print("8. ЗАГРУЗЧИКИ (get_custom_loaders)")
print("=" * 72)
from data_prep import get_custom_loaders


class FakeConfig:
    custom_data_path = str(TMP)
    custom_data_path_abs = str(TMP)
    segment_length = 480000
    sample_rate = 44100
    batch_size = 2
    num_workers = 0
    device = "cpu"
    sources = ["vocals", "drums", "bass", "other"]


tl, vl = get_custom_loaders(FakeConfig())
check("train_loader создан", tl is not None and len(tl.dataset) == 4,
      f"треков в train: {len(tl.dataset)}")
check("valid_loader создан", vl is not None and len(vl.dataset) == 2)
check("batch_size из config = 2", tl.batch_size == 2)
check("train перемешивается", "RandomSampler" in str(type(tl.sampler).__name__),
      type(tl.sampler).__name__)
check("valid НЕ перемешивается", "Sequential" in str(type(vl.sampler).__name__),
      type(vl.sampler).__name__)
batch = next(iter(tl))
check("батч mixture (2, 2, 480000)", tuple(batch["mixture"].shape) == (2, 2, 480000),
      str(tuple(batch["mixture"].shape)))
check("батч sources (2, 4, 2, 480000)", tuple(batch["sources"].shape) == (2, 4, 2, 480000),
      str(tuple(batch["sources"].shape)))
# Проверяем, что перемешивание реально работает. Сравниваем НЕ два раза
# подряд (при 4 треках и батче 2 порядок может случайно совпасть, шанс 1 из 24),
# а сразу много прогонов: порядков должно быть больше одной.
_orders = set()
for _ in range(12):
    _orders.add(tuple(b["track"][0] for b in tl))
check("порядок треков меняется при перемешивании", len(_orders) > 1,
      f"{len(_orders)} разных порядков из 12 прогонов")

shutil.rmtree(TMP, ignore_errors=True)
print()
print("=" * 72)
if FAIL:
    print(f"ПРОВАЛЕНО {len(FAIL)}: " + "; ".join(FAIL))
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
