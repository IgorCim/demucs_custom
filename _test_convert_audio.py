# -*- coding: utf-8 -*-
"""
Автотесты convert_audio.py.

Проверяем, что конвертация даёт ПРАВИЛЬНЫЙ звук, а не просто не падает:
частота, каналы, битность, сохранение высоты тона.

Соглашение в тесте: аудио всегда (каналы, N) - как в torch.
soundfile ждёт наоборот, (N, каналы), поэтому запись идёт через sfw().

Запуск: python _test_convert_audio.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from convert_audio import batch_convert, convert_to_wav, peek_wav_info
from data_prep import CustomDataset

TMP = Path(r"C:\Users\User\AppData\Local\Temp\opencode\conv_test")
SRC = TMP / "input"
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


def tone(freq, sr, seconds, amp=0.4, channels=2):
    """Тон частотой freq Гц. Возвращает (каналы, N)."""
    t = np.arange(int(sr * seconds)) / sr
    mono = amp * np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return mono[None, :]
    return np.stack([mono, mono * 0.8], axis=0)


def sfw(path, data, sr, fmt="WAV", subtype="PCM_16"):
    """soundfile.write с переносом оси: (каналы, N) -> (N, каналы)."""
    arr = np.asarray(data, dtype="float32")
    if arr.ndim == 1:
        arr = arr[None, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt.upper() in ("OGG", "VORBIS"):
        subtype = "VORBIS"   # OGG умеет только Vorbis, PCM туда не пустить
    sf.write(str(path), arr.T, sr, format=fmt, subtype=subtype)


def make_mp3(path, seconds=1.0, sr=44100):
    """Настоящий mp3 из пустых MPEG-1 Layer III кадров (тишина)."""
    frame = bytes([0xFF, 0xFB, 0x90, 0xC4]) + bytes(144 * 128000 // sr - 4)
    path.write_bytes(frame * max(1, int(seconds * sr / 1152)))


def dominant_freq(audio_1d, sr):
    """Частота самого громкого пика в спектре."""
    window = np.hanning(len(audio_1d))
    spec = np.abs(np.fft.rfft(audio_1d * window))
    return float(np.argmax(spec)) * sr / len(audio_1d)


if TMP.exists():
    shutil.rmtree(TMP)
SRC.mkdir(parents=True)

print("=" * 72)
print("1. ОДИН ФАЙЛ: 22050 Гц моно -> 44100 Гц стерео WAV")
print("=" * 72)
sfw(SRC / "mono-22050.wav", tone(440, 22050, 2.0, channels=1), 22050)

out = TMP / "out1" / "mono-22050.wav"
seconds = convert_to_wav(str(SRC / "mono-22050.wav"), str(out))
info = peek_wav_info(out)
check("частота 44100 Гц", info["sample_rate"] == 44100, str(info))
check("2 канала (из моно)", info["channels"] == 2, str(info))
check("16 бит", info["bit_depth"] == 16, str(info))
check("длительность вернулась в секундах", abs(seconds - 2.0) < 0.05,
      f"{seconds:.3f} сек")
check("отсчётов как в 2 сек", abs(info["frames"] / 44100 - 2.0) < 0.05,
      f"{info['frames']} отсчётов")

data, _ = sf.read(str(out), dtype="float32", always_2d=True)
measured = dominant_freq(data[5000:-5000, 0], 44100)
check("высота тона не поехала: 440 Гц -> ~440 Гц", abs(measured - 440) < 8,
      f"измерено {measured:.1f} Гц")
check("оба канала заполнены (моно развёрнуто в оба уха)",
      float(np.abs(data[:, 0]).max()) > 0.2 and float(np.abs(data[:, 1]).max()) > 0.2,
      f"пики {float(np.abs(data[:, 0]).max()):.3f} / {float(np.abs(data[:, 1]).max()):.3f}")

print()
print("=" * 72)
print("2. РАЗНЫЕ ФОРМАТЫ ЧТЕНИЕ -> WAV")
print("=" * 72)
sfw(SRC / "стерео.flac", tone(1000, 44100, 1.0), 44100, fmt="FLAC")
sfw(SRC / "стерео.ogg", tone(1000, 44100, 1.0), 44100, fmt="OGG")
sfw(SRC / "стерео48k.wav", tone(1000, 48000, 1.0), 48000)
make_mp3(SRC / "тишина.mp3", 1.0)

for name, expect_freq in [("стерео.flac", 1000), ("стерео.ogg", 1000),
                          ("стерео48k.wav", 1000), ("тишина.mp3", None)]:
    dst = TMP / "out2" / (Path(name).stem + ".wav")
    try:
        sec = convert_to_wav(str(SRC / name), str(dst))
        i = peek_wav_info(dst)
        ok = i["sample_rate"] == 44100 and i["channels"] == 2
        extra = ""
        if expect_freq:
            d, _ = sf.read(str(dst), dtype="float32", always_2d=True)
            freq = dominant_freq(d[5000:-5000, 0], 44100)
            ok = ok and abs(freq - expect_freq) < 8
            extra = f"тон {freq:.0f} Гц"
        else:
            extra = "mp3 = тишина, тон не проверяем"
        check(f"{name} -> {i['sample_rate']} Гц, {i['channels']} кн, {sec:.2f} сек",
              ok, extra)
    except Exception as exc:
        check(f"{name}: {type(exc).__name__}: {exc}", False)

print()
print("=" * 72)
print("3. БИТНОСТЬ 16 и 32 (IEEE float)")
print("=" * 72)
# один и тот же исходник в 16 и в 32 бита, чтобы честно сравнить размер
same = SRC / "стерео.flac"
p16 = TMP / "out3" / "same16.wav"
p32 = TMP / "out3" / "same32.wav"
convert_to_wav(str(same), str(p16), bit_depth="16")
convert_to_wav(str(same), str(p32), bit_depth="32")

i16, i32 = peek_wav_info(p16), peek_wav_info(p32)
check("16-битный помечен как 16 бит", i16["bit_depth"] == 16, str(i16))
check("32-битный помечен как 32 бита", i32["bit_depth"] == 32, str(i32))
check("32-битный - формат IEEE float, а не PCM",
      sf.info(str(p32)).subtype == "FLOAT", sf.info(str(p32)).subtype)
check("16-битный - PCM, а не float",
      sf.info(str(p16)).subtype == "PCM_16", sf.info(str(p16)).subtype)
d32, _ = sf.read(str(p32), dtype="float32", always_2d=True)
f32 = dominant_freq(d32[5000:-5000, 0], 44100)
check("тон в 32-битном не исказился", abs(f32 - 1000) < 8, f"{f32:.0f} Гц")
check("32-битный файл ровно вдвое больше 16-битного",
      abs(p32.stat().st_size / p16.stat().st_size - 2.0) < 0.01,
      f"{p32.stat().st_size} против {p16.stat().st_size}")
try:
    convert_to_wav(str(same), str(TMP / "out3" / "x.wav"), bit_depth="8")
    check("несуществующая битность -> ошибка", False)
except ValueError:
    check("несуществующая битность -> ошибка", True)

print()
print("=" * 72)
print("4. СОГЛАСОВАННОСТЬ (обязательное правило обучения)")
print("=" * 72)
track = TMP / "dataset" / "train" / "track_001"
stems = {}
for i, s in enumerate(["vocals", "drums", "bass", "other"]):
    d = tone(200 + 150 * i, 44100, 3.0, amp=0.2)
    sfw(SRC / f"дорожка-{s}.flac", d, 44100, fmt="FLAC")
    stems[s] = d
sfw(SRC / "дорожка-mixture.flac", sum(stems.values()), 44100, fmt="FLAC")

conv = TMP / "out4"
batch_convert(str(SRC), str(conv), sample_rate=44100)
track.mkdir(parents=True, exist_ok=True)
for s in ["vocals", "drums", "bass", "other", "mixture"]:
    shutil.copy(conv / f"дорожка-{s}.wav", track / f"{s}.wav")

ds = CustomDataset(root_dir=str(TMP / "dataset"), split="train", verbose=False)
item = ds[0]
diff = float((item["mixture"] - item["sources"].sum(dim=0)).abs().max())
check("CustomDataset читает результат конвертации",
      tuple(item["mixture"].shape) == (2, 480000)
      and tuple(item["sources"].shape) == (4, 2, 480000),
      f"mixture {tuple(item['mixture'].shape)}, sources {tuple(item['sources'].shape)}")
check("mixture равна сумме дорожек и после конвертации", diff < 0.01,
      f"расхождение {diff:.5f} (допуск 0.01)")

print()
print("=" * 72)
print("5. BATCH: прогресс, пропуски, ошибки, папки")
print("=" * 72)

batch_in = TMP / "bin"
(batch_in / "подпапка").mkdir(parents=True)
sfw(batch_in / "а.wav", tone(440, 44100, 1.0), 44100)
sfw(batch_in / "б.flac", tone(440, 22050, 1.0), 22050, fmt="FLAC")
sfw(batch_in / "в.ogg", tone(440, 48000, 1.0), 48000, fmt="OGG")
sfw(batch_in / "подпапка" / "г.wav", tone(440, 44100, 1.0), 44100)
(batch_in / "битый.mp3").write_bytes(b"\xff\xfb\x90" + b"\x00" * 50)
(batch_in / "не-аудио.txt").write_text("просто текст", encoding="utf-8")

res = batch_convert(str(batch_in), str(TMP / "bout"))
total_seen = res["converted"] + res["skipped"] + res["failed"]
check("txt не считается аудиофайлом", total_seen == 4, f"всего {total_seen}")
check("битый файл посчитан как ошибка", res["failed"] == 1, f"ошибок {res['failed']}")
# а.wav уже 44100 стерео 16 бит -> честно пропускается, конвертировать нечего
check("готовый wav пропущен, а flac и ogg сконвертированы",
      res["converted"] == 2 and res["skipped"] == 1,
      f"сконвертировано {res['converted']}, пропущено {res['skipped']}")
check("вложенная папка НЕ заходила без --recursive",
      not (TMP / "bout" / "г.wav").exists())

res2 = batch_convert(str(batch_in), str(TMP / "bout2"), recursive=True)
found2 = res2["converted"] + res2["skipped"] + res2["failed"]
check("с --recursive вложенный файл тоже попал в обход",
      found2 == 5, f"найдено {found2} из 5 (без --recursive было 4)")

res3 = batch_convert(str(batch_in), str(TMP / "bout"))
check("повторный запуск ничего не перезаписал", res3["converted"] == 0,
      f"сконвертировано {res3['converted']}, пропущено {res3['skipped']}")
res4 = batch_convert(str(batch_in), str(TMP / "bout"), force=True)
check("--force перезаписывает", res4["converted"] == 3, f"сконвертировано {res4['converted']}")

res5 = batch_convert(str(TMP / "вообще-нет"), str(TMP / "новый-выход"))
check("нет входной папки -> создана, 0 ошибок",
      res5["failed"] == 0 and res5["converted"] == 0)
check("выходная папка создана", (TMP / "новый-выход").is_dir())

print()
print("=" * 72)
print("6. КОМАНДНАЯ СТРОКА (как в ТЗ: python convert_audio.py input/ output/)")
print("=" * 72)
cli_in, cli_out = TMP / "cli-in", TMP / "cli-out"
cli_in.mkdir(parents=True)
sfw(cli_in / "песня.ogg", tone(440, 22050, 1.0), 22050, fmt="OGG")
sfw(cli_in / "песня2.flac", tone(440, 22050, 1.0), 22050, fmt="FLAC")

proc = subprocess.run(
    [sys.executable, "convert_audio.py", str(cli_in), str(cli_out)],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
text = proc.stdout
check("код возврата 0", proc.returncode == 0, f"код {proc.returncode}")
check("напечатано 'Конвертация завершена. Файлы в:'",
      "Конвертация завершена. Файлы в:" in text)
check("напечатан прогресс 'Обработано X/Y файлов'", "Обработано" in text,
      "; ".join(l.strip() for l in text.splitlines() if "Обработано" in l))
check("wav появился в выходной папке", (cli_out / "песня2.wav").is_file())

shutil.rmtree(TMP, ignore_errors=True)
print()
print("=" * 72)
if FAIL:
    print(f"ПРОВАЛЕНО {len(FAIL)}:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
