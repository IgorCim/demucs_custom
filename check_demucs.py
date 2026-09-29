# -*- coding: utf-8 -*-
"""
demucs_custom / check_demucs.py
===============================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Проверяет, что Demucs установлен, скачивается и работает. Прогоняет
    настоящий (случайный) сигнал через настоящую модель и смотрит, что на
    выходе получилось 4 дорожки.

    Это «дымовой тест» перед началом дообучения. Пока он не проходит,
    трогать train_custom.py бессмысленно: если базовая модель не работает,
    непонятно, сломалась наша доработка или проблема была изначально.

ЧТО ПРОВЕРЯЕТ ПО ШАГАМ:
    1. Установлен ли torch.
    2. Установлен ли demucs.
    3. Скачивается и грузится ли модель htdemucs (веса ~80 МБ, кэшируются).
    4. Сколько в модели параметров.
    5. Какая у неё архитектура (список слоёв).
    6. Какие источники она разделяет и в каком ПОРЯДКЕ.
    7. Пропускает ли она случайный стерео-сигнал.
    8. Какой формы получился выход.

ЗАПУСК:
    python check_demucs.py            # полная проверка (рекомендуется)
    python check_demucs.py --quick    # быстрая проверка на коротком сигнале
    python check_demucs.py --help     # все ключи

КОДЫ ВОЗВРАТА (для автоматических проверок):
    0 - всё хорошо
    1 - не хватает библиотек
    2 - модель не скачалась / не загрузилась
    3 - модель упала на прямом прогоне
    4 - неожиданная форма выхода
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Настройки библиотеки HuggingFace делаем ДО её импорта (импорта в этом файле
# нет, он будет внутри функции), иначе будет лишний шум в консоли.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import logging

# HuggingFace пишет предупреждение "unauthenticated requests" - оно
# безобидное (просто совет завести токен), но нам мешает видеть результат.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from config import config, resolve_device


# Коды возврата
EXIT_OK = 0
EXIT_NO_DEPS = 1
EXIT_NO_MODEL = 2
EXIT_FORWARD_FAILED = 3
EXIT_BAD_SHAPE = 4


def line(char: str = "-", width: int = 78) -> str:
    """Горизонтальная линия. Только ASCII-символы, чтобы печаталось в любой консоли."""
    return char * width


def section(title: str) -> None:
    """Печатает заголовок раздела."""
    print()
    print(line("="))
    print(f" {title}")
    print(line("="))


# ---------------------------------------------------------------------------
# ШАГ 1. ПРОВЕРКА БИБЛИОТЕК
# ---------------------------------------------------------------------------

def check_torch():
    """
    Проверяет, что torch установлен.
    Возвращает модуль torch или None.
    """
    try:
        import torch
        return torch
    except ImportError:
        print()
        print("[!] Не установлен torch - это фундамент для любой нейросети.")
        print("    Установи: pip install torch")
        return None


def check_demucs():
    """
    Проверяет, что demucs установлен.
    Возвращает модуль demucs.pretrained или None.
    """
    try:
        from demucs import pretrained
        return pretrained
    except ImportError as exc:
        print()
        print(f"[!] Не установлен demucs ({type(exc).__name__}).")
        print("    Установи: pip install demucs")
        return None


# ---------------------------------------------------------------------------
# ШАГ 2. ЗАГРУЗКА МОДЕЛИ
# ---------------------------------------------------------------------------

class _QuietHuggingFaceStderr:
    """
    Временная «заглушка» для stderr, которая прячет ОДНО безобидное
    предупреждение HuggingFace:

        Warning: You are sending unauthenticated requests to the HF Hub...

    Оно означает лишь «ты не завёл токен на hf.co» и ни на что не влияет.
    Подавлять его нужно аккуратно: всё остальное проходит насквозь,
    а сам stderr обязательно возвращается обратно (try/finally).
    """

    MARKER = "unauthenticated requests"

    def __init__(self, original) -> None:
        self._original = original
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        # Печатаем всё, кроме строк с этим предупреждением.
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if self.MARKER not in line:
                self._original.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            if self.MARKER not in self._buffer:
                self._original.write(self._buffer)
            self._buffer = ""
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        # Всё остальное (isatty, fileno, closed...) - к настоящему stderr.
        return getattr(self._original, name)


def load_model(model_name: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Скачивает и загружает модель Demucs.

    Возвращает пару (модель, текст_ошибки):
      - удалось  -> (модель, None)
      - не удалось -> (None, "человеческий текст ошибки")

    Ошибки разбираем по типам, чтобы сообщение было понятным:
      - нет интернета / не скачался файл
      - нет такой модели
      - не хватило памяти
      - прочие
    """
    from demucs.pretrained import get_model

    print(f"Загружаю модель '{model_name}'...")
    print("  (веса около 80 МБ, скачиваются ОДИН раз и кэшируются на диске)")

    started = time.time()
    original_stderr = sys.stderr
    try:
        sys.stderr = _QuietHuggingFaceStderr(original_stderr)
        model = get_model(model_name)
    except ImportError as exc:
        # Например, не хватает ffmpeg или другого системного пакета.
        return None, (
            f"Не хватает зависимости для загрузки модели: {exc}\n"
            f"    Обычно лечится: pip install -U demucs"
        )
    except MemoryError:
        return None, (
            "Не хватило оперативной памяти при загрузке весов.\n"
            "    Закрой лишние программы и попробуй снова."
        )
    except Exception as exc:
        message = str(exc)
        # Типичные сетевые ошибки прячем за человеческим текстом.
        looks_like_network = any(
            word in message.lower()
            for word in ("connection", "timeout", "network", "resolve",
                         "temporarily", "max retries", "offline")
        )
        if looks_like_network:
            return None, (
                f"Не удалось скачать веса (проблема с интернетом): {message}\n"
                f"    Проверь подключение к сети.\n"
                f"    Если веса уже скачаны, но сеть недоступна, запусти так:\n"
                f"        set HF_HUB_OFFLINE=1"
            )
        return None, (
            f"Не удалось загрузить модель '{model_name}': {type(exc).__name__}: {message}\n"
            f"    Проверь название модели. Доступные: htdemucs, htdemucs_ft,\n"
            f"    htdemucs_6s, demucs, mdx_extra_q"
        )
    finally:
        # Возвращаем stderr на место в любом случае.
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stderr = original_stderr

    print(f"  Модель загружена за {time.time() - started:.1f} сек.")
    return model, None


# ---------------------------------------------------------------------------
# ШАГ 3. ИНФОРМАЦИЯ О МОДЕЛИ
# ---------------------------------------------------------------------------

def count_parameters(model) -> Tuple[int, int]:
    """
    Считает параметры модели.

    Возвращает пару (всего, из них обучаемых).
    Обучаемых важно для нас: при дообучении обычно замораживают большую
    часть сети и учат только верхние слои.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_inner_model(model):
    """
    Достаёт "настоящую" нейросеть из возможной обёртки.

    demucs отдаёт BagOfModels - это контейнер, внутри которого лежат
    одна или несколько моделей (усреднение их результатов даёт лучший
    итог). Нам интересна внутренняя.
    """
    inner_models = getattr(model, "models", None)
    if inner_models:
        try:
            return inner_models[0], len(inner_models)
        except (TypeError, IndexError):
            return model, 0
    return model, 0


def print_model_info(model) -> Dict[str, Any]:
    """
    Печатает всё, что полезно знать о модели, и возвращает
    словарь с ключевыми характеристиками (для дальнейшей проверки).
    """
    inner, models_in_bag = get_inner_model(model)
    total_params, trainable_params = count_parameters(model)

    # Сегмент и частота берём у ВНУТРЕННЕЙ модели: у обёртки BagOfModels
    # они равны None (проверено на demucs 4.1.0).
    inner_segment = getattr(inner, "segment", None)
    inner_samplerate = getattr(inner, "samplerate", None)

    info: Dict[str, Any] = {
        "wrapper_class": type(model).__name__,
        "inner_class": type(inner).__name__,
        "models_in_bag": models_in_bag,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "sources": list(getattr(model, "sources", []) or []),
        "inner_sources": list(getattr(inner, "sources", []) or []),
        "audio_channels": getattr(model, "audio_channels", None),
        "model_samplerate": inner_samplerate,
        "model_segment_seconds": float(inner_segment) if inner_segment is not None else None,
    }

    # --- Класс и обёртка -------------------------------------------------
    print(f"  Обёртка модели      : {info['wrapper_class']}")
    print(f"  Настоящая модель   : {info['inner_class']}")
    if models_in_bag:
        print(f"  Моделей внутри     : {models_in_bag} "
              f"(результаты усредняются - это даёт лучшее качество)")

    # --- Параметры --------------------------------------------------------
    print()
    print("  ПАРАМЕТРЫ:")
    print(f"    всего            : {total_params:,} шт "
          f"({total_params / 1_000_000:.1f} млн)")
    print(f"    из них обучаемых : {trainable_params:,} шт "
          f"({trainable_params / 1_000_000:.1f} млн)")
    print(f"    вес на диске    : ~{total_params * 4 / 1024 / 1024:.0f} МБ "
          f"(4 байта на число)")

    # --- Источники (САМОЕ ВАЖНОЕ) ----------------------------------------
    print()
    print("  ИСТОЧНИКИ (порядок каналов на выходе):")
    sources = info["sources"] or info["inner_sources"]
    for index, source in enumerate(sources):
        print(f"    канал {index}: {source}")

    if sources and sources != config.sources:
        # Это не поломка, но важное предупреждение для нашего проекта:
        # дорожки надо переименовывать по реальному порядку модели,
        # иначе вокал сохранится как барабаны.
        print()
        print("  [!] ВНИМАНИЕ: порядок источников в модели не совпадает")
        print(f"      с config.sources ({', '.join(config.sources)}).")
        print(f"      Реальный порядок модели: {', '.join(sources)}")
        print("      В train_custom.py и inference.py нужно переименовывать")
        print("      дорожки по model.sources, иначе они перепутаются.")

    # --- Прочее -----------------------------------------------------------
    seconds = info["model_segment_seconds"]
    model_samples = None
    if seconds is not None and info["model_samplerate"]:
        model_samples = int(info["model_samplerate"] * seconds)

    print()
    print("  ПАРАМЕТРЫ АУДИО:")
    print(f"    аудиоканалов     : {info['audio_channels']} "
          f"({'стерео' if info['audio_channels'] == 2 else 'моно'})")
    print(f"    частота модели   : {info['model_samplerate']} Гц")
    if seconds is not None:
        print(f"    сегмент модели   : {seconds:.2f} сек = {model_samples} семплов "
              f"(при {config.sample_rate} Гц)")
    print(f"    наша частота     : {config.sample_rate} Гц "
          f"({'совпадает' if info['model_samplerate'] == config.sample_rate else 'НЕ СОВПАДАЕТ!'})")
    print(f"    наш сегмент     : {config.segment_length} семплов "
          f"({config.segment_seconds:.2f} сек)")

    if model_samples is not None and model_samples != config.segment_length:
        print()
        print("  [!] Наш сегмент не совпадает с тем, на котором модель обучалась")
        print(f"      ({config.segment_length} против {model_samples} семплов).")
        print("      Для дообучения это не поломка, но точность будет выше,")
        print(f"      если в config.py поставить segment_length = {model_samples}.")

    return info


def print_architecture(model, max_layers: int = 60) -> None:
    """
    Печатает список слоёв модели (архитектуру) с количеством параметров
    у каждого. Так видно, из чего модель собрана.

    max_layers ограничивает длину вывода, чтобы консоль не утонула
    в тысяче строк (внутри модели сотни слоёв).
    """
    inner, _ = get_inner_model(model)
    print()
    print("  АРХИТЕКТУРА (верхний уровень):")

    children = list(inner.named_children())
    for name, module in children:
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name:<24} {type(module).__name__:<22} "
              f"{params / 1_000_000:>8.2f} млн параметров")

    # Блок кодировщика и декодировщика - главная часть архитектуры.
    for part_name in ("encoder", "decoder"):
        part = getattr(inner, part_name, None)
        if part is None:
            continue
        print()
        print(f"  {part_name.upper()} ({type(part).__name__}), слои:")
        printed = 0
        for name, module in part.named_children():
            if printed >= max_layers:
                print(f"    ... и ещё слои (всего {len(list(part.children()))})")
                break
            params = sum(p.numel() for p in module.parameters())
            shape = ""
            # Показываем, во что превращается аудио на этом слое, если это
            # возможно узнать без запуска (" downsampling factor").
            factor = getattr(module, "stride", None)
            if factor is not None:
                stride = getattr(factor, "stride", None)
                if stride is not None and isinstance(stride, (tuple, list)) and stride:
                    shape = f"  [шаг {stride[0]}]"
            print(f"    {name:<24} {type(module).__name__:<22} "
                  f"{params / 1_000_000:>7.2f} млн{shape}")
            printed += 1

    # Общее число слоёв - полезно для оценки «тяжести» модели.
    total_modules = sum(1 for _ in inner.modules())
    print()
    print(f"  Всего модулей в сети: {total_modules}")


# ---------------------------------------------------------------------------
# ШАГ 4-6. ПРОГОН СИГНАЛА
# ---------------------------------------------------------------------------

def make_test_signal(torch, length: int, channels: int = 2):
    """
    Создаёт случайный сигнал формы (1, channels, length) - имитацию стерео-звука.

    Это НЕ настоящая музыка, а шум. Но для проверки «живости» модели шума
    достаточно: нам важно, что модель принимает вход нужного размера
    и возвращает 4 дорожки, а не то, как она их разделяет.
    """
    signal = torch.randn(1, channels, length)
    # Модель нормализует вход сама, но вернём ей аккуратный диапазон.
    signal = signal * 0.1
    return signal


def run_forward(torch, model, signal, device: str):
    """
    Прогоняет сигнал через модель и возвращает (выход, секунды).

    ВАЖНЫЙ МОМЕНТ (проверено на demucs 4.1.0):
        Вызывать model(signal) НЕЛЬЗЯ. get_model() возвращает BagOfModels -
        это "мешок" с моделями, и его forward() специально запрещён:
            NotImplementedError: Call `apply_model` on this.
        Правильный вызов - через demucs.apply.apply_model():
            из (batch, каналы, время) -> (batch, источники, каналы, время)
        Раньше (demucs 4.0.1) можно было вызвать model(signal) напрямую,
        поэтому во многих примерах в интернете так и написано - не ведись.

    Ошибки не глотаем - вызывающий код сам разберётся, что печатать.
    """
    from demucs.apply import apply_model

    model = model.to(device)
    model.eval()
    signal = signal.to(device)

    print(f"  Устройство   : {device}")
    print(f"  Вход         : {tuple(signal.shape)} = {list(signal.shape)}")
    print("  Считаю через apply_model()...")
    print("  (на процессоре это может занять от 10 секунд до пары минут)")

    started = time.time()
    with torch.no_grad():   # не считаем градиенты: они тут не нужны и едят память
        # num_workers=0 - без многопроцессности (на Windows она капризная).
        # shifts=0 - ровно один проход, без случайных сдвигов (быстрее).
        output = apply_model(
            model,
            signal,
            device=device,
            shifts=0,
            split=True,
            overlap=config.overlap,
            progress=False,
            num_workers=0,
        )
    elapsed = time.time() - started

    return output, elapsed


def check_output_shape(torch, output, expected_shape: Tuple[int, ...]) -> bool:
    """
    Сравнивает форму выхода с ожидаемой и подробно объясняет расхождения.

    Возвращает True, если всё совпало.
    """
    actual = tuple(output.shape)
    print()
    print("  ФОРМА ВЫХОДА:")
    print(f"    получено     : {list(actual)}")
    print(f"    ожидалось    : {list(expected_shape)}")

    if actual == tuple(expected_shape):
        print("    [OK] Форма полностью совпала с ожидаемой.")
        return True

    print("    [!] Форма НЕ совпала. Разбираем по шагам:")
    if len(actual) != len(expected_shape):
        print(f"      - размерностей: получили {len(actual)}, ждали {len(expected_shape)}")
        print("        (модель вернула не тензор, а что-то другое)")
        return False

    names = ("batch", "источники", "каналы", "время")
    ok = True
    for index, (got, want) in enumerate(zip(actual, expected_shape)):
        axis = names[index] if index < len(names) else f"ось {index}"
        if got == want:
            print(f"      - {axis:<10} {got:>8} = нужно {want:>8}   OK")
        else:
            ok = False
            print(f"      - {axis:<10} {got:>8} != нужно {want:>8}   РАСХОЖДЕНИЕ")
    return ok


# ---------------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ---------------------------------------------------------------------------

def run_check(
    model_name: Optional[str] = None,
    length: Optional[int] = None,
    device: Optional[str] = None,
    skip_forward: bool = False,
) -> int:
    """
    Выполняет всю проверку. Возвращает код возврата (см. шапку файла).
    """
    model_name = model_name or config.demucs_model
    length = length or config.segment_length
    device = device or resolve_device(config.device)

    print(line("="))
    print(" demucs_custom / ПРОВЕРКА DEMUCS")
    print(line("="))
    print(f"  Модель   : {model_name}")
    print(f"  Длина    : {length} семплов "
          f"({length / config.sample_rate:.2f} сек при {config.sample_rate} Гц)")
    print(f"  Проект   : {config.project_root}")

    # --- ШАГ 1: библиотеки ------------------------------------------------
    section("ШАГ 1. Проверяю библиотеки")

    torch = check_torch()
    if torch is None:
        return EXIT_NO_DEPS

    print(f"  [OK] torch       {torch.__version__}")

    pretrained = check_demucs()
    if pretrained is None:
        return EXIT_NO_DEPS

    version = getattr(pretrained, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version
            version = pkg_version("demucs")
        except Exception:
            version = "неизвестно"
    print(f"  [OK] demucs      {version}")

    # --- ШАГ 2: загрузка модели -------------------------------------------
    section("ШАГ 2. Загружаю модель")

    model, error = load_model(model_name)
    if model is None:
        print()
        print("[!] НЕ ПОЛУЧИЛОСЬ. Проверка провалена на шаге загрузки модели.")
        print(f"    Причина: {error}")
        return EXIT_NO_MODEL

    print("  [OK] Модель в памяти.")

    # --- ШАГ 3-4: информация и архитектура -------------------------------
    section("ШАГ 3. Информация о модели")
    info = print_model_info(model)

    section("ШАГ 4. Архитектура")
    print_architecture(model)

    if skip_forward:
        print()
        print("Прогон сигнала пропущен (--skip-forward).")
        return EXIT_OK

    # --- ШАГ 5: сигнал и прогон -------------------------------------------
    section("ШАГ 5. Прогоняю случайный сигнал через модель")

    channels = info["audio_channels"] or config.num_sources
    # Здесь channels = 2 (аудиоканалы), а не число дорожек.
    signal = make_test_signal(torch, length, channels=channels)

    try:
        output, elapsed = run_forward(torch, model, signal, device)
    except Exception as exc:
        print()
        print(f"[!] Модель упала на прямом прогоне: {type(exc).__name__}: {exc}")
        print()
        print("Подробности ошибки:")
        traceback.print_exc()
        print()
        if isinstance(exc, (RuntimeError, MemoryError)) and "memory" in str(exc).lower():
            print("Похоже на нехватку памяти. Попробуй короче сигнал:")
            print(f"    python {Path(__file__).name} --quick")
        return EXIT_FORWARD_FAILED

    # --- ШАГ 6: форма выхода ----------------------------------------------
    section("ШАГ 6. Проверяю форму выхода")

    num_sources = len(info["sources"]) if info["sources"] else config.num_sources
    expected_shape = (1, num_sources, channels, length)
    shape_ok = check_output_shape(torch, output, expected_shape)

    print()
    print("  СКОРОСТЬ И ПАМЯТЬ:")
    print(f"    время прогона  : {elapsed:.1f} сек "
          f"для {length / config.sample_rate:.2f} сек аудио")
    if elapsed > 0:
        realtime = (length / config.sample_rate) / elapsed
        print(f"    быстрее реального времени в {realtime:.2f} раз")
    print(f"    выход, МБ      : {output.numel() * 4 / 1024 / 1024:.1f}")

    if not shape_ok:
        print()
        print("[!] НЕ ПОЛУЧИЛОСЬ: форма выхода не та, что ожидалась.")
        return EXIT_BAD_SHAPE

    # --- ИТОГ --------------------------------------------------------------
    print()
    print(line("="))
    print(" ИТОГ ПРОВЕРКИ")
    print(line("="))
    print(f"  Модель   : {model_name} ({info['inner_class']})")
    print(f"  Параметры: {info['total_params'] / 1_000_000:.1f} млн")
    print(f"  Дорожки  : {len(info['sources'])} -> {', '.join(info['sources'])}")
    print(f"  Выход    : {list(output.shape)}")
    print()
    print(" Demucs работает корректно. Готов к дообучению.")
    print(line("="))
    return EXIT_OK


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: проверка, что Demucs установлен и работает",
    )
    parser.add_argument("--model", default=None,
                        help="какую модель проверять (по умолчанию из config.py)")
    parser.add_argument("--length", type=int, default=None,
                        help="длина тестового сигнала в семплах "
                             "(по умолчанию config.segment_length = 480000)")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="где считать (по умолчанию из config.py)")
    parser.add_argument("--quick", action="store_true",
                        help="короткий сигнал (1<<18 = 262144 семпла) вместо 480000")
    parser.add_argument("--skip-forward", action="store_true",
                        help="только показать модель, не прогонять сигнал")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа."""
    args = parse_args(argv)

    length = args.length
    if length is None and args.quick:
        length = 1 << 18     # 262144 семпла = 5.94 сек

    try:
        return run_check(
            model_name=args.model,
            length=length,
            device=args.device,
            skip_forward=args.skip_forward,
        )
    except KeyboardInterrupt:
        print()
        print("Прервано пользователем (Ctrl+C).")
        return 130


if __name__ == "__main__":
    # Обязательно на Windows при multiprocessing, и просто аккуратно.
    sys.exit(main())
