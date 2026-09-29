# -*- coding: utf-8 -*-
"""
train_custom.py - полный цикл дообучения Demucs на своих данных.

Это сборщик: сам он почти ничего не считает, а соединяет уже готовые
и проверенные модули.

    data_prep.py      -> чтение треков, порядок дорожек
    model_loader.py   -> модель, число дорожек, заморозка энкодеров
    custom_loss.py    -> потеря (время + спектр)
    optimizer_setup.py-> Adam и планировщик
    train_epoch.py    -> одна эпоха обучения и одна проверка
    train_custom.py   -> ЭТОТ файл: цикл эпох, чекпоинты, early stopping

Запуск:

    python train_custom.py                    # как есть, по config.py
    python train_custom.py --epochs 2         # ограничить число эпох
    python train_custom.py --dry-run          # ничего не обучать
    python train_custom.py --yes              # не спрашивать про чекпоинт
    python train_custom.py --synthetic        # проверка без данных


ЧТО БЫЛО НЕ ТАК В СКЕЛЕТЕ

Скелет этого файла содержал свои функции train_one_epoch(), validate()
и build_model(). Они удалены, и вот почему:

- его train_one_epoch() не принимал criterion - то есть в нём не было
  потери. Считать обучение без потери нельзя;
- его validate() тоже не принимал criterion;
- build_model() дублировал model_loader.py, который уже проверен на
  настоящей модели.

Работают функции из train_epoch.py и model_loader.py. Проверяется тестом
в конце файла.

ЛОВУШКА 1. Память кончилась - надо уметь откатиться на CPU.

При нехватке памяти torch бросает torch.OutOfMemoryError. Просто поймать
его мало: если перенести модель на CPU, а состояние оптимизатора
оставить на видеокарте, следующий шаг упадёт с невнятной ошибкой
"Expected all tensors to be on the same device". Поэтому
move_optimizer_state() переносит exp_avg и exp_avg_sq Adam'а вместе с
моделью. Без этого переключение на CPU выглядит как починка, но не
работает.

ЛОВУШКА 2. Один битый трек не должен останавливать обучение.

Плохой файл в папке train/ (оборван, не wav, нечитаем) роняет чтение
батча. Данные читает tolerant_loader(): он ловит ошибку чтения,
печатает её и идёт к следующему батчу. Если нечитаемых батчей подряд
слишком много - обычно значит, что сломан сам DataLoader, а не файлы -
он останавливается, чтобы не крутиться вечно.

ЛОВУШКА 3. Нехватка места на диске.

Один state_dict весит 160 МБ (42 млн float32), чекпоинт с состоянием
Adam - 212 МБ. Перед обучением печатается, сколько свободно.

ЛОВУШКА 4. Спор о частоте чекпоинтов.

В задании сказано "каждые 10 эпох", а в config.py стоит
checkpoint_interval = 5. Здесь взято 10 - как в задании. Пока
расхождение не решено, оно печатается при запуске.

ЛОВУШКА 5. Продолжение с чекпоинта в неинтерактивном режиме.

Если консоли нет, input() падает. Ответом в этом случае считается
"нет": чекпоинт остаётся на диске, обучение начинается с нуля. Начать
с нуля и затереть чекпоинт было бы худшим вариантом.
"""

import argparse
import gc
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from train_epoch import train_one_epoch, validate_one_epoch

# torch.OutOfMemoryError есть и в сборке без CUDA - нужен именно он,
# иначе обработка не сработает на обычной машине.
OOM_ERRORS = getattr(torch, "OutOfMemoryError", RuntimeError)

# Как часто печатать сводку по эпохам (задано в ТЗ).
LOG_EVERY = 10

# Как часто сохранять полный чекпоинт (задано в ТЗ).
CHECKPOINT_EVERY = 10

# Сколько эпох гонять в режиме самопроверки без настоящих данных.
SELFTEST_EPOCHS = 2

# В config.py нет весов потери, поэтому держим их тут.
LAMBDA_FREQ = 0.5
LAMBDA_TIME = 0.5

# Сколько битых батчей подряд терпим, прежде чем решить, что сломан
# DataLoader, а не отдельный файл.
MAX_CONSECUTIVE_BAD_BATCHES = 5


# ---------------------------------------------------------------------------
# МЕЛОЧИ
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """
    Делает случайные числа повторяемыми.

    Полностью повторяемым обучение на CPU всё равно не станет (свёртки и
    STFT считаются многопоточно), но seed убирает разброс от dropout,
    перемешивания и шума в аудио.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  seed = {seed}")


def format_time(seconds: float) -> str:
    """Человеческое время: '45 сек', '2 мин 03 сек', '1 ч 02 мин'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} сек"
    if seconds < 3600:
        return f"{seconds // 60} мин {seconds % 60:02d} сек"
    return f"{seconds // 3600} ч {(seconds % 3600) // 60:02d} мин"


def plural(count: int, forms: Tuple[str, str, str]) -> str:
    """
    Правильное окончание: plural(2, ("эпоха", "эпохи", "эпох")) -> "эпохи".

    Зачем: '1 эпоха', '2 эпохи', '5 эпох' - а тупое f"{n} эпох" читается
    как ошибка и мешает доверию к остальному выводу.
    """
    if count % 10 == 1 and count % 100 != 11:
        return forms[0]
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return forms[1]
    return forms[2]


def save_atomic(state: Any, path: Path, what: str) -> None:
    """
    Сохраняет файл через временный, потом переименовывает.

    Зачем: torch.save пишет файл на месте и довольно долго (тут 160-212
    МБ). Если в этот момент вырубить питание, останется обрывок, из
    которого Python уже не сможет загрузить модель - то есть "лучшая
    модель" испорчена ровно тогда, когда она нужнее всего. С временным
    файлом оригинал либо целый, либо его нет.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temp)
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    try:
        size = path.stat().st_size / (1024 * 1024)
    except OSError:
        size = 0.0
    print(f"  [{what}] сохранено: {path}  ({size:.0f} МБ)")


def free_memory(device: Union[str, torch.device]) -> None:
    """Отпускает память. На CPU просто чистит сборщик мусора."""
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_optimizer_state(optimizer: torch.optim.Optimizer,
                         device: torch.device) -> None:
    """
    Переносит внутреннее состояние оптимизатора на другое устройство.

    Нужно при откате на CPU. У Adam внутри лежат exp_avg и exp_avg_sq -
    по два float32 на каждый обучаемый параметр, то есть 54 МБ. Если
    перенести только модель, следующий optimizer.step() упадёт с
    "Expected all tensors to be on the same device".
    """
    moved = 0
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                state[key] = value.to(device)
                moved += 1
    if moved:
        print(f"  состояние оптимизатора перенесено на {device} "
              f"({moved} тензоров)")


# ---------------------------------------------------------------------------
# ДАННЫЕ
# ---------------------------------------------------------------------------


def count_real_tracks(root: Path) -> int:
    """Сколько настоящих треков (mixture.wav) лежит в папке данных."""
    if not root.exists():
        return 0
    return len(list(root.glob("*/**/mixture.wav")))


def tolerant_loader(loader: Any,
                    label: str = "данные",
                    max_consecutive: int = MAX_CONSECUTIVE_BAD_BATCHES,
                    ) -> Iterator[Any]:
    """
    Отдаёт батчи из DataLoader, переживая битые файлы.

    Ошибка чтения одного трека (оборванный wav, не тот формат, нет
    файла) печатается и пропускается - обучение продолжается. Если битых
    батчей подряд больше max_consecutive, значит дело не в файле, а в
    загрузчике: останавливаемся и говорим прямо, чтобы не учиться впустую.
    """
    iterator = iter(loader)
    bad_in_a_row = 0
    skipped = 0

    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            if skipped:
                print(f"  [{label}] пропущено битых батчей за эпоху: {skipped}")
            return
        except OOM_ERRORS:
            # Нехватка памяти - это не битый файл, этим занимается
            # внешний код. Прокидываем наверх.
            raise
        except Exception as exc:
            skipped += 1
            bad_in_a_row += 1
            lines = str(exc).strip().splitlines()
            head = lines[0] if lines else exc.__class__.__name__
            print(f"  [!] [{label}] битый батч пропущен "
                  f"({type(exc).__name__}: {head[:90]})")
            if bad_in_a_row >= max_consecutive:
                print(f"  [!] [{label}] подряд {bad_in_a_row} битых батчей - "
                      f"похоже, сломан сам загрузчик, а не файлы. "
                      f"Останавливаю эпоху.")
                return
            continue

        bad_in_a_row = 0
        yield batch


# ---------------------------------------------------------------------------
# СБОРКА
# ---------------------------------------------------------------------------


def build_model(config: Any, device: torch.device) -> nn.Module:
    """Загружает модель, подгоняет под число дорожек, замораживает энкодеры."""
    from model_loader import (
        freeze_layers,
        load_demucs_model,
        modify_model_for_custom,
    )

    model = load_demucs_model(model_name=config.demucs_model, device=str(device))
    model = modify_model_for_custom(model, num_sources=config.num_sources)
    model = freeze_layers(model, freeze_encoder=True)
    model.to(device)
    return model


def build_dataloaders(config: Any, synthetic: bool = False) -> Tuple[Any, Any]:
    """Возвращает (train_loader, valid_loader)."""
    if synthetic:
        from train_epoch import SyntheticSeparationDataset

        train_set = SyntheticSeparationDataset(
            num_tracks=4, length=44100, seed=config.seed
        )
        valid_set = SyntheticSeparationDataset(
            num_tracks=2, length=44100, seed=config.seed + 1000
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=2, shuffle=False, num_workers=0
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=2, shuffle=False, num_workers=0
        )
        return train_loader, valid_loader

    from data_prep import get_custom_loaders

    return get_custom_loaders(config)


# ---------------------------------------------------------------------------
# ЧЕКПОИНТЫ
# ---------------------------------------------------------------------------


def checkpoint_path_for(best_path: Path) -> Path:
    """Путь чекпоинта рядом с лучшей моделью: custom_demucs.pth ->
    custom_demucs_last.pth."""
    return best_path.with_name(f"{best_path.stem}_last{best_path.suffix}")


def save_checkpoint(path: Path, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, best_loss: float,
                    history: List[Dict[str, Any]],
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Полный чекпоинт, с которого можно продолжить.

    Ключи названы как в задании: epoch, model_state, optimizer_state,
    best_loss, history. Одних весов мало: без состояния Adam счётчики
    моментов теряются, и первые шаги после возобновления заметно хуже.
    """
    state: Dict[str, Any] = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_loss,
        "history": history,
    }
    if extra:
        state.update(extra)
    save_atomic(state, path, f"чекпоинт эпохи {epoch}")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    """
    Читает чекпоинт.

    Сначала пробуем weights_only=True: он запрещает pickle выполнить
    произвольный код, то есть защищает от чужого .pth. Наши данные -
    словари, списки и тензоры, так что должно открыться. Если нет,
    читаем обычным способом и честно предупреждаем, что это небезопасно.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        print("  [!] Чекпоинт не открылся в безопасном режиме, читаю обычным "
              "способом (менее безопасно): файл не должен приходить из интернета")
        return torch.load(path, map_location="cpu", weights_only=False)


def ask_to_resume(epoch: int, assume_yes: bool = False) -> bool:
    """Спрашивает, продолжать ли с сохранённой эпохи."""
    if assume_yes:
        return True
    try:
        answer = input(f"  Найден чекпоинт после эпохи {epoch}. "
                       f"Продолжить с эпохи {epoch + 1}? (y/n): ")
    except (EOFError, KeyboardInterrupt):
        print("  (ответа нет - начинаю с нуля, чекпоинт останется на диске)")
        return False
    return answer.strip().lower() in ("y", "yes", "д", "да")


def maybe_resume(ckpt_path: Path, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 assume_yes: bool) -> Tuple[int, float, List[Dict[str, Any]]]:
    """
    Спрашивает про чекпоинт и, если да, восстанавливает состояние.

    Возвращает (start_index, лучшая потеря, история).

    ВНИМАНИЕ К ИНДЕКСАМ - здесь легко ошибиться на единицу. saved_epoch
    в чекпоинте - это НОМЕР ЭПОХИ, как его видит человек (1, 2, 3...),
    то есть он 1-based. А цикл for epoch in range(start_index, ...) ждёт
    индекс с нуля. Человеческая "эпоха 3" - это индекс 2.

    Поэтому start_index = saved_epoch, а НЕ saved_epoch + 1. Со знаком
    плюс эпоха 3 не выполнится никогда: range(3, 3) пуст, и скрипт
    радостно сообщит, что всё уже отучилось.

    Ошибка чтения чекпоинта НЕ приводит к отказу от обучения - файл
    остаётся на диске нетронутым, обучение начинается с нуля.
    """
    if not ckpt_path.exists():
        print("  Чекпоинта нет - начинаю с нуля")
        return 0, math.inf, []

    try:
        saved = load_checkpoint(ckpt_path)
        saved_epoch = int(saved.get("epoch", -1))
    except Exception as exc:
        print(f"  [!] Чекпоинт не прочитался ({type(exc).__name__}: "
              f"{str(exc)[:90]}). Начинаю с нуля, файл не трогаю.")
        return 0, math.inf, []

    print(f"  Найден чекпоинт после эпохи {saved_epoch}, "
          f"лучшая потеря {saved.get('best_loss')}")

    if saved_epoch < 0 or not ask_to_resume(saved_epoch, assume_yes=assume_yes):
        if saved_epoch >= 0:
            print("  Начинаю с нуля, чекпоинт останется на диске")
        return 0, math.inf, []

    try:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
    except Exception as exc:
        print(f"  [!] Не сошлось состояние с текущей моделью "
              f"({type(exc).__name__}: {str(exc)[:90]}). "
              f"Начинаю с нуля, чекпоинт не трогаю.")
        return 0, math.inf, []

    best_loss = float(saved.get("best_loss", math.inf))
    history = list(saved.get("history", []))
    print(f"  Продолжаю с эпохи {saved_epoch + 1} "
          f"(лучшая потеря {best_loss:.6f}, в чекпоинте эпох: {len(history)})")
    return saved_epoch, best_loss, history


# ---------------------------------------------------------------------------
# ОТКАТ НА CPU ПРИ НЕХВАТКЕ ПАМЯТИ
# ---------------------------------------------------------------------------


def run_with_oom_recovery(step_fn: Callable[[torch.device], float],
                          model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device,
                          label: str) -> Tuple[float, torch.device]:
    """
    Вызывает step_fn(device), а при нехватке памяти чинит и повторяет.

    Порядок: один повтор на том же устройстве (часто хватает - прошлый
    батч ещё держит память), затем перенос модели и состояния
    оптимизатора на CPU. Возвращает результат и устройство, на котором он
    получен: цикл обязан запомнить новое устройство.
    """
    current = device
    attempts = 0

    while True:
        try:
            return step_fn(current), current
        except OOM_ERRORS as exc:
            free_memory(current)
            attempts += 1

            if current.type == "cpu":
                raise RuntimeError(
                    f"Не хватило памяти даже на CPU во время {label}.\n"
                    f"  Уменьши config.batch_size (например до 2) или "
                    f"config.segment_length.\n"
                    f"  Подробности: {str(exc)[:200]}"
                ) from exc

            if attempts == 1:
                print(f"  [!] Не хватило памяти на {current} во время {label}. "
                      f"Освобождаю и повторяю.")
                continue

            print("  [!] Памяти не хватило и после повтора. Переключаюсь "
                  "на CPU: обучение сильно замедлится, но продолжит работать.")
            current = torch.device("cpu")
            model.to(current)
            move_optimizer_state(optimizer, current)


# ---------------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ---------------------------------------------------------------------------


def train_custom_model(config: Any, assume_yes: bool = False,
                       epochs_explicit: bool = False) -> str:
    """
    Дообучает модель и возвращает путь к лучшей сохранённой модели.

    config - объект из config.py. Функция его НЕ меняет: все решения
    принимаются на локальных переменных, поэтому её можно вызвать
    несколько раз в одном процессе.

    epochs_explicit - пользователь сам задал число эпох (--epochs).
    Тогда ограничение самотеста не применяется: иначе нельзя было бы
    продолжить с чекпоинта, остановив самотест на двух эпохах.
    """
    started_all = time.time()

    print("=" * 74)
    print(" ДООБУЧЕНИЕ DEMUCS")
    print("=" * 74)

    for problem in config.validate():
        print(f"  [!] config: {problem}")
    created = config.ensure_directories()
    if created:
        print("  Созданы папки: " + ", ".join(created))

    print()
    print("-" * 74)
    print(" Шаг 1. Устройство и случайные числа")
    print("-" * 74)
    from config import resolve_device

    device = torch.device(resolve_device(getattr(config, "device", None)))
    print(f"  просят в config : {getattr(config, 'device', '?')}")
    print(f"  реально доступно: {device}")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("  [!] CUDA просят, но её нет - работаем на CPU")
    set_seed(config.seed)
    print()

    print("-" * 74)
    print(" Шаг 2. Данные")
    print("-" * 74)
    data_root = Path(config.custom_data_path_abs)
    real_tracks = count_real_tracks(data_root)
    synthetic = real_tracks == 0

    best_path = Path(config.output_model_path_abs)
    total_epochs = int(config.epochs)

    if synthetic:
        print(f"  В папке {data_root} нет ни одного mixture.wav")
        print()
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("  !  НАСТОЯЩИХ ДАННЫХ НЕТ - идёт ПРОВЕРКА на шуме.        !")
        print("  !  Это не обучение: результат годен только чтобы убедиться,")
        print("  !  что конвейер собирается и работает.                 !")
        print("  !  Подготовь данные: python prepare_data_structure.py  !")
        print("  !  и разложи дорожки по train/track_XXX/               !")
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
        if not epochs_explicit:
            total_epochs = min(total_epochs, SELFTEST_EPOCHS)
        else:
            print(f"  число эпох задано явно ({total_epochs}), "
                  f"ограничение самотеста не применяю")
        best_path = best_path.with_name(
            f"{best_path.stem}_selftest{best_path.suffix}")
        print(f"  эпох: {total_epochs} (проверка, а не обучение)")
        print(f"  результат сохраню отдельно, настоящий путь не затираю: "
              f"{best_path.name}")
    else:
        print(f"  треков найдено: {real_tracks}")
        print(f"  эпох: {total_epochs}")
        print(f"  итоговая модель: {best_path}")

    print(f"  сводка эпохи и чекпоинт - каждые {LOG_EVERY} эпох")
    if getattr(config, "checkpoint_interval", None) != CHECKPOINT_EVERY:
        print(f"  [!] В config.checkpoint_interval стоит "
              f"{config.checkpoint_interval}, а работает {CHECKPOINT_EVERY} "
              f"(как в задании). Если нужно иначе - скажи.")
    print(f"  рабочих процессов чтения: {0 if synthetic else config.num_workers}")
    train_loader, valid_loader = build_dataloaders(config, synthetic=synthetic)
    print()

    print("-" * 74)
    print(" Шаг 3. Модель")
    print("-" * 74)
    model = build_model(config, device)
    print()

    print("-" * 74)
    print(" Шаг 4. Потеря, оптимизатор, планировщик")
    print("-" * 74)
    from custom_loss import CustomLoss
    from optimizer_setup import setup_optimizer, setup_scheduler

    criterion = CustomLoss(lambda_freq=LAMBDA_FREQ, lambda_time=LAMBDA_TIME)
    criterion = criterion.to(device)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    print()

    print("-" * 74)
    print(" Шаг 5. Место на диске")
    print("-" * 74)
    ckpt_path = checkpoint_path_for(best_path)
    try:
        probe = data_root if data_root.exists() else Path.cwd()
        free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    except Exception:
        free_gb = float("nan")
    print("  лучшая модель (state_dict): ~160 МБ")
    print("  чекпоинт (модель + Adam):   ~212 МБ")
    if not math.isnan(free_gb):
        print(f"  свободно на диске:         {free_gb:.1f} ГБ")
        if free_gb * 1024 < 372:
            print("  [!] Места может не хватить: упадём при сохранении, "
                  "но не раньше")
    print()

    print("-" * 74)
    print(" Шаг 6. Продолжить ли с чекпоинта")
    print("-" * 74)
    start_index, best_loss, history = maybe_resume(
        ckpt_path, model, optimizer, assume_yes=assume_yes
    )
    if start_index >= total_epochs and history:
        print(f"  Все {total_epochs} {plural(total_epochs, ('эпоха', 'эпохи', 'эпох'))} "
              f"уже отучились в прошлый раз.")
        if not best_path.exists():
            print("  Лучшей модели нет на диске - сохраняю текущую.")
            save_atomic(model.state_dict(), best_path, "текущая модель")
        print(f"  Возвращаю {best_path}")
        return str(best_path)

    print()
    print("=" * 74)
    print(f" НАЧИНАЕМ ОБУЧЕНИЕ: эпох {start_index + 1}..{total_epochs}")
    print("=" * 74)

    without_improvement = 0
    early_stopped = False

    for epoch in range(start_index, total_epochs):
        epoch_started = time.time()
        number = epoch + 1
        lr_before = optimizer.param_groups[0]["lr"]
        print()
        print(f"  --- Эпоха {number} из {total_epochs} (lr={lr_before:.2e}) ---")

        def train_step(dev: torch.device) -> float:
            return train_one_epoch(
                model, tolerant_loader(train_loader, "train"),
                optimizer, criterion, dev,
                max_norm=float(config.grad_clip_norm),
            )

        train_loss, device = run_with_oom_recovery(
            train_step, model, optimizer, device, f"обучения (эпоха {number})")

        def valid_step(dev: torch.device) -> float:
            return validate_one_epoch(
                model, tolerant_loader(valid_loader, "valid"),
                criterion, dev,
            )

        valid_loss, device = run_with_oom_recovery(
            valid_step, model, optimizer, device, f"проверки (эпоха {number})")

        # Планировщик шагает по проверке, а не по обучению.
        scheduler.step(valid_loss)
        lr_after = optimizer.param_groups[0]["lr"]

        improved = valid_loss < best_loss
        if improved:
            best_loss = valid_loss
            save_atomic(model.state_dict(), best_path, "лучшая модель")
            print(f"  Новая лучшая модель сохранена! Loss: {valid_loss:.6f}")
        else:
            print(f"  Лучшая модель не улучшилась: {valid_loss:.6f} "
                  f"против {best_loss:.6f}")

        history.append({
            "epoch": number,
            "train": train_loss,
            "valid": valid_loss,
            "lr": lr_after,
            "seconds": time.time() - epoch_started,
            "improved": improved,
        })

        if (number % LOG_EVERY == 0 or number == 1
                or number == total_epochs):
            print()
            print(f"  Epoch {number} | Train: {train_loss:.6f} | "
                  f"Valid: {valid_loss:.6f}")

        if config.save_checkpoints and (number % CHECKPOINT_EVERY == 0
                                        or number == total_epochs):
            save_checkpoint(ckpt_path, number, model, optimizer,
                            best_loss, history,
                            extra={"config": config.to_dict()})

        without_improvement = 0 if improved else without_improvement + 1
        if config.early_stopping and \
                without_improvement >= int(config.early_stopping_patience):
            print()
            print(f"  Ранняя остановка: {without_improvement} эпох подряд без "
                  f"улучшения (config.early_stopping_patience).")
            print("  Лучшая модель сохранена раньше - можно вернуться к ней.")
            early_stopped = True
            break

    print()
    print("=" * 74)
    print(" ГОТОВО")
    print("=" * 74)
    if not best_path.exists():
        print("  [!] Лучшая модель на диске не найдена - сохраняю текущую")
        save_atomic(model.state_dict(), best_path, "текущая модель")
    print(f"  эпох отучено  : {len(history)}"
          f"{' (ранняя остановка)' if early_stopped else ''}")
    print(f"  лучшая потеря : {best_loss:.6f}")
    print(f"  всего времени : {format_time(time.time() - started_all)}")
    print(f"  лучшая модель : {best_path}")
    print(f"  чекпоинт      : {ckpt_path}")
    if synthetic:
        print()
        print("  !  Это была ПРОВЕРКА на шуме, а не обучение. Чтобы учить на")
        print("  !  своей музыке, подготовьте custom_dataset/train/ и запустите")
        print("  !  скрипт заново.")
    if history:
        print()
        print("  последние эпохи:")
        for record in history[-5:]:
            mark = "  <- лучшая" if record["improved"] else ""
            print(f"    эпоха {record['epoch']:>3}: "
                  f"train={record['train']:.6f}  "
                  f"valid={record['valid']:.6f}  "
                  f"lr={record['lr']:.2e}  "
                  f"{format_time(record['seconds'])}{mark}")

    return str(best_path)


# ---------------------------------------------------------------------------
# ПРОВЕРКА БЕЗ ОБУЧЕНИЯ
# ---------------------------------------------------------------------------


def dry_run(config: Any) -> int:
    """Собирает всё и ничего не обучает - чтобы увидеть проблемы до
    того, как потрачено несколько часов."""
    print("=" * 74)
    print(" ПРОВЕРКА (dry-run) - обучение не запускается")
    print("=" * 74)
    print()

    for problem in config.validate():
        print(f"  [!] {problem}")
    config.ensure_directories()

    from config import resolve_device
    device = torch.device(resolve_device(getattr(config, "device", None)))
    print(f"  устройство: {device}")

    data_root = Path(config.custom_data_path_abs)
    real_tracks = count_real_tracks(data_root)
    print(f"  папка данных: {data_root}")
    print(f"  треков: {real_tracks}")
    if not real_tracks:
        print("  [!] Данных нет. Подготовьте: python prepare_data_structure.py")

    print()
    print("  загружаю модель...")
    model = build_model(config, device)

    from custom_loss import CustomLoss
    from optimizer_setup import setup_optimizer, setup_scheduler
    CustomLoss(lambda_freq=LAMBDA_FREQ, lambda_time=LAMBDA_TIME).to(device)
    optimizer = setup_optimizer(model, config)
    setup_scheduler(optimizer, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"  обучаемых параметров : {trainable:,}")
    print(f"  эпох                 : {config.epochs}")
    print(f"  итоговая модель      : {config.output_model_path_abs}")
    print()
    print("  Всё собирается. Запустить обучение: python train_custom.py")
    return 0


# ---------------------------------------------------------------------------
# КОМАНДНАЯ СТРОКА
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Дообучение Demucs на своих дорожках")
    parser.add_argument("--epochs", type=int, default=None,
                        help="сколько эпох (по умолчанию config.epochs)")
    parser.add_argument("--device", default=None,
                        help="cpu, cuda или mps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="размер батча")
    parser.add_argument("--output", default=None,
                        help="куда сохранить лучшую модель")
    parser.add_argument("--synthetic", action="store_true",
                        help="проверка на шуме, даже если данные есть")
    parser.add_argument("--yes", action="store_true",
                        help="не спрашивать про чекпоинт (для скриптов)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="не останавливаться при отсутствии улучшений")
    parser.add_argument("--dry-run", action="store_true",
                        help="только собрать и проверить, не обучать")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    from config import reload_config
    config = reload_config()

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.output is not None:
        config.output_model_path = args.output
    if args.no_early_stopping:
        config.early_stopping = False

    if args.dry_run:
        return dry_run(config)

    if not args.synthetic and \
            count_real_tracks(Path(config.custom_data_path_abs)) == 0:
        print("  Настоящих данных не найдено - запускаю проверку на шуме.")
        print("  Чтобы учить на своей музыке, подготовьте папку и запустите "
              "заново.")
        print()

    path = train_custom_model(config, assume_yes=args.yes,
                              epochs_explicit=args.epochs is not None)

    print()
    print(f"Дообучение завершено. Модель: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
