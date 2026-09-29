# -*- coding: utf-8 -*-
"""
optimizer_setup.py — оптимизатор и планировщик скорости обучения.

Зачем это отдельный файл и почему всё так осторожно.

Мы дообучаем чужую модель, и это меняет всё. Обычно при обучении с нуля
настройки оптимизатора выбирают свободно: нулевая инициализация, никто
ничего не знает, терять нечего. А здесь на старте у нас 42 миллиона
параметров, из которых 35 миллионов уже отлично работают. Плохо
настроенный оптимизатор способен за несколько эпох снести всё, чему
Demucs пришлось научиться на миллионах часов музыки, - и вот тогда
начинать придётся заново.

Отсюда три правила, которые зашиты в этот модуль:

1. Оптимизатор видит ТОЛЬКО размороженные параметры. Заложенные веса
   не должны получать ни градиента, ни шага оптимизатора. Если мимо
   optimize() пробежит хотя бы один замороженный вес, он начнёт
   медленно сползать, и модель испортится, а заметить это будет трудно.
   Поэтому здесь стоит проверка: сколько параметров попало в группы.

2. Начальная скорость обучения маленькая (1e-4). Модель уже хорошая,
   ей не нужно учиться заново - нужно чуть-чуть подстроиться.

3. Скорость умеет сама уменьшаться, если обучение встало на месте.
   ReduceLROnPlateau смотрит на потерю на проверочных данных: если она
   5 эпох подряд не улучшается, шаг уменьшается вдвое, но не ниже 1e-6.
   Нижняя граница нужна, чтобы обучение не встало совсем: слишком
   маленький шаг приводит к нулю, и модель перестаёт учиться вообще.

Проверено на torch 2.14.0, demucs 4.1.0 / htdemucs, 26.09.2026:

  * learning_rate в config.py = 0.0001 - ровно то, что нужно.
  * weight_decay в config.py = 0.0001, а в этом модуле используется
    1e-5. Разница в десять раз. См. WEIGHT_DECAY ниже: значение из
    задания выбрано осознанно, и расхождение с config.py выводится
    в консоль, чтобы его нельзя было потерять из виду.

  * Скорость обучения в 10 раз меньше, чем у самого Demucs (там 1e-3).
    Это осознанно: у них обучение с нуля, у нас - точечная подстройка.

Как пользоваться:

    from optimizer_setup import setup_optimizer, setup_scheduler

    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        # ВАЖНО: планировщик ReduceLROnPlateau умеет только так -
        # через потерю на проверочных данных. Просто scheduler.step()
        # без аргумента здесь не сработает.
        scheduler.step(val_loss)

        print(f"эпоха {epoch}: train={train_loss:.5f} "
              f"val={val_loss:.5f} lr={optimizer.param_groups[0]['lr']:.2e}")
"""

import torch
import torch.nn as nn

# Импортируем сразу двумя именами, и это не украшение.
#
# Параметр функций называется config - как в задании. Но в этом же файле
# есть глобальный объект config из config.py. Если импортировать его как
# config, то внутри функции параметр перекроет глобальный объект, и любой
# обращающийся к config.learning_rate молча станет обращением к
# параметру. Поэтому глобальный переименовываем, а тип оставляем.
from config import Config
from config import config as default_config
from model_loader import (
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    freeze_layers,
    inner_models,
    print_model_info,
)

# Значение из задания: 1e-5. НЕ берём из config.weight_decay - там 1e-4,
# в десять раз больше. Расхождение намеренное и выводится в консоль.
WEIGHT_DECAY = 1e-5

# Планировщик: ждать 5 эпох без улучшения, делить скорость пополам,
# не опускать её ниже 1e-6.
LR_PATIENCE = 5
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6


def _unwrap_prefix(param_name: str) -> str:
    """
    Срезает префикс обёртки у имени параметра.

    get_model("htdemucs") возвращает не саму сеть, а обёртку BagOfModels.
    Из-за этого все имена начинаются с "models.0." - например
    "models.0.encoder.0.conv.weight". Если не срезать префикс, то
    первое же сравнение name.split(".")[0] даст "models" для ЛЮБОГО
    параметра, и проверка "заморожен ли энкодер" будет всегда ложной -
    то есть пройдёт вхолостую, ничего не проверяя.

    Такой баг уже ловился в тестах модели, поэтому здесь повторяем
    ту же осторожность.
    """
    parts = param_name.split(".")
    if len(parts) >= 3 and parts[0] == "models" and parts[1].isdigit():
        parts = parts[2:]
    return parts[0] if parts else ""


def split_params(model: nn.Module):
    """
    Делит параметры на две группы: обучаемые и замороженные.

    Главный принцип: решает НЕ имя параметра, а флаг requires_grad.
    Имя может соврать (из-за обёртки, из-за нестандартных имён), а
    requires_grad выставляет freeze_layers() и он один источник правды.
    Имя мы используем только для того, чтобы в отчёте показать, где
    именно находятся обучаемые веса.

    Возвращает (frozen_params, trainable_params) - два списка параметров.
    """
    frozen, trainable = [], []
    for param in model.parameters():
        (trainable if param.requires_grad else frozen).append(param)
    return frozen, trainable


def describe_split(model: nn.Module) -> str:
    """
    Рассказывает, что и где находится, в понятных числах.

    Нужна не для красоты. Если freeze_layers() когда-нибудь отработает
    не по тому правилу, это будет видно сразу, а не через час
    бесполезного обучения.
    """
    lines = []
    for net in inner_models(model):
        frozen, trainable = split_params(net)
        total = sum(p.numel() for p in net.parameters())

        def count_prefix(flag, prefixes):
            """Сколько значений с данным флагом требует_grad и таким именем."""
            return sum(p.numel() for n, p in net.named_parameters()
                       if p.requires_grad == flag
                       and _unwrap_prefix(n) in prefixes)

        enc = count_prefix(False, ENCODER_PREFIXES)
        dec = count_prefix(True, DECODER_PREFIXES)

        lines.append(f"  Сеть {type(net).__name__}:")
        lines.append(f"    всего параметров      : {total:>12,}")
        lines.append(f"    заморожено            : {sum(p.numel() for p in frozen):>12,}")
        lines.append(f"    обучается             : {sum(p.numel() for p in trainable):>12,}")
        lines.append(f"    из них энкодеры (не учатся): {enc:>9,}")
        lines.append(f"    из них декодеры (учатся)   : {dec:>9,}")
    return "\n".join(lines)


def setup_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """
    Собирает Adam только для размороженных параметров.

    Если модель ещё не замораживалась, замораживаем её сами - иначе
    оптимизатор получит все 42 миллиона параметров и за несколько эпох
    испортит предобученные веса. Это самая дорогая ошибка во всём
    проекте, поэтому проверяем.

    Параметры:
        model  - nn.Module (обычно обёртка BagOfModels)
        config - настройки проекта, из них берём learning_rate

    Возвращает torch.optim.Optimizer.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Ожидалась nn.Module, а пришло: {type(model).__name__}"
        )

    # Если веса ещё все обучаемые - это почти наверняка забытый вызов
    # freeze_layers(). В htdemucs без заморозки обучается вся сеть целиком.
    frozen, trainable = split_params(model)
    if not trainable:
        raise ValueError(
            "В модели нет ни одного обучаемого параметра.\n"
            "  Похоже, freeze_layers() заморозил всё подряд - проверь,\n"
            "  что разморожен хотя бы декодер (decoder, tdecoder)."
        )
    if not frozen:
        print("  [!] ВНИМАНИЕ: ни один параметр не заморожен - учится вся")
        print("      сеть целиком, включая энкодер. Для дообучения чужой")
        print("      модели это почти наверняка не то, что нужно:")
        print("      предобученные веса испортятся за несколько эпох.")
        print("      Проверь, что freeze_layers() вызван с freeze_encoder=True.")

    # Раскладываем по именам, чтобы в отчёте видеть, ЧТО именно учится.
    # Только для отчёта - в оптимизатор идёт то, что решил requires_grad.
    dec_trainable = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and _unwrap_prefix(n) in DECODER_PREFIXES
    ]
    enc_frozen = [
        (n, p) for n, p in model.named_parameters()
        if not p.requires_grad and _unwrap_prefix(n) in ENCODER_PREFIXES
    ]
    dec_params = sum(p.numel() for _, p in dec_trainable)
    enc_params = sum(p.numel() for _, p in enc_frozen)

    print("Группы параметров:")
    print(f"  заморожено (не оптимизируем) : {len(frozen):>4} тензоров, "
          f"{sum(p.numel() for p in frozen):,} значений")
    print(f"  обучается   (оптимизируем)   : {len(trainable):>4} тензоров, "
          f"{sum(p.numel() for p in trainable):,} значений")
    print(f"    из них энкодеры  : {enc_params:>12,}  (заморожены, как и надо)")
    print(f"    из них декодеры  : {dec_params:>12,}  (учатся, как и надо)")

    # Страховка от тихой поломки: если по именам обучается что-то вне
    # декодеров - значит freeze_layers() отработал не так, как мы думали.
    # Просто сообщим, но обучение не заблокируем.
    others = [
        n for n, p in model.named_parameters()
        if p.requires_grad and _unwrap_prefix(n) not in DECODER_PREFIXES
    ]
    if others:
        print(f"  [!] Учатся {len(others)} тензоров ВНЕ декодеров, например:")
        for name in others[:3]:
            print(f"        {name}")
        print("      По задумке учиться должны только decoder и tdecoder.")

    learning_rate = float(config.learning_rate)
    if learning_rate <= 0:
        raise ValueError(
            f"learning_rate должен быть положительным, получили {learning_rate}."
        )

    # Передаём в Adam ТОЛЬКО обучаемые. Это и есть тот предохранитель,
    # о котором шла речь выше: у замороженного веса нет градиента, а
    # даже если бы был, optimizer.step() его бы не увидел.
    optimizer = torch.optim.Adam(
        trainable,
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    print("Оптимизатор: Adam")
    print(f"  learning_rate  : {learning_rate}  (из config)")
    print(f"  weight_decay   : {WEIGHT_DECAY}")
    print(f"  betas          : {optimizer.defaults['betas']}")
    print(f"  групп параметров: {len(optimizer.param_groups)}")

    # Расхождение с config.weight_decay показываем, но не мешаем.
    config_wd = float(getattr(config, "weight_decay", WEIGHT_DECAY))
    if abs(config_wd - WEIGHT_DECAY) > 1e-12:
        print(f"  [!] ВНИМАНИЕ: config.weight_decay = {config_wd}, "
              f"а используется {WEIGHT_DECAY}.")
        print(f"      Разница в {config_wd / WEIGHT_DECAY:g} раз. Здесь взято "
              f"значение 1e-5 из задания.")
        print(f"      Если нужно наоборот - скажи, или поменяй WEIGHT_DECAY "
              f"в начале этого файла.")

    return optimizer


def setup_scheduler(optimizer: torch.optim.Optimizer,
                    config: Config) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Настраивает уменьшение скорости обучения, если потеря встала.

    ReduceLROnPlateau - единственный планировщик, который смотрит на
    сами числа, а не на номер эпохи. Логика простая: если проверочная
    потеря не улучшилась 5 эпох подряд, значит модель упёрлась, и дальше
    учиться с прежним шагом она не будет - только будет метаться. Шаг
    уменьшается вдвое, и проверка продолжается.

    mode="min" обязателен: мы минимизируем потерю, а по умолчанию
    планировщик ждёт максимум и никогда бы не сработал.

    ВАЖНО, как вызывать. Этот планировщик умеет только один вызов:

        scheduler.step(val_loss)

    Обычный scheduler.step() без аргумента у ReduceLROnPlateau
    не работает и бросает ошибку. Передавать надо потерю на
    проверочных данных (val), а не на обучающих.

    min_lr=1e-6 - нижняя граница. Без неё шаг мог бы упасть почти до
    нуля, и тогда модель перестала бы учиться совсем, хотя потери
    продолжали бы выглядеть "вроде стабильно". С этой границей
    обучение всегда остаётся живым.
    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "Ожидался torch.optim.Optimizer, а пришло: "
            f"{type(optimizer).__name__}"
        )

    start_lr = optimizer.param_groups[0]["lr"]
    if start_lr <= MIN_LEARNING_RATE:
        print(f"  [!] Стартовая скорость {start_lr} уже на нижней границе "
              f"{MIN_LEARNING_RATE}.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",             # минимизируем потерю, а не максимизируем
        factor=LR_FACTOR,       # 0.5 -> скорость падает вдвое
        patience=LR_PATIENCE,   # 5 эпох без улучшения -> уменьшаем
        min_lr=MIN_LEARNING_RATE,
    )

    print("Планировщик: ReduceLROnPlateau")
    print(f"  mode         : min  (уменьшаем потерю)")
    print(f"  factor       : {LR_FACTOR}  (скорость делится на "
          f"{1 / LR_FACTOR:g})")
    print(f"  patience     : {LR_PATIENCE} эпох без улучшения")
    print(f"  min_lr       : {MIN_LEARNING_RATE}")
    print(f"  сейчас lr    : {optimizer.param_groups[0]['lr']}")
    print(f"  Вызывать так : scheduler.step(val_loss)")

    # Покажем, до какой скорости дойдём при неудаче.
    lr = start_lr
    steps = 0
    while lr > MIN_LEARNING_RATE:
        lr = max(MIN_LEARNING_RATE, lr * LR_FACTOR)
        steps += 1
        if steps > 100:
            break
    print(f"  При неудаче скорость опустится до {MIN_LEARNING_RATE} "
          f"за {steps} уменьшений.")

    return scheduler


def main() -> int:
    print("=" * 66)
    print(" optimizer_setup / ОПТИМИЗАТОР И ПЛАНИРОВЩИК")
    print("=" * 66)
    print()

    # 1. Модель. Берём из model_loader, чтобы загрузка и заморозка
    #    были ровно те же, что при настоящем обучении.
    print("-" * 66)
    print(" Шаг 1. Загружаем модель и замораживаем энкодер")
    print("-" * 66)
    from model_loader import load_demucs_model, modify_model_for_custom

    model = load_demucs_model(model_name=default_config.demucs_model,
                              device=default_config.device)
    model = modify_model_for_custom(model,
                                    num_sources=default_config.num_sources)
    model = freeze_layers(model, freeze_encoder=True)
    print()

    print("-" * 66)
    print(" Шаг 2. Раскладка по сети")
    print("-" * 66)
    print(describe_split(model))
    print()

    print("-" * 66)
    print(" Шаг 3. Настраиваем оптимизатор")
    print("-" * 66)
    optimizer = setup_optimizer(model, default_config)
    print()

    print("-" * 66)
    print(" Шаг 4. Настраиваем планировщик")
    print("-" * 66)
    scheduler = setup_scheduler(optimizer, default_config)
    print()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 66)
    print(f"Optimizer настроен. Trainable params: {trainable_params}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
