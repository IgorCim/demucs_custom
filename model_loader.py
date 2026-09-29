# -*- coding: utf-8 -*-
"""
model_loader.py — загрузка готовой модели Demucs и её подготовка к дообучению.

Главная мысль всего проекта в одном абзаце: мы НЕ пишем Demucs с нуля.
Мы берём предобученную сеть (она уже умеет отделять вокал от барабанов
на миллионах часов музыки) и слегка подкручиваем её под наши данные.

Что здесь происходит по шагам:

1. load_demucs_model()      - скачивает/берёт из кэша готовые веса.
2. modify_model_for_custom() - если нужно, меняет число выходных дорожек.
3. freeze_layers()          - замораживает "тяжёлую" часть сети,
                               чтобы при дообучении менялся только decoder.

Проверено на demucs 4.1.0, модель htdemucs, 26.09.2026:

  * get_model("htdemucs") возвращает НЕ саму сеть, а обёртку BagOfModels.
    Внутри неё лежит HTDemucs. Обёртка нужна, чтобы усреднять несколько
    сетей, но вызывать её напрямую нельзя - её forward специально
    запрещён и просит использовать demucs.apply.apply_model().

  * Настоящая сеть HTDemucs разделена на две ветки:
        encoder / decoder  - работает во времени (как в обычной свёрточной сети)
        tencoder / tdecoder - работает со спектром (STFT)
    Плюс crosstransformer - общий "мозг", который их связывает.

  * Последний слой, который реально выдаёт дорожки:
        tdecoder[-1].conv_tr : ConvTranspose1d(48 -> 8, kernel=8, stride=4)
    Число 8 = 4 дорожки * 2 канала. Менять число дорожек нужно именно здесь.

  * Порядок дорожек у самой модели: drums, bass, other, vocals
    (НЕ наш vocals, drums, bass, other - см. config.py, комментарий на строке 97).

  * Размеры (всего 41 984 456 весов):
        encoder        1 424 940
        tencoder       1 424 172
        decoder        4 562 620
        tdecoder       2 209 460
        crosstransformer 31 550 464   <- три четверти всей сети
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from config import config, resolve_device

# Названия модулей, которые считаются "энкодером" и "декодером".
# У Demucs их по две штуки: ветка во времени и ветка в спектре.
# tdecoder важен отдельно: именно в нём лежит последний слой с дорожками.
ENCODER_PREFIXES = ("encoder", "tencoder")
DECODER_PREFIXES = ("decoder", "tdecoder")

# Модели, которые умеет грузить библиотека demucs.
KNOWN_MODELS = (
    "htdemucs",       # 4 дорожки, гибридная трансформерная (наша база)
    "htdemucs_ft",    # 4 дорожки, дообученная авторами на большем датасете
    "htdemucs_6s",    # 6 дорожек
    "demucs",         # 4 дорожки, без трансформера
    "mdx_extra_q",    # 4 дорожки, другой подход
)


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------


def inner_models(model: nn.Module) -> List[nn.Module]:
    """
    Возвращает список настоящих сетей внутри модели.

    get_model() отдаёт обёртку BagOfModels, а внутри неё - список сетей
    (у htdemucs он обычно один, у некоторых моделей - несколько).
    Нам править надо именно сети, а не обёртку.

    Если обёртки нет (кто-то передал готовую сеть) - вернём её саму.
    """
    inner = getattr(model, "models", None)
    if isinstance(inner, nn.ModuleList) and len(inner) > 0:
        return list(inner)
    return [model]


def get_sources(model: nn.Module) -> List[str]:
    """Настоящие названия дорожек модели. Порядок важен - не переставлять."""
    return list(getattr(model, "sources", []) or [])


def _fmt(number: int) -> str:
    """12345678 -> '12 345 678'. Просто чтобы легче было читать."""
    return f"{number:,}".replace(",", " ")


# Ветки декодера. У Demucs их обычно две: одна работает во времени,
# вторая - в спектре. Число дорожек влияет на ОБЕ, причём по-разному.
DECODER_BRANCHES = ("decoder", "tdecoder")

# Выходные слои бывают и 1D, и 2D: ветка tdecoder работает со временным
# рядом (ConvTranspose1d), а ветка decoder - сразу с частотой и временем
# (ConvTranspose2d). Проверено на htdemucs 26.09.2026.
CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)


class OutputLayer(NamedTuple):
    """Выходной слой одной ветки декодера."""
    branch: str        # имя ветки: "decoder" или "tdecoder"
    path: str          # полный путь, например "tdecoder.3.conv_tr"
    parent: nn.Module  # модуль-владелец слоя
    attr: str          # имя слоя внутри parent ("conv_tr")
    conv: nn.Module    # сам слой


def find_branch_outputs(net: nn.Module) -> List[OutputLayer]:
    """
    Находит выходной слой КАЖДОЙ ветки декодера.

    Их не одна, а две, и это главная хитрость, на которой спотыкается
    наивная реализация. Проверено на htdemucs 26.09.2026:

        decoder.3.conv_tr  ->  48 -> 16 каналов   (ветка во времени)
        tdecoder.3.conv_tr ->  48 ->  8 каналов   (ветка в спектре)

    При четырёх дорожках на выходе нужно 4 * 2 = 8 каналов. Спектральная
    ветка даёт ровно 8, а временная - 16, то есть вдвое больше.
    И в htdemucs.forward оба этих результата потом складываются:

        x  = x.view(B, S, -1, Fq, T)      <- ветка decoder
        xt = xt.view(B, S, -1, length)     <- ветка tdecoder

    где S = len(self.sources). Если поменять число дорожек и отредактировать
    только одну ветку, размеры перестанут делиться и сеть упадёт с
    "shape is invalid for input of size ...". Поэтому масштабировать
    нужно обе, и каждую - по-своему.

    Слой ищем по признаку last=True ( Demucs сам ставит его последнему
    слою ветки). Если признака нет - берём последний слой по списку.
    """
    found: List[OutputLayer] = []
    for branch in DECODER_BRANCHES:
        layers = getattr(net, branch, None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            continue

        picked: Optional[Tuple[int, nn.Module, nn.Module]] = None
        for index in range(len(layers) - 1, -1, -1):
            layer = layers[index]
            conv = getattr(layer, "conv_tr", None)
            if not isinstance(conv, CONV_TYPES):
                continue
            if picked is None:
                picked = (index, layer, conv)
            if getattr(layer, "last", False):
                picked = (index, layer, conv)
                break

        if picked is not None:
            index, layer, conv = picked
            found.append(
                OutputLayer(branch, f"{branch}.{index}.conv_tr",
                            layer, "conv_tr", conv)
            )
    return found


def _copy_conv_weights(old: nn.Module, new: nn.Module) -> None:
    """
    Переносит веса из старого слоя в новый.

    Это важно: если мы меняем 4 дорожки на 3, веса первых трёх
    мы сохраняем как есть. Модель уже "умеет" эти дорожки, не надо
    учить заново с нуля. Новые каналы (если дорожек стало больше)
    инициализируем с тем же разбросом, что у старых весов.

    ВАЖНО, где резать веса. У двух типов слоёв раскладка разная:
        Conv1d / Conv2d              weight = (выход, вход, ядро)
        ConvTranspose1d / 2d         weight = (вход, выход, ядро)
    Если резать не по той оси, модель не сломается с ошибкой, а тихо
    испортится - поэтому ось вычисляем явно.

    И не путать 1D с 2D: у htdemucs ветка tdecoder заканчивается
    ConvTranspose1d, а ветка decoder - ConvTranspose2d. Ось у них
    одинаковая, но проверить надо обе разновидности - иначе ветка
    decoder посчитает ось неверно и упадёт при копировании весов.
    """
    out_dim = 0 if isinstance(old, (nn.Conv1d, nn.Conv2d)) else 1
    rank = old.weight.dim()
    if new.weight.dim() != rank:
        return

    keep = min(old.weight.shape[out_dim], new.weight.shape[out_dim])

    take = [slice(None)] * rank
    take[out_dim] = slice(0, keep)
    with torch.no_grad():
        new.weight[tuple(take)].copy_(old.weight[tuple(take)])

        if new.weight.shape[out_dim] > old.weight.shape[out_dim]:
            # новые выходные каналы - шум с тем же масштабом
            extra = [slice(None)] * rank
            extra[out_dim] = slice(keep, new.weight.shape[out_dim])
            std = float(old.weight.std())
            nn.init.normal_(new.weight[tuple(extra)], mean=0.0,
                            std=std if std > 0 else 0.01)

        if old.bias is not None and new.bias is not None:
            keep_bias = min(old.bias.shape[0], new.bias.shape[0])
            new.bias[:keep_bias].copy_(old.bias[:keep_bias])
            if new.bias.shape[0] > old.bias.shape[0]:
                new.bias[old.bias.shape[0]:].zero_()


def _rebuild_norm(old_norm: nn.Module, new_channels: int) -> nn.Module:
    """Пересоздаёт слой нормализации под новое число каналов."""
    if isinstance(old_norm, nn.GroupNorm):
        new_norm = nn.GroupNorm(old_norm.num_groups, new_channels,
                                affine=old_norm.affine)
    elif isinstance(old_norm, nn.LayerNorm):
        new_norm = nn.LayerNorm(new_channels, eps=old_norm.eps)
    else:
        return old_norm

    with torch.no_grad():
        if getattr(old_norm, "weight", None) is not None:
            new_norm.weight.copy_(old_norm.weight[:new_channels])
        if getattr(old_norm, "bias", None) is not None:
            new_norm.bias.copy_(old_norm.bias[:new_channels])
    return new_norm


# ---------------------------------------------------------------------------
# ОСНОВНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------


def load_demucs_model(model_name: str = "htdemucs",
                      device: Optional[str] = None) -> nn.Module:
    """
    Загружает готовую предобученную модель Demucs.

    Ничего не обучает и не меняет - просто берёт веса, которые авторы
    demucs уже обучили на большом датасете. Первый раз модель скачивается
    из интернета (~80 МБ) и кладётся в кэш, дальше берётся оттуда.

    Возвращает модель, готовую к дообучению.
    """
    from demucs.pretrained import get_model

    if model_name not in KNOWN_MODELS:
        print(f"  [!] Модель '{model_name}' не в списке проверенных.")
        print(f"      Обычно доступны: {', '.join(KNOWN_MODELS)}")
        print("      Пробую загрузить как есть - вдруг список устарел.")

    target_device = resolve_device(device or config.device)

    try:
        model = get_model(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить модель '{model_name}'.\n"
            f"  Причина: {type(exc).__name__}: {exc}\n"
            f"  Проверь название модели. Доступные: "
            f"{', '.join(KNOWN_MODELS)}\n"
            f"  Если это первая загрузка, нужен интернет: веса качаются\n"
            f"  с серверов Hugging Face (файл ~80 МБ)."
        ) from exc

    # get_model возвращает модель в режиме eval(). Для дообучения позже
    # позовём model.train(), а пока оставим как есть - так безопаснее:
    # случайно запущенный наш код не испортит веса.
    model.eval()
    model.to(target_device)

    print(f"Загружена модель: {model_name}")

    inners = inner_models(model)
    print(f"  Обёртка      : {type(model).__name__}")
    print(f"  Сетей внутри : {len(inners)}"
          + (f" ({type(inners[0]).__name__})" if inners else ""))
    print(f"  Дорожки      : {', '.join(get_sources(model))}")
    print(f"  Устройство   : {target_device}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Параметров   : {_fmt(total)}")

    return model


def modify_model_for_custom(model: nn.Module, num_sources: int = 4
                            ) -> nn.Module:
    """
    Приводит модель к нужному числу дорожек.

    Если у модели уже num_sources дорожек - ничего не трогаем
    (для htdemucs с 4 дорожками это как раз наш случай).

    Если дорожек другое количество - перестраиваем выходные слои
    ОБЕИХ веток декодера (см. find_branch_outputs), сохраняя старые веса
    для уже знакомых дорожек: модель что-то умеет, не надо учить заново.

    НЕ НАДО искать "слой с 8 каналами" - в сети таких слоев нет.
    Дорожки делают два выходных слоя, у которых каналов разное число:
        decoder.3.conv_tr  -> S * каналы * 2
        tdecoder.3.conv_tr -> S * каналы
    Меняем оба, каждый пропорционально своему размеру.

    В конце модель реально прогоняется на секунде тишины: нельзя верить
    только подсчёту каналов. Сеть - это не одна свёртка, а цепочка с
    reshape'ами, и любая из них может разойтись с числом дорожек.
    """
    inners = inner_models(model)
    if not inners:
        raise ValueError("В модели не найдено ни одной сети.")

    if not isinstance(num_sources, int) or isinstance(num_sources, bool):
        raise ValueError(
            f"num_sources должно быть целым числом, а получилось: {num_sources!r}"
        )
    if num_sources < 1:
        raise ValueError(
            f"num_sources должно быть хотя бы 1, а получилось: {num_sources}.\n"
            f"  Ноль дорожек - не имеет смысла, сеть не сможет ничего выдать."
        )

    current_names = get_sources(model)
    current = len(current_names)
    audio_channels = int(getattr(inners[0], "audio_channels", 2))

    print(f"Подгоняю модель под {num_sources} дорожек "
          f"(сейчас {current}, каналов на дорожку: {audio_channels})")

    if current == num_sources:
        print(f"  Уже ровно {num_sources} дорожек: {', '.join(current_names)}")
        print("  Ничего менять не нужно.")
        _check_model_output(model, num_sources, audio_channels)
        return model

    # Новые названия: первые num_sources старых, а если надо больше -
    # добавляем нейтральные "source_N". Порядок старых НЕ переставляем.
    new_names = list(current_names[:num_sources])
    for index in range(len(new_names), num_sources):
        new_names.append(f"source_{index + 1}")

    # Каждая ветка декодера масштабируется по-своему, поэтому новую ширину
    # считаем из её собственной, а не из "дорожки * каналы".
    for net in inners:
        branches = find_branch_outputs(net)
        if not branches:
            raise RuntimeError(
                f"В сети {type(net).__name__} не нашлось ни одной ветки "
                f"декодера ({', '.join(DECODER_BRANCHES)}).\n"
                f"  Модель устроена не так, как ожидалось. Этот слой\n"
                f"  придётся указать вручную."
            )

        for out in branches:
            old_out = out.conv.out_channels
            new_out = int(round(old_out * num_sources / current))
            new_out = max(1, new_out)

            print(f"  Сеть {type(net).__name__}: {out.path}  "
                  f"{old_out} -> {new_out} каналов")

            new_conv = type(out.conv)(
                out.conv.in_channels,
                new_out,
                out.conv.kernel_size,
                out.conv.stride,
                out.conv.padding,
                out.conv.output_padding,
                out.conv.groups,
                out.conv.bias is not None,
                out.conv.dilation,
                out.conv.padding_mode,
            )
            _copy_conv_weights(out.conv, new_conv)
            setattr(out.parent, out.attr, new_conv)

            # Рядом с выходным слоем может стоять нормализация под старые
            # каналы - её тоже надо пересоздать, иначе будет ошибка.
            for sibling_name, sibling in out.parent.named_children():
                if sibling_name == out.attr:
                    continue
                if isinstance(sibling, (nn.GroupNorm, nn.LayerNorm)):
                    if getattr(sibling, "num_channels", None) == old_out:
                        setattr(out.parent, sibling_name,
                                _rebuild_norm(sibling, new_out))

        # Имена дорожек хранятся и в самой сети
        net.sources = list(new_names)

    # Обёртка тоже должна знать про новое число дорожек. Иначе
    # demucs.apply.apply_model() сломается: он умножает выход на
    # веса sources и пересчитывает totals по старому числу дорожек.
    if isinstance(getattr(model, "models", None), nn.ModuleList):
        model.sources = list(new_names)
        model.weights = [[1.0] * num_sources for _ in model.models]

    if current > num_sources:
        print(f"  Убрали дорожки: {', '.join(current_names[num_sources:])}")
    else:
        print(f"  Добавили дорожки: {', '.join(new_names[current:])}")
    print(f"  Старые веса сохранены для дорожек "
          f"{', '.join(new_names[:min(current, num_sources)])}")

    _check_model_output(model, num_sources, audio_channels)
    return model


def _module_device(module: nn.Module) -> torch.device:
    """
    Устройство, на котором реально лежат веса модуля.

    Нужно, чтобы тестовые прогонялки создавали входные данные ТАМ ЖЕ, где
    модель, а не там, где оказался default device. На машине без видеокарты
    это незаметно, а на GPU падает с невнятной ошибкой про несовпадение
    устройств.
    """
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _check_model_output(model: nn.Module, num_sources: int,
                        audio_channels: int) -> None:
    """
    Страховка: реально прогоняем сеть и смотрим форму выхода.

    Проверки "по слоям" мало: между выходным слоем и результатом лежат
    reshape'ы вида x.view(B, S, -1, Fq, T), где S = len(sources).
    Если число каналов не делится на число дорожек - падает уже не
    свёртка, а вся модель. Поэтому единственная надёжная проверка -
    пустить сигнал и посмотреть на форму.
    """
    nets = inner_models(model)
    was_training = [net.training for net in nets]
    for net in nets:
        net.eval()

    length = 44100
    try:
        with torch.no_grad():
            for index, net in enumerate(nets):
                # Устройство берём у самой сети, а не задаём по умолчанию.
                # Иначе на GPU тест кормит свёртку тензором с CPU и падает с
                # "Input type (torch.FloatTensor) and weight type
                #  (torch.cuda.FloatTensor) should be the same" - причём
                # задолго до начала обучения, на самой сборке модели.
                device = _module_device(net)
                out = net(torch.zeros(1, audio_channels, length, device=device))
                got = tuple(out.shape)
                want = (1, num_sources, audio_channels, length)
                if got != want:
                    raise RuntimeError(
                        f"После изменения модель выдаёт не то, что нужно.\n"
                        f"  Сеть #{index + 1} ({type(net).__name__}) вернула "
                        f"{got}\n"
                        f"  Ожидалось {want}\n"
                        f"  Число дорожек в сети: {len(net.sources)} "
                        f"({', '.join(net.sources)})"
                    )
    finally:
        for net, flag in zip(nets, was_training):
            net.train(flag)

    print(f"  Проверка: сеть реально выдаёт "
          f"{num_sources} дорожки по {audio_channels} канала (прогон на 1 сек)")


def freeze_layers(model: nn.Module, freeze_encoder: bool = True
                  ) -> nn.Module:
    """
    Готовит модель к дообучению: что учим, что не трогаем.

    Зачем это нужно. Модель уже обучена на огромном датасете. Если
    начать менять все её веса на маленьком наборе из 20-30 треков,
    она забудет всё, что знала (это называется "забыть" или catastrophic
    forgetting). Поэтому замораживаем большую часть сети и учим только
    decoder - он отвечает за то, как собирается готовая дорожка, и
    дешевле всего подстраивается под наш стиль музыки.

    Что считается чем:
        encoder, tencoder  -> энкодеры, замораживаем (freeze_encoder=True)
        decoder, tdecoder  -> декодеры, оставляем обучаемыми
                              (в tdecoder лежит последний слой с дорожками,
                              он обязан учиться)
        crosstransformer и прочее -> замораживаем, чтобы не переобучать
                              на маленьком датасете

    Печатает требуемую строку:
        Заморожено X параметров, разморожено Y параметров
    """
    inners = inner_models(model)
    if not inners:
        raise ValueError("В модели не найдено ни одной сети.")

    frozen = 0
    trainable = 0
    by_part: Dict[str, List[int]] = {}

    for net in inners:
        for name, param in net.named_parameters():
            part = name.split(".")[0]
            # tdecoder начинается с "tdecoder", но про "decoder" НЕ должен
            # считаться энкодером, поэтому сравниваем ровно верхний уровень.
            is_decoder = part in DECODER_PREFIXES
            is_encoder = part in ENCODER_PREFIXES

            if is_decoder:
                param.requires_grad_(True)
            elif freeze_encoder:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()

            stat = by_part.setdefault(part, [0, 0])
            stat[0 if param.requires_grad else 1] += param.numel()

    # Требуемая строка - ровно как в задании, без разделителей разрядов,
    # чтобы её можно было безнадёжно искать поиском по тексту.
    print(f"Заморожено {frozen} параметров, разморожено {trainable} параметров")

    total = frozen + trainable
    percent = 100.0 * trainable / total if total else 0.0
    if freeze_encoder:
        print(f"  Всего { _fmt(total) }, учится {percent:.1f}% сети.")
    else:
        print(f"  Всего { _fmt(total) }, заморожено {percent:.1f}% сети.")

    print("  По частям сети:")
    for part in sorted(by_part, key=lambda k: -sum(by_part[k])):
        tr, fr = by_part[part]
        mark = "учится" if tr > 0 else "заморожено"
        print(f"    {part:<16} {mark:<11} {_fmt(tr + fr):>13} весов")

    # Проверка, что энкодер действительно заморожен, а декодер - нет.
    for net in inners:
        checks = [(name, True) for name in DECODER_PREFIXES]
        checks += [(name, not freeze_encoder) for name in ENCODER_PREFIXES]
        for part, should_train in checks:
            module = getattr(net, part, None)
            if module is None:
                continue
            flags = {p.requires_grad for p in module.parameters()}
            if flags and flags != {should_train}:
                state = "учится" if should_train else "заморожен"
                print(f"  [!] Внимание: {part} не полностью {state}.")

    return model


def print_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Печатает и возвращает подробности о модели: что внутри, что заморожено,
    что учится. Этим же пользуется __main__.
    """
    inners = inner_models(model)
    device = resolve_device(config.device)

    print()
    print("=" * 66)
    print(" МОДЕЛЬ")
    print("=" * 66)
    print(f"  Тип обёртки   : {type(model).__name__}")
    for index, net in enumerate(inners):
        print(f"  Сеть №{index + 1}     : {type(net).__name__}")
    print(f"  Дорожки       : {', '.join(get_sources(model))}")
    print(f"  Их количество : {len(get_sources(model))}")
    print(f"  Каналов       : {getattr(inners[0], 'audio_channels', '?')}")
    print(f"  Частота       : {getattr(model, 'samplerate', '?')} Гц")
    print(f"  Устройство    : {device}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Параметров    : {_fmt(total)} всего, "
          f"{_fmt(trainable)} обучаемых")
    if total:
        print(f"  Обучается     : {100.0 * trainable / total:.1f}%")

    # Что реально будет меняться при обучении - самый важный вывод.
    if trainable:
        parts: List[str] = []
        for net in inners:
            for name, param in net.named_parameters():
                part = name.split(".")[0]
                if param.requires_grad and part not in parts:
                    parts.append(part)
        print(f"  Обучаются     : {', '.join(parts)}")
    else:
        print("  [!] Ни один параметр не обучается - модель не изменится.")

    return {
        "wrapper": type(model).__name__,
        "inner_classes": [type(net).__name__ for net in inners],
        "num_inner_models": len(inners),
        "sources": get_sources(model),
        "num_sources": len(get_sources(model)),
        "audio_channels": int(getattr(inners[0], "audio_channels", 2)),
        "samplerate": int(getattr(model, "samplerate", 0)),
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "device": device,
    }


def smoke_test(model: nn.Module, seconds: float = 1.0) -> bool:
    """
    Короткая проверка: прогоняет кусочек тишины через сеть и смотрит,
    что на выходе столько дорожек, сколько ожидалось.

    Для обёртки BagOfModels нужен apply_model, а не прямой вызов forward.
    """
    from demucs.apply import apply_model

    samplerate = int(getattr(model, "samplerate", 44100))
    length = int(samplerate * seconds)
    device = resolve_device(config.device)
    signal = torch.zeros(1, 2, length, device=device)

    try:
        with torch.no_grad():
            estimates = apply_model(model, signal, device=device)
    except Exception as exc:
        print(f"  [!] Проверка не удалась: {type(exc).__name__}: {exc}")
        return False

    if isinstance(estimates, (list, tuple)):
        got_sources = len(estimates)
        got_channels = estimates[0].shape[-2] if estimates else 0
    else:
        got_sources = estimates.shape[1]
        got_channels = estimates.shape[2]

    expected = len(get_sources(model))
    print(f"  Прогон {seconds:.1f} сек тишины через сеть:")
    print(f"    на выходе {got_sources} дорожек по {got_channels} канала")
    print(f"    ожидалось {expected}")

    if got_sources != expected:
        print(f"  [!] Несовпадение: {got_sources} вместо {expected}.")
        return False
    return True


# ---------------------------------------------------------------------------
# ПРОВЕРКА ПРИ ЗАПУСКЕ СКРИПТА
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 66)
    print(" demucs_custom / ЗАГРУЗКА МОДЕЛИ")
    print("=" * 66)
    print()
    print("  Мы не обучаем Demucs с нуля - берём готовую модель и дообучаем")
    print("  под свою музыку.")
    print()

    print("-" * 66)
    print(" Шаг 1. Загружаем готовую модель")
    print("-" * 66)
    model = load_demucs_model(model_name=config.demucs_model,
                              device=config.device)
    print()

    print("-" * 66)
    print(f" Шаг 2. Проверяем число дорожек (нужно {config.num_sources})")
    print("-" * 66)
    model = modify_model_for_custom(model, num_sources=config.num_sources)
    print()

    print("-" * 66)
    print(" Шаг 3. Замораживаем энкодер, оставляем обучаемым декодер")
    print("-" * 66)
    model = freeze_layers(model, freeze_encoder=True)
    print()

    print("-" * 66)
    print(" Шаг 4. Собираем подробности о модели")
    print("-" * 66)
    info = print_model_info(model)
    print()

    print("-" * 66)
    print(" Шаг 5. Проверяем, что сеть считает")
    print("-" * 66)
    ok = smoke_test(model, seconds=1.0)
    print()

    print("=" * 66)
    if ok:
        print(f"Готово. Модель '{config.demucs_model}' загружена и готова "
              f"к дообучению.")
        print(f"  Дорожек: {info['num_sources']} "
              f"({', '.join(info['sources'])})")
        print(f"  Обучается {info['trainable_params']} из "
              f"{info['total_params']} параметров")
        print()
        print("Следующий шаг: python train_custom.py")
    else:
        print("Что-то пошло не так - смотри сообщения выше.")
    print("=" * 66)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
