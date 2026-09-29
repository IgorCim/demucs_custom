# -*- coding: utf-8 -*-
"""
custom_loss.py — своя функция потерь для разделения музыки на дорожки.

Зачем она нужна. В штатном Demucs потеря очень простая: только L1 во
временной области, больше ничего (проверено в demucs 4.1.0, solver.py:325):

    loss = F.l1_loss(estimate, sources)

Для нашей задачи этого мало. L1 во времени говорит «посчитай ошибку по
каждому отсчёту», но музыка — это не просто набор отсчётов. Ошибка
«вокал сдвинут на 5 мс» даёт потерю, сравнимую с ошибкой «вокал
вырезан полностью», хотя для слушателя это совершенно разные вещи.
Мелкие огрехи вроде сдвига фазы или потери нескольких верхних
частот в спектре вообще плохо видны во временной области.

Поэтому складываем две потери:

    lambda_time * L1_во_времени + lambda_freq * L1_в_спектре

Временная часть отвечает за общее «похоже на оригинал», спектральная
учит тонкую структуру: где-то сдвинулась фаза, пропал верхний диапазон,
изменился баланс частот. Одной части всегда не хватает.

Проверено на demucs 4.1.0 / htdemucs, 26.09.2026:

  * Модель отдаёт тензор формы (B, S, C, T), то есть ЧЕТЫРЕ измерения:
        батч x дорожки x каналы x отсчёты
    Например (1, 4, 2, 44100) на секунде стерео.

  * torch.stft умеет принимать ТОЛЬКО одномерный или двумерный вход.
    Наш четырёхмерный тензор он не примет - будет ошибка. Поэтому перед
    STFT все лишние измерения (кроме времени) склеиваем в батч, а потом
    считаем среднее. Ровно так делает сам demucs (demucs/spec.py:12).

  * Параметры STFT берём те же, что у Demucs: окно Ханна, center=True,
    pad_mode='reflect' и - главное - normalized=True. Именно
    normalized=True держит амплитуды в спектре того же порядка, что и
    сигнал во времени. Без него частотная часть выдавала бы числа в
    сотни раз больше временной, и при lambda 0.5/0.5 временная часть
    просто не влияла бы на обучение (см. раздел про подбор весов).

Как пользоваться:

    from custom_loss import CustomLoss, sdr_loss

    criterion = CustomLoss(lambda_freq=0.5, lambda_time=0.5)
    criterion = criterion.to(device)

    estimate = model(mixture)          # (B, S, C, T)
    loss = criterion(estimate, target) # то же самое, что estimate-target
    loss.backward()

    quality = sdr_loss(estimate, target)  # чем больше, тем лучше (со знаком минус)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Параметры STFT по умолчанию. Похожи на Demucs, но окно короче:
# у самой htdemucs n_fft=4096, это много для функции потерь - она считается
# каждый шаг обучения на каждом батче, и память растёт как n_fft * T.
# 1024 при 44100 Гц даёт шаг сетки частот около 43 Гц - достаточно, чтобы
# различить, скажем, скрипку от барабанов, и при этом экономно.
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256          # перекрытие в 4 раза - стандарт для музыки
DEFAULT_EPS = 1e-8


def _check_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    """
    Проверяет, что pred и target вообще можно сравнивать.

    Звучит занудно, но это самый частый источник непонятных ошибок в
    обучении: модель отдала 4 дорожки, а эталон лежит с 3, или у цели
    другая длина. Ошибка вылезет через три слоя кода как
    "RuntimeError: shape mismatch", и непонятно где.
    """
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError(
            "Ожидались тензоры torch, а пришли: "
            f"pred={type(pred).__name__}, target={type(target).__name__}"
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"Размеры не совпадают.\n"
            f"  pred   : {tuple(pred.shape)}\n"
            f"  target : {tuple(target.shape)}\n"
            f"  Ожидаем одинаковые: батч x дорожки x каналы x отсчёты.\n"
            f"  Проверь, что число дорожек у модели и у данных одно и то же\n"
            f"  (у htdemucs по умолчанию 4 дорожки)."
        )
    if pred.is_complex():
        raise ValueError("Ожидался обычный (вещественный) звук, а пришёл комплексный.")
    if pred.dim() < 2:
        raise ValueError(
            f"Слишком мало измерений: {tuple(pred.shape)}.\n"
            f"  Ожидается хотя бы (батч, отсчёты), а в проекте обычно\n"
            f"  (батч, дорожки, каналы, отсчёты)."
        )
    if pred.shape[-1] < 2:
        raise ValueError(f"Слишком короткий сигнал: {pred.shape[-1]} отсчётов.")


def _flatten_for_stft(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Готовит сигнал к STFT: склеивает всё, кроме времени, в батч.

    torch.stft принимает только 1D или 2D, а у нас (B, S, C, T) -
    четыре измерения. Возвращает (склеенный сигнал, исходные измерения),
    чтобы вызывающий потом смог при желании вернуть исходную форму.
    """
    lead = tuple(x.shape[:-1])
    length = x.shape[-1]
    return x.reshape(-1, length), lead


def stft_magnitude(x: torch.Tensor, n_fft: int = DEFAULT_N_FFT,
                   hop_length: int = DEFAULT_HOP,
                   win_length: Optional[int] = None,
                   window: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Модуль спектрограммы: |STFT(x)|.

    Возвращает вещественный тензор той же формы по времени, что и на входе:
        (B, S, C, частоты, кадры)

    Почему берём модуль, а не сам комплексный спектр. Ошибка в фазе
    (сдвиг на пару отсчётов) почти не влияет на музыку, но в комплексном
    спектре она даёт большую ошибку. Модуль отражает то, что реально слышно:
    сколько энергии в каждой частоте. Сравнение фаз - забота другой функции.
    """
    if win_length is None:
        win_length = n_fft

    if x.shape[-1] < n_fft:
        raise ValueError(
            f"Сигнал короче окна STFT: {x.shape[-1]} отсчётов, "
            f"а нужно минимум {n_fft}.\n"
            f"  Увеличь n_fft или нарежь более длинные куски "
            f"(config.segment_length)."
        )

    if window is None:
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)

    flat, _ = _flatten_for_stft(x)

    spec = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=True,      # как в demucs: амплитуды остаются в разумном масштабе
        onesided=True,
        return_complex=True,
    )
    return spec.abs()


class CustomLoss(nn.Module):
    """
    Потеря = lambda_freq * L1(спектр) + lambda_time * L1(время).

    Обе части считаются через модуль: torch.nn.functional.l1_loss
    с усреднением по всем измерениям (среднее, не сумма - иначе значение
    зависело бы от размера батча и нельзя было бы сравнивать запуски).

    Аргументы:
        lambda_freq — вес частотной части (спектрограмма), по умолчанию 0.5
        lambda_time — вес временной части (сырой звук), по умолчанию 0.5
        n_fft, hop_length, win_length — параметры STFT
        eps — защита от деления на ноль

    Как выбирать веса. Значения по умолчанию 0.5/0.5 - это стартовая точка,
    а не готовый ответ. Два слагаемых измеряются в разных величинах, и их
    масштабы не совпадают: временная L1 лежит примерно в масштабе амплитуды
    сигнала, а спектральная - заметно меньше. Поэтому сначала посмотри на
    реальные числа (forward с return_parts=True печатает оба слагаемых),
    и если одно из них тонет в шуме другого - увеличь его вес.

Сколько это стоит по памяти. Замерено 26.09.2026 на процессоре, при
    config.batch_size=8, 4 дорожки, 2 канала, config.segment_length=480000:
    одна прямая передача этой потери - около 1.5 ГБ RAM и 0.9 сек.
    Обратный проход примерно вдвое дороже, и это СВЕРХ того, что ест сама
    модель. Если памяти не хватает, есть три рычага по порядку:
        уменьшить config.batch_size,
        увеличить hop_length (меньше кадров - меньше памяти),
        уменьшить n_fft.
    При lambda_freq=0 частотная часть не считается вообще, и потеря
    становится совсем дешёвой.
    """

    def __init__(self, lambda_freq: float = 0.5, lambda_time: float = 0.5,
                 n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP,
                 win_length: Optional[int] = None,
                 eps: float = DEFAULT_EPS) -> None:
        super().__init__()

        for name, value in (("lambda_freq", lambda_freq),
                            ("lambda_time", lambda_time)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} должно быть числом, а не {type(value).__name__}")
            if value < 0:
                raise ValueError(
                    f"{name} = {value} - отрицательным быть не может.\n"
                    f"  Отрицательный вес значит «штрафовать за точность»."
                )
            if value != value:  # NaN
                raise ValueError(f"{name} = NaN - так нельзя.")

        if lambda_freq == 0 and lambda_time == 0:
            raise ValueError(
                "Оба веса равны нулю - потеря всегда будет 0,\n"
                "  и модель не станет учиться ничему."
            )

        if n_fft <= 0 or hop_length <= 0:
            raise ValueError(
                f"n_fft и hop_length должны быть положительными, "
                f"получили n_fft={n_fft}, hop_length={hop_length}."
            )
        if win_length is None:
            win_length = n_fft
        if not (0 < win_length <= n_fft):
            raise ValueError(
                f"win_length должен быть от 1 до n_fft, "
                f"получили win_length={win_length}, n_fft={n_fft}."
            )
        if eps <= 0:
            raise ValueError(f"eps должен быть положительным, получили {eps}.")

        self.lambda_freq = float(lambda_freq)
        self.lambda_time = float(lambda_time)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.eps = float(eps)

        # Окно Ханна - буфер, а не обычный атрибут. Так при .to(device)
        # и .cuda() оно уедет на GPU вместе с потерей. Если бы это был
        # просто атрибут, на GPU пришлось бы каждый раз создавать окно
        # заново, а забытый .to(device) ронял бы обучение с невнятной
        # ошибкой про несовпадение устройств.
        self.register_buffer("window", torch.hann_window(self.win_length),
                             persistent=False)

    def extra_repr(self) -> str:
        return (f"lambda_freq={self.lambda_freq}, "
                f"lambda_time={self.lambda_time}, "
                f"n_fft={self.n_fft}, hop_length={self.hop_length}")

    def frequency_loss(self, pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
        """L1 между модулями спектрограмм."""
        pred_spec = stft_magnitude(pred, self.n_fft, self.hop_length,
                                   self.win_length, self.window)
        target_spec = stft_magnitude(target, self.n_fft, self.hop_length,
                                     self.win_length, self.window)
        return F.l1_loss(pred_spec, target_spec)

    def time_loss(self, pred: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
        """L1 между сигналами как есть."""
        return F.l1_loss(pred, target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                return_parts: bool = False):
        """
        Считает взвешенную сумму.

        pred, target — одинаковой формы (B, S, C, T).
        return_parts=True вернёт ещё и словарь с обеими частями по
        отдельности - удобно для отладки весов и для логов обучения.
        """
        _check_pair(pred, target)

        # Окно лежит в self.window, но если потерю не перенесли на
        # устройство вместе с данными, подставим нужное здесь.
        window = self.window
        if window.device != pred.device or window.dtype != pred.dtype:
            window = window.to(device=pred.device, dtype=pred.dtype)

        time_part = F.l1_loss(pred, target)

        if self.lambda_freq == 0:
            # Частотную часть не считаем вовсе - STFT на большом батче
            # это заметная часть времени шага обучения.
            freq_part = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            pred_spec = stft_magnitude(pred, self.n_fft, self.hop_length,
                                       self.win_length, window)
            target_spec = stft_magnitude(target, self.n_fft, self.hop_length,
                                         self.win_length, window)
            freq_part = F.l1_loss(pred_spec, target_spec)

        loss = self.lambda_freq * freq_part + self.lambda_time * time_part

        if return_parts:
            return loss, {
                "total": loss,
                "freq": freq_part,
                "time": time_part,
            }
        return loss


def sdr_loss(pred: torch.Tensor, target: torch.Tensor,
             eps: float = DEFAULT_EPS, reduction: str = "mean") -> torch.Tensor:
    """
    Отрицательный SDR - как мера качества, но со знаком для минимизации.

    SDR (Signal-to-Distortion Ratio) отвечает на вопрос «насколько чисто
    модель вытащила дорожку», в децибелах. Чем больше - тем лучше.
    Обычно для разделения музыки считают 5-15 дБ, приличный результат
    15-20 дБ.

    Считаем по проекции. Идея: представим эталон как сумму того, что
    модель уловила, и остатка ошибки. Направление остатка ищем
    ортогональным к предсказанию, иначе результат зависел бы от
    случайной громкости:

        alpha  = <эталон, предсказание> / ||предсказание||^2
        ошибка = эталон - alpha * предсказание
        SDR    = 10 * log10(||эталон||^2 / (||ошибка||^2 + eps))

    Почему с проекцией, а не просто ||эталон - предсказание||. Без
    проекции метрика ругалась бы на модель, которая выдала дорожку
    тише в полтора раза, - хотя звучит она при этом ровно так же.
    Проекция убирает общий уровень громкости и оставляет только
    искажения формы сигнала.

    Возвращает МИНУС SDR, потому что дальше это значение обычно
    подмешивают к общей потере, а оптимизатор умеет только уменьшать.
    Поэтому "чем больше - тем лучше" читается как "чем меньше loss,
    тем чище звук".

    reduction:
        "mean" - одно число на весь батч (обычный случай)
        "none" - тензор формы (батч,), по значению на каждый пример
    """
    _check_pair(pred, target)

    if reduction not in ("mean", "none"):
        raise ValueError(
            f"reduction может быть 'mean' или 'none', а получили {reduction!r}."
        )

    # Считаем по каждому примеру батча отдельно: сворачиваем всё,
    # кроме первого измерения.
    reduce_dims = tuple(range(1, pred.dim()))

    estimate_power = pred.pow(2).sum(dim=reduce_dims, keepdim=True)
    reference_power = target.pow(2).sum(dim=reduce_dims, keepdim=True)
    cross = (target * pred).sum(dim=reduce_dims, keepdim=True)

    # Проекция эталона на направление предсказания.
    alpha = cross / (estimate_power + eps)
    noise = target - alpha * pred
    noise_power = noise.pow(2).sum(dim=reduce_dims, keepdim=True)

    sdr = 10.0 * torch.log10((reference_power + eps) / (noise_power + eps))

    if reduction == "mean":
        return -sdr.mean()
    # По одному значению на пример батча, без лишних единичных измерений.
    return -sdr.reshape(-1)


def demo() -> None:
    """
    Показывает потеря в работе на случайных данных.

    Не обучает ничего - просто показывает, что счётчики живые и что
    функция ведёт себя так, как ожидается.
    """
    torch.manual_seed(0)

    batch, sources, channels, length = 2, 4, 2, 44100  # 1 секунда стерео

    print("=" * 66)
    print(" custom_loss / ПРОВЕРКА ПОТЕРЬ")
    print("=" * 66)
    print()
    print(f"Форма данных: ({batch}, {sources}, {channels}, {length})")
    print("  батч x дорожки x каналы x отсчёты (1 сек при 44100 Гц)")
    print()

    target = torch.randn(batch, sources, channels, length) * 0.1
    pred = target + torch.randn_like(target) * 0.03   # близко, но не точно

    criterion = CustomLoss(lambda_freq=0.5, lambda_time=0.5)
    print("Параметры потери:")
    print(f"  {criterion}")
    print()

    loss = criterion(pred, target)
    _, parts = criterion(pred, target, return_parts=True)

    print("Случайный предсказатель (совсем не умеет):")
    noise = torch.randn_like(target)
    print(f"  потеря            = {criterion(noise, target):.6f}")
    print()
    print("Предсказание рядом с эталоном:")
    print(f"  потеря            = {float(loss):.6f}")
    print(f"  частотная часть   = {float(parts['freq']):.6f}  "
          f"(вес {criterion.lambda_freq})")
    print(f"  временная часть   = {float(parts['time']):.6f}  "
          f"(вес {criterion.lambda_time})")
    print()

    # --- проверки, что функция себя ведёт как задумано ---
    print("Проверки:")

    same = float(criterion(target, target))
    print(f"  предсказание == эталон: потеря = {same:.10f}")
    assert same < 1e-7, "на одинаковых сигналах потеря должна быть нулевой"

    # Только временная часть
    only_time = float(CustomLoss(lambda_freq=0.0, lambda_time=1.0)(pred, target))
    # Только частотная часть
    only_freq = float(CustomLoss(lambda_freq=1.0, lambda_time=0.0)(pred, target))
    print(f"  lambda_time=1, lambda_freq=0: {only_time:.6f} "
          f"(совпало с временной частью: {abs(only_time - float(parts['time'])) < 1e-7})")
    print(f"  lambda_freq=1, lambda_time=0: {only_freq:.6f} "
          f"(совпало с частотной частью: {abs(only_freq - float(parts['freq'])) < 1e-7})")

    # Взвешивание должно быть линейным
    half = float(CustomLoss(lambda_freq=0.5, lambda_time=0.5)(pred, target))
    manual = 0.5 * only_freq + 0.5 * only_time
    print(f"  взвешенная сумма сходится: {half:.6f} против {manual:.6f}")
    assert abs(half - manual) < 1e-6, "сумма должна сходиться"

    # Чем больше шума, тем больше потеря - потеря вообще что-то меряет
    worse = float(criterion(target + torch.randn_like(target) * 0.3, target))
    print(f"  больше шума - больше потеря: {float(loss):.6f} -> {worse:.6f}")
    assert worse > float(loss), "потеря обязана расти вместе с ошибкой"

    print()

    # --- метрика качества ---
    sdr_good = float(sdr_loss(pred, target))
    sdr_bad = float(sdr_loss(torch.randn_like(target), target))
    sdr_exact = float(sdr_loss(target, target))

    print("Метрика качества. ВНИМАНИЕ НА ЗНАК:")
    print("  сам SDR (децибелы)  : больше = лучше")
    print("  sdr_loss (мы его)   : МЕНЬШЕ = лучше, потому что возвращается -SDR")
    print()
    print(f"  точное попадание   : {sdr_exact:+8.2f}")
    print(f"  небольшая ошибка  : {sdr_good:+8.2f}   <- эквивалентно SDR = "
          f"{-sdr_good:.2f} дБ")
    print(f"  случайный шум     : {sdr_bad:+8.2f}   <- эквивалентно SDR = "
          f"{-sdr_bad:.2f} дБ")
    print()
    print("  Случайный шум даёт около 0 дБ, и это правильно: он никак не")
    print("  связан с эталоном, поэтому проекция на него нулевая, ошибка")
    print("  равна самому эталону, и дБ выходит ноль.")

    # Порядок должен быть таким: точное попадание < небольшая ошибка < шум.
    # Убывание - обратная сторона того, что мы возвращаем -SDR.
    assert sdr_exact < sdr_good, "точное попадание обязано быть лучше ошибки"
    assert sdr_good < sdr_bad, "ошибка обязана быть лучше случайного шума"
    # Случайный шум - это примерно 0 дБ, то есть sdr_loss около нуля.
    assert abs(sdr_bad) < 0.5, (
        f"случайный шум должен давать примерно 0 дБ, а получилось {sdr_bad:.4f}"
    )

    print()
    print(f"ИТОГ: loss = {float(loss):.6f}")
    print("=" * 66)


if __name__ == "__main__":
    demo()
