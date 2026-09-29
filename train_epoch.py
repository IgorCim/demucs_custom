# -*- coding: utf-8 -*-
"""
train_epoch.py — один проход по данным: эпоха обучения и эпоха проверки.

Это самая горячая часть обучения: здесь модель считает миллионы раз, и
именно тут обычно ломается всё остальное. Поэтому в файле два
предохранителя, каждый отвечает за свою известную беду.

ПРЕДОХРАНИТЕЛЬ 1. Модель нельзя звать как обычную сеть.

    pred = model(mixture)          # так написано в учебниках, и так НЕ будет работать

get_model("htdemucs") возвращает не саму сеть, а обёртку BagOfModels.
У неё forward() специально заблокирован:

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")

Это не поломка, а задумка авторов: apply_model() умеет нарезать длинный
файл на куски, обрабатывать с перекрытием и склеивать обратно, а прямой
forward этого не делает. Но apply_model() предназначен для РАЗДЕЛЕНИЯ
готового файла, а нам нужно обучение, где мы сами считаем потерю и
делаем шаг оптимизатора.

Поэтому здесь своя функция forward_sources(): она обходит обёртку и
звать сети внутри, складывая их результаты с весами - ровно так же, как
это делает apply_model, только без нарезки на куски.

ПРЕДОХРАНИТЕЛЬ 2. Градиенты надо обрезать.

Даже с маленькой скоростью обучения (1e-4) градиент иногда проскакивает
на два порядка больше обычного - достаточно одного неудачного батча с
резким сигналом, и optimizer.step() вносит в веса изменение в сто раз
больше, чем всё, что модель выучила за эпоту. clip_grad_norm_ обрезает
вектор градиентов до заданной длины, так что один плохой батч не может
испортить модель. Это не подстраховка на всякий случай - без неё
дообучение чужой модели регулярно ломается.

Что делает clip_grad_norm_ ровно:

    всего = sqrt(сумма квадратов всех градиентов)
    если всего > max_norm:
        все градиенты умножаются на max_norm / всего

Замороженные параметры в этой сумме не участвуют: у них grad = None,
и torch их пропускает. Поэтому обрезка не тратит время на 35 миллионов
мёртвых весов.

Проверено на torch 2.14.0, demucs 4.1.0 / htdemucs, 26.09.2026:

  * Модель отдаёт (B, S, C, T) = батч x дорожки x каналы x отсчёты.
  * Наш CustomDataset отдаёт СЛОВАРЬ с ключами "mixture" и "sources",
    а официальный demucs - кортеж. Здесь понимаются оба формата,
    а на любой другой будет понятная ошибка, а не падение через три
    строки кода.

  * Вызванная в тренировочном режиме сеть НЕ подрезает вход до
    своей длины segment: атрибут segment = 39/5 (это СЕКУНДЫ, то есть
    7.8 секунды = 343980 отсчётов), но на выходе получается ровно
    столько отсчётов, сколько подали на вход. Проверено на обоих
    режимах: вход 480000 -> выход 480000, вход 343980 -> выход 343980.
    Поэтому config.segment_length = 480000 работает без единой
    правки модели.

Как пользоваться:

    from train_epoch import train_one_epoch, validate_one_epoch

    train_loss = train_one_epoch(model, train_loader, optimizer,
                                 criterion, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)
    scheduler.step(val_loss)     # потери проверки, НЕ обучения
"""

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Максимальная длина вектора градиентов. Ровно то же значение стоит
# в config.grad_clip_norm, но здесь держим константой по той же причине,
# по которой weight_decay живёт в optimizer_setup.py: потерям нужно
# работать и без файла config.py.
GRAD_CLIP_NORM = 1.0


def forward_sources(model: nn.Module, mixture: torch.Tensor) -> torch.Tensor:
    """
    Считает сеть напрямую, в обход заблокированного forward обёртки.

    mixture : (B, C, T)  - смесь, батч x каналы x отсчёты
    возврат : (B, S, C, T) - предсказанные дорожки

    Если внутри несколько сетей (BagOfModels умеет усреднять), берём
    среднее с весами - так же, как это делает demucs.apply.apply_model.
    Для нашего htdemucs сеть одна, но код работает и для нескольких.
    """
    inners = getattr(model, "models", None)
    if not isinstance(inners, nn.ModuleList) or len(inners) == 0:
        # Обычная сеть без обёртки - можно звать напрямую.
        return model(mixture)

    weights = getattr(model, "weights", None)

    total: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None

    for index, net in enumerate(inners):
        out = net(mixture)

        if weights is not None and index < len(weights):
            w = weights[index]
            if not isinstance(w, torch.Tensor):
                w = torch.tensor(w, device=out.device, dtype=out.dtype)
            w = w.to(device=out.device, dtype=out.dtype)
            out = out * w.view(1, -1, 1, 1)
            weight_sum = w if weight_sum is None else weight_sum + w
        else:
            weight_sum = (torch.ones(out.shape[1], device=out.device,
                                     dtype=out.dtype)
                          if weight_sum is None
                          else weight_sum + 1.0)

        total = out if total is None else total + out

    if total is None:                       # на случай пустой обёртки
        raise RuntimeError(
            "В обёртке нет ни одной сети, нечего считать.\n"
            "  Проверь, что модель загрузилась: model_loader.load_demucs_model()"
        )

    if weight_sum is not None:
        total = total / weight_sum.view(1, -1, 1, 1)

    return total


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Достаёт из батча смесь и дорожки.

    Поддерживаются два формата, потому что в проекте встречаются оба:

        словарь  {"mixture": ..., "sources": ...}   - наш CustomDataset
        кортеж   (mixture, sources)                   - как в задании

    Что НЕ поддерживается: официальный demucs отдаёт одним куском
    (B, S+1, C, T), где на нулевом месте смесь, а дальше дорожки. Там
    смесь не сложить с целями наивно, и путать это с (B, S, C, T) -
    источник очень неприятных ошибок, поэтому мы такой формат
    не принимаем молча.
    """
    if isinstance(batch, dict):
        missing = [k for k in ("mixture", "sources") if k not in batch]
        if missing:
            raise KeyError(
                f"В батче нет ключей {', '.join(missing)}.\n"
                f"  Что есть: {', '.join(str(k) for k in batch)}\n"
                f"  Ожидались 'mixture' и 'sources' - так отдаёт CustomDataset."
            )
        return batch["mixture"], batch["sources"]

    if isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                f"В батче {len(batch)} элементов, а нужно минимум 2: "
                f"смесь и дорожки."
            )
        return batch[0], batch[1]

    raise TypeError(
        f"Не понимаю формат батча: {type(batch).__name__}.\n"
        f"  Ожидался словарь с ключами 'mixture' и 'sources'\n"
        f"  (так отдаёт data_prep.CustomDataset)\n"
        f"  или кортеж из двух тензоров (смесь, дорожки)."
    )


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Переносит тензор на нужное устройство, если он ещё не там."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Ожидался тензор torch, а пришло: {type(tensor).__name__}"
        )
    return tensor.to(device)


def train_one_epoch(model: nn.Module, dataloader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    criterion: Callable[..., torch.Tensor],
                    device: Union[str, torch.device],
                    max_norm: float = GRAD_CLIP_NORM,
                    verbose: bool = True) -> float:
    """
    Одна эпоха обучения. Возвращает среднюю потерю за эпоху.

    Порядок в батче важен, и он такой:

        1. model.train()                 - включить обучение (Dropout,
                                         нормализация по батчу)
        2. pred = сеть(mixture)          - предсказание
        3. loss = criterion(pred, targets)
        4. loss.backward()               - посчитать градиенты
        5. clip_grad_norm_               - обрезать слишком большой градиент
        6. optimizer.step()              - изменить веса
        7. optimizer.zero_grad()         - забыть старые градиенты

    Про zero_grad в конце, а не в начале: с градиентами, начатыми с
    нуля (set_to_none=True), результат тот же, зато память после шага
    освобождается сразу - а это сотни мегабайт при наших размерах.

    max_norm - до какой длины обрезать градиент (по умолчанию 1.0).

    Если потеря вдруг стала NaN или бесконечной, батч пропускается:
    веса не трогаются, а в консоль выводится предупреждение. Так одно
    битое место в данных не убьёт всё обучение, но заметно будет.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Ожидалась nn.Module, а пришло: {type(model).__name__}"
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            f"Ожидался torch.optim.Optimizer, а пришло: "
            f"{type(optimizer).__name__}"
        )

    dev = torch.device(device)
    model.to(dev)
    model.train()          # <- пункт 1: режим обучения

    total_loss = 0.0
    total_items = 0
    total_grad_norm = 0.0
    grad_norms = 0
    skipped = 0
    started = time.time()

    for step, batch in enumerate(dataloader):
        mixture, sources = unpack_batch(batch)
        mixture = _to_device(mixture, dev)      # <- пункт a
        sources = _to_device(sources, dev)

        optimizer.zero_grad(set_to_none=True)

        # 2. предсказание. Не model(mixture) - обёртка так не умеет,
        #    см. описание файла.
        pred = forward_sources(model, mixture)

        # 3. потеря
        loss = criterion(pred, sources)

        # Числовое значение потери берём ОДИН РАЗ и сразу отсоединяем от
        # графа вычислений. Иначе torch ругается предупреждением
        # "Converting a tensor with requires_grad=True to a scalar" на
        # каждом батче - за тысячи батчей такой шум забивает настоящие
        # предупреждения, и в него перестаёшь вслушиваться.
        loss_value = float(loss.detach())

        if not math.isfinite(loss_value):
            skipped += 1
            if skipped <= 3:
                print(f"  [!] Пропущен батч {step}: потеря получилась "
                      f"{loss_value}. Скорее всего, в данных есть тишина "
                      f"или обрыв.")
            continue

        # 4. обратное распространение
        loss.backward()

        # 5. обрезка градиента - см. описание файла
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_norm
        )
        grad_norm_value = float(grad_norm) if grad_norm is not None else 0.0
        if grad_norm is not None and torch.isfinite(grad_norm):
            total_grad_norm += grad_norm_value
            grad_norms += 1

        # 6. шаг оптимизатора
        optimizer.step()

        # 7. забыть градиенты (set_to_none=True - память освободится)
        optimizer.zero_grad(set_to_none=True)

        # Считаем среднее по ВСЕМ примерам, а не среднее по средним:
        # последний батч обычно меньше остальных, и иначе он повлиял бы
        # на результат сильнее, чем должен.
        batch_items = sources.shape[0]
        total_loss += loss_value * batch_items
        total_items += batch_items

        if verbose:
            print(f"  батч {step + 1:>4}: loss={loss_value:.6f}  "
                  f"grad={grad_norm_value:.3f}")

    if total_items == 0:
        raise RuntimeError(
            "Ни один батч не отработал - нечего усреднять.\n"
            "  Проверь, что dataloader не пустой, а потери считаются.\n"
            f"  Пропущено батчей: {skipped}."
        )

    average = total_loss / total_items
    elapsed = time.time() - started

    if verbose:
        print(f"  Эпоха обучения: средняя потеря = {average:.6f} "
              f"за {elapsed:.1f} сек на {total_items} примерах")
        if grad_norms:
            print(f"  Средняя длина градиента = "
                  f"{total_grad_norm / grad_norms:.3f} (обрезаем до {max_norm})")
        if skipped:
            print(f"  [!] Пропущено битых батчей: {skipped}")

    return average


def validate_one_epoch(model: nn.Module, dataloader: Iterable,
                       criterion: Callable[..., torch.Tensor],
                       device: Union[str, torch.device],
                       verbose: bool = True) -> float:
    """
    Одна эпоха проверки. Возвращает среднюю потерю.

    Отличия от обучения, и они принципиальные:

        model.eval()      - выключить обучение. Нормализация по батчу
                            перестаёт следить за батчем, иначе результат
                            зависел бы от того, с какими соседями
                            случайно попал в одну пачку. Dropout
                            отключается - предсказание становится
                            повторяемым.
        torch.no_grad()   - не строить граф вычислений. Экономит
                            примерно половину памяти: без графа не
                            нужно хранить всё, что участвовало
                            в вычислении, чтобы посчитать градиенты.

    Градиенты здесь не нужны и не считаются, поэтому нет ни backward,
    ни обрезки, ни шага оптимизатора. Проверка ничего не меняет в модели.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Ожидалась nn.Module, а пришло: {type(model).__name__}"
        )

    dev = torch.device(device)
    model.to(dev)
    model.eval()          # <- режим оценки

    total_loss = 0.0
    total_items = 0
    skipped = 0
    started = time.time()

    with torch.no_grad():                  # <- экономия памяти
        for step, batch in enumerate(dataloader):
            mixture, sources = unpack_batch(batch)
            mixture = _to_device(mixture, dev)
            sources = _to_device(sources, dev)

            pred = forward_sources(model, mixture)
            loss = criterion(pred, sources)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            batch_items = sources.shape[0]
            total_loss += float(loss) * batch_items
            total_items += batch_items

            if verbose:
                print(f"  батч {step + 1:>4}: loss={float(loss):.6f}")

    if total_items == 0:
        raise RuntimeError(
            "Ни один батч проверки не отработал - нечего усреднять.\n"
            "  Возможно, dataloader пустой. Пропущено: "
            f"{skipped}."
        )

    average = total_loss / total_items
    elapsed = time.time() - started

    if verbose:
        print(f"  Проверка: средняя потеря = {average:.6f} "
              f"за {elapsed:.1f} сек на {total_items} примерах")
        if skipped:
            print(f"  [!] Пропущено битых батчей: {skipped}")

    return average


# ---------------------------------------------------------------------------
# СИНТЕТИЧЕСКИЕ ДАННЫЕ ДЛЯ ПРОВЕРКИ
# ---------------------------------------------------------------------------


class SyntheticSeparationDataset(torch.utils.data.Dataset):
    """
    Фальшивые данные для проверки цикла обучения.

    Настоящие треки для этого не нужны, и хорошо: проверка должна быть
    быстрой и не зависеть от того, что лежит в папке train/.

    Каждый "трек" - это:
        mixture (2, T)      сумма четырёх дорожек
        sources (4, 2, T)   эти четыре дорожки по отдельности

    Сумма дорожек РОВНО равна смеси (плюс крошечный шум), как и в
    настоящих данных. Это важно: если бы смесь была случайной, модель
    не смогла бы выучить ничего, и потери только бы показывали, что
    всё сломалось.

    Отдаём словарь с теми же ключами, что и настоящий
    data_prep.CustomDataset, - чтобы проверить совместимость.
    """

    def __init__(self, num_tracks: int = 4, num_sources: int = 4,
                 channels: int = 2, length: int = 44100,
                 seed: int = 0) -> None:
        self.num_tracks = num_tracks
        self.num_sources = num_sources
        self.channels = channels
        self.length = length
        self.seed = seed

    def __len__(self) -> int:
        return self.num_tracks

    def __getitem__(self, index: int) -> Dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed + index)

        def noise(*shape: int) -> torch.Tensor:
            return torch.randn(*shape, generator=generator) * 0.1

        sources = torch.stack([
            noise(self.channels, self.length)
            for _ in range(self.num_sources)
        ])                                            # (S, C, T)

        # Немного разной частоты у каждой дорожки, чтобы они не были
        # одинаковым шумом - так проверка ближе к настоящей задаче.
        for s in range(self.num_sources):
            t = torch.linspace(0, 1, self.length)
            tone = torch.sin(2 * torch.pi * (110 * (s + 1)) * t)
            sources[s] = sources[s] + tone * 0.05

        mixture = sources.sum(dim=0) + noise(self.channels, self.length) * 0.001

        return {
            "mixture": mixture,      # (C, T)
            "sources": sources,      # (S, C, T)
            "track": f"synthetic_{index:03d}",
        }


def main() -> int:
    print("=" * 66)
    print(" train_epoch / ОДНА ЭПОХА")
    print("=" * 66)
    print()
    print("  Проверяем не качество модели, а что цикл обучения честно")
    print("  работает: считает потери, обновляет веса и НЕ трогает")
    print("  замороженные.")
    print()

    device = torch.device("cpu")

    print("-" * 66)
    print(" Шаг 1. Модель")
    print("-" * 66)
    from model_loader import (
        freeze_layers,
        load_demucs_model,
        modify_model_for_custom,
    )

    model = load_demucs_model()
    model = modify_model_for_custom(model, num_sources=4)
    model = freeze_layers(model, freeze_encoder=True)
    print()

    print("-" * 66)
    print(" Шаг 2. Данные (синтетические)")
    print("-" * 66)
    train_set = SyntheticSeparationDataset(num_tracks=4, length=44100)
    valid_set = SyntheticSeparationDataset(num_tracks=2, length=44100, seed=100)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=False, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=2, shuffle=False, num_workers=0
    )
    print(f"  обучение : {len(train_set)} примера по 1 сек, "
          f"батчами по 2")
    print(f"  проверка : {len(valid_set)} примера по 1 сек, "
          f"батчами по 2")
    print(f"  формат   : {train_set[0]['mixture'].shape} -> "
          f"{train_set[0]['sources'].shape}")
    print()

    print("-" * 66)
    print(" Шаг 3. Потеря, оптимизатор, планировщик")
    print("-" * 66)
    from custom_loss import CustomLoss, sdr_loss
    from optimizer_setup import setup_optimizer, setup_scheduler
    from config import config

    criterion = CustomLoss(lambda_freq=0.5, lambda_time=0.5)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    print()

    print("-" * 66)
    print(" Шаг 4. Снимок весов ДО обучения")
    print("-" * 66)
    # НАСТОЯЩИЙ снимок: detach().clone() - отдельная память. Просто
    # state_dict() не годится: он отдаёт те же самые тензоры, что и
    # модель, и сравнение всегда будет показывать "ничего не изменилось".
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    print(f"  снято {len(before)} тензоров")
    print()

    print("-" * 66)
    print(" Шаг 5. Эпоха обучения")
    print("-" * 66)
    train_loss = train_one_epoch(model, train_loader, optimizer,
                                 criterion, device)
    print()

    print("-" * 66)
    print(" Шаг 6. Что изменилось в весах")
    print("-" * 66)
    after = {n: p.detach().clone() for n, p in model.named_parameters()}
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    changed = {n for n in before if not torch.equal(before[n], after[n])}
    changed_frozen = changed & frozen_names
    changed_trainable = changed & trainable_names

    print(f"  замороженных тензоров : {len(frozen_names)}")
    print(f"  обучаемых тензоров     : {len(trainable_names)}")
    print(f"  изменилось всего       : {len(changed)}")
    print(f"    замороженных ПОМЕНЯЛОСЬ : {len(changed_frozen)}  "
          f"{sorted(changed_frozen)[:2]}")
    print(f"    обучаемых ПОМЕНЯЛОСЬ    : {len(changed_trainable)}")
    if changed_frozen:
        print("  [!] ОШИБКА: замороженные веса изменились!")
    else:
        print("  Замороженные веса не тронуты - всё как надо.")
    print()

    print("-" * 66)
    print(" Шаг 7. Эпоха проверки")
    print("-" * 66)
    valid_loss = validate_one_epoch(model, valid_loader, criterion, device)
    print()

    print("-" * 66)
    print(" Шаг 8. Планировщик и метрика качества")
    print("-" * 66)
    scheduler.step(valid_loss)
    print(f"  scheduler.step(val_loss) -> lr = "
          f"{optimizer.param_groups[0]['lr']:.2e}")

    # Разовая оценка качества на всех проверочных примерах сразу.
    model.eval()
    with torch.no_grad():
        real_losses, sdrs = [], []
        for index in range(len(valid_set)):
            item = valid_set[index]
            pred = forward_sources(model, item["mixture"].unsqueeze(0))
            real_losses.append(
                float(criterion(pred, item["sources"].unsqueeze(0)))
            )
            sdrs.append(float(sdr_loss(pred, item["sources"].unsqueeze(0))))
    print(f"  потеря на проверке (по одному примеру): "
          f"{sum(real_losses) / len(real_losses):.6f}")
    print(f"  sdr_loss (меньше = лучше): "
          f"{sum(sdrs) / len(sdrs):+.2f}  (это случайные данные, "
          f"хорошего результата ждать нельзя)")
    print()

    print("=" * 66)
    print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
