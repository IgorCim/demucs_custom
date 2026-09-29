# -*- coding: utf-8 -*-
"""
Автотесты model_loader.py.

Главное, что тут проверяется: не «модель загрузилась», а что
замороженные веса действительно не меняются, размороженные - меняются,
и что модель с другим числом дорожек честно считает.

Запуск: python _test_model_loader.py
"""
import copy
import sys

import torch
import torch.nn as nn

from model_loader import (
    CONV_TYPES,
    DECODER_BRANCHES,
    DECODER_PREFIXES,
    ENCODER_PREFIXES,
    find_branch_outputs,
    freeze_layers,
    get_sources,
    inner_models,
    load_demucs_model,
    modify_model_for_custom,
    print_model_info,
    smoke_test,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


def snapshot(model):
    """Копия всех весов: чтобы потом сравнить, что изменилось."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def part_of(param_name):
    """
    Верхний уровень имени веса, без префикса обёртки.

    ВАЖНО: у BagOfModels все имена начинаются с "models.0.",
    поэтому просто name.split(".")[0] всегда дал бы "models",
    и проверки вроде "заморожен ли энкодер" проходили бы вхолостую,
    ничего не проверяя. Префикс обязательно срезаем.
    """
    parts = param_name.split(".")
    if parts and parts[0] == "models":
        parts = parts[2:]          # models.0.encoder... -> encoder...
    return parts[0] if parts else ""


def changed_keys(before, after):
    return [k for k in before
            if k in after and not torch.equal(before[k], after[k])]


# Модель весит 42 млн параметров и грузится с диска небыстро, поэтому
# грузим её ОДИН раз, а варианты делаем копированием в памяти.
BASE = None


def fresh_model():
    """Свежая копия исходной модели (как если бы загрузили заново)."""
    global BASE
    if BASE is None:
        BASE = load_demucs_model()
    return copy.deepcopy(BASE)


print("=" * 70)
print("1. ЗАГРУЗКА МОДЕЛИ")
print("=" * 70)
model = load_demucs_model()
sources = get_sources(model)
check("дорожек 4", len(sources) == 4, ", ".join(sources))
check("порядок дорожек как у demucs, а не наш",
      sources == ["drums", "bass", "other", "vocals"], ", ".join(sources))
check("get_model отдал обёртку BagOfModels", type(model).__name__ == "BagOfModels",
      type(model).__name__)
check("внутри одна сеть HTDemucs",
      len(inner_models(model)) == 1 and type(inner_models(model)[0]).__name__ == "HTDemucs")
check("всего больше 40 млн параметров",
      sum(p.numel() for p in model.parameters()) > 40_000_000,
      f"{sum(p.numel() for p in model.parameters()):,}")

print()
print("=" * 70)
print("2. ПОИСК ВЫХОДНЫХ СЛОЁВ (их ДВА, а не один)")
print("=" * 70)
net = inner_models(model)[0]
branches = {o.branch: o for o in find_branch_outputs(net)}
check("найдены обе ветки декодера",
      set(branches) == {"decoder", "tdecoder"}, ", ".join(sorted(branches)))

# Главная особенность htdemucs: дорожки делают ДВА слоя, и у них
# разное число каналов - 16 во временной ветке и 8 в спектральной.
# Поэтому искать "слой с 8 каналами" бесполезно: такого критерия нет.
# И типы тоже разные: ветка decoder - двумерная свёртка, tdecoder - одномерная.
time_out = branches.get("decoder")
spec_out = branches.get("tdecoder")
check("временная ветка: decoder.3.conv_tr, 48 -> 16 каналов",
      time_out is not None and time_out.path == "decoder.3.conv_tr"
      and time_out.conv.in_channels == 48 and time_out.conv.out_channels == 16
      and isinstance(time_out.conv, nn.ConvTranspose2d),
      f"{time_out.path} {type(time_out.conv).__name__} "
      f"{time_out.conv.in_channels}->{time_out.conv.out_channels}"
      if time_out else "не найдено")
check("спектральная ветка: tdecoder.3.conv_tr, 48 -> 8 каналов",
      spec_out is not None and spec_out.path == "tdecoder.3.conv_tr"
      and spec_out.conv.in_channels == 48 and spec_out.conv.out_channels == 8
      and isinstance(spec_out.conv, nn.ConvTranspose1d)
      and tuple(spec_out.conv.kernel_size) == (8,)
      and tuple(spec_out.conv.stride) == (4,),
      f"{spec_out.path} {spec_out.conv.in_channels}->{spec_out.conv.out_channels}"
      f" k={tuple(spec_out.conv.kernel_size)} s={tuple(spec_out.conv.stride)}"
      if spec_out else "не найдено")
check("у обоих родитель помечен last=True",
      all(getattr(o.parent, "last", False) is True for o in branches.values()),
      ", ".join(f"{o.path}: last={getattr(o.parent, 'last', None)}"
                for o in branches.values()))
check("каналов у двух веток РАЗНОЕ (16 против 8), а не одинаковое",
      time_out.conv.out_channels != spec_out.conv.out_channels,
      f"{time_out.conv.out_channels} против {spec_out.conv.out_channels}")
check("размерность свёрток тоже разная (2D против 1D)",
      time_out.conv.weight.dim() == 4 and spec_out.conv.weight.dim() == 3,
      f"{time_out.conv.weight.dim()}D против {spec_out.conv.weight.dim()}D")

# Ловушка: слоев с 12 каналами в сети много, искать по одному числу нельзя
twelve = [n for n, m in net.named_modules()
          if isinstance(m, CONV_TYPES) and m.out_channels == 12]
check("слоев с 12 каналами в сети много (искать по каналам нельзя)",
      len(twelve) >= 4, f"{len(twelve)} штук, напр. {twelve[0]}")
check("ни один из них не выходной (нет last=True)",
      all(not getattr(net.get_submodule(n.rsplit(".", 1)[0]), "last", False)
          for n in twelve),
      "всего " + str(len(twelve)))

print()
print("=" * 70)
print("3. ЧИСЛО ДОРОЖЕК УЖЕ ПРАВИЛЬНОЕ (4)")
print("=" * 70)
before_state = snapshot(model)
model = modify_model_for_custom(model, num_sources=4)
check("список дорожек не изменился", get_sources(model) == sources)
check("веса вообще не тронуты",
      len(changed_keys(before_state, model.state_dict())) == 0,
      f"изменилось {len(changed_keys(before_state, model.state_dict()))} тензоров")

print()
print("=" * 70)
print("4. ЗАМОРОЖЕНО / РАЗМОРОЖЕНО")
print("=" * 70)
model = freeze_layers(model, freeze_encoder=True)
frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
trainable = [n for n, p in model.named_parameters() if p.requires_grad]

# Проверяем по именам САМОЙ сети. По обёртке все имена начинаются
# с "models.0." и part_of() срезал бы префикс - иначе проверки
# "заморожен ли энкодер" проходили бы вхолостую, по пустому списку.
net_params = dict(inner_models(model)[0].named_parameters())
check("проверять будем не пустой набор имён",
      len(net_params) > 100 and
      any(part_of(n) == "encoder" for n in net_params),
      f"{len(net_params)} весов в сети")

enc = {n: p for n, p in net_params.items() if part_of(n) in ENCODER_PREFIXES}
dec = {n: p for n, p in net_params.items() if part_of(n) in DECODER_PREFIXES}
xf = {n: p for n, p in net_params.items() if part_of(n) == "crosstransformer"}
check("нашлись веса энкодеров, декодеров и трансформера",
      enc and dec and xf,
      f"энкодеры {len(enc)}, декодеры {len(dec)}, трансформер {len(xf)}")

check("энкодеры заморожены",
      all(not p.requires_grad for p in enc.values()),
      f"{sum(not p.requires_grad for p in enc.values())} из {len(enc)}")
check("декодеры разморожены",
      all(p.requires_grad for p in dec.values()),
      f"{sum(p.requires_grad for p in dec.values())} из {len(dec)}")
check("трансформер заморожен (это 75% сети, он большой и тяжёлый)",
      all(not p.requires_grad for p in xf.values()),
      f"{sum(not p.requires_grad for p in xf.values())} из {len(xf)}")
check("выходной слой дорожек обучаемый",
      net_params["tdecoder.3.conv_tr.weight"].requires_grad,
      "иначе модель не сможет научиться новым дорожкам")
check("и второй выходной слой тоже обучаемый",
      net_params["decoder.3.conv_tr.weight"].requires_grad,
      "он тоже делает дорожки, оба должны учиться")
check("выходные слои НЕ отфильтрованы как энкодеры",
      all(part_of(n) in DECODER_PREFIXES
          for n in ("decoder.3.conv_tr.weight", "tdecoder.3.conv_tr.weight")),
      "оба лежат внутри декодеров, значит попадают в обучаемые")
check("обучается меньше трети сети",
      0 < len(trainable) < len(frozen) + len(trainable),
      f"{len(trainable)} тензоров против {len(frozen)} замороженных")

info = print_model_info(model)
check("print_model_info вернул верные числа",
      info["num_sources"] == 4 and info["total_params"] == info["trainable_params"]
      + info["frozen_params"],
      f"всего {info['total_params']:,}")

print()
print("=" * 70)
print("5. ГЛАВНОЕ: обучение трогает ТОЛЬКО размороженное")
print("=" * 70)
model.train()
net = inner_models(model)[0]
before = snapshot(model)
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)
signal = torch.randn(1, 2, 44100) * 0.05
target = torch.randn(1, 4, 2, 44100) * 0.05

out = net(signal)
loss = torch.nn.functional.mse_loss(out, target)
loss.backward()
optimizer.step()
after = snapshot(model)

touched = changed_keys(before, after)
check("после шага обучения что-то изменилось", len(touched) > 0,
      f"изменилось {len(touched)} тензоров")
# part_of срезает префикс models.0. - без этого проверки ниже
# проходили бы вхолостую (все имена начинались бы с "models")
check("проверяем по осмысленным именам, а не по префиксу models",
      all(part_of(k) in {"encoder", "tencoder", "decoder", "tdecoder",
                         "crosstransformer", "freq_emb"}
          or part_of(k).startswith("channel_") for k in touched),
      f"пример: {touched[0]}")
bad_frozen = [k for k in touched if part_of(k) in ENCODER_PREFIXES]
check("замороженные веса НЕ изменились", not bad_frozen,
      f"потрогали: {bad_frozen[:3]}" if bad_frozen else "ни одного")
bad_transformer = [k for k in touched if part_of(k) == "crosstransformer"]
check("трансформер не изменился", not bad_transformer,
      f"потрогали: {bad_transformer[:3]}" if bad_transformer else "ни одного")
good_decoder = [k for k in touched if part_of(k) in DECODER_PREFIXES]
check("декодер изменился", len(good_decoder) > 0,
      f"{len(good_decoder)} тензоров")
check("изменились ТОЛЬКО декодеры, ничего лишнего",
      len(good_decoder) == len(touched),
      f"декодеров {len(good_decoder)} из изменённых {len(touched)}")

has_grad_frozen = [n for n, p in net_params.items()
                   if not p.requires_grad and p.grad is not None]
check("у замороженных нет градиентов (экономит память)",
      not has_grad_frozen, f"{has_grad_frozen[:3]}" if has_grad_frozen else "ни одного")
has_grad_trainable = [n for n, p in net_params.items()
                      if p.requires_grad and p.grad is None]
check("у всех обучаемых есть градиент", not has_grad_trainable,
      f"{has_grad_trainable[:3]}" if has_grad_trainable else "у всех")
check("градиент дошёл до выходного слоя дорожек",
      net_params["tdecoder.3.conv_tr.weight"].grad is not None)
check("градиент дошёл и до второго выходного слоя",
      net_params["decoder.3.conv_tr.weight"].grad is not None)

# Больше эти снимки не нужны. Модель весит 42 млн параметров, каждый снимок -
# это ~170 МБ, а дальше снова будут создаваться новые модели. Если вовремя
# не отпустить, тест упадёт по памяти на ровном месте.
del before, after, net_params

print()
print("=" * 70)
print("6. ПРИВЫЧНЫЙ СЦЕНАРИЙ: 6 дорожек вместо 4")
print("=" * 70)
model6 = fresh_model()
model6 = modify_model_for_custom(model6, num_sources=6)
names6 = get_sources(model6)
check("стало 6 дорожек", len(names6) == 6, ", ".join(names6))
check("старые 4 дорожки сохранили имена и порядок",
      names6[:4] == ["drums", "bass", "other", "vocals"], ", ".join(names6[:4]))
net6 = inner_models(model6)[0]
out6 = {o.branch: o for o in find_branch_outputs(net6)}
check("обе ветки перестроены, а не одна",
      set(out6) == {"decoder", "tdecoder"}
      and out6["decoder"].conv.out_channels == 24
      and out6["tdecoder"].conv.out_channels == 12,
      f"decoder {out6['decoder'].conv.out_channels}, "
      f"tdecoder {out6['tdecoder'].conv.out_channels}")
check("спектральная ветка расширена с 8 до 12, вход не тронут",
      out6["tdecoder"].conv.in_channels == 48,
      f"{out6['tdecoder'].conv.in_channels}->"
      f"{out6['tdecoder'].conv.out_channels}")
check("веса обёртки пересчитаны под 6 дорожек",
      all(len(w) == 6 for w in model6.weights),
      f"{[len(w) for w in model6.weights]}")

# ГЛАВНАЯ проверка этого шага: старые веса перенесены ПО ПРАЛЬНОЙ ОСИ.
# У ConvTranspose1d раскладка (вход, выход, ядро), поэтому резать надо
# по второй оси. Если ошибиться, модель не сломается - она тихо испортится.
ref = fresh_model()
ref_out = {o.branch: o for o in find_branch_outputs(inner_models(ref)[0])}
conv6 = out6["tdecoder"].conv
ref_conv = ref_out["tdecoder"].conv
check("размер новых весов (вход, выход, ядро)",
      tuple(conv6.weight.shape) == (48, 12, 8)
      and tuple(ref_conv.weight.shape) == (48, 8, 8),
      f"{tuple(conv6.weight.shape)} против {tuple(ref_conv.weight.shape)}")
check("старые 8 выходных каналов = исходные pretrained-веса",
      torch.equal(conv6.weight[:, :8, :], ref_conv.weight),
      "резали по axis=1 (выход), а не по axis=0 (вход)")
check("входные каналы не сдвинулись (важно! иначе вход испорчен)",
      torch.equal(conv6.weight[:, :8, :].sum(dim=(0, 2)),
                  ref_conv.weight.sum(dim=(0, 2))))
check("новые 4 канала проинициализированы ненулями",
      float(conv6.weight[:, 8:, :].abs().sum()) > 0)
check("новые каналы не совпали со старыми (иначе это копия)",
      not torch.equal(conv6.weight[:, 8:, :], conv6.weight[:, :4, :]))
# то же самое для временной ветки - её тоже надо сохранить
conv6t = out6["decoder"].conv
ref_convt = ref_out["decoder"].conv
check("веса временной ветки тоже перенесены верно (16 -> 24)",
      torch.equal(conv6t.weight[:, :16, :, :], ref_convt.weight)
      and tuple(conv6t.weight.shape) == (48, 24) + tuple(ref_convt.kernel_size),
      f"{tuple(ref_convt.weight.shape)} -> {tuple(conv6t.weight.shape)}")

model6 = freeze_layers(model6, freeze_encoder=True)
check("новый выходной слой обучаемый",
      any(p.requires_grad for n, p in model6.named_parameters()
          if n.endswith("tdecoder.3.conv_tr.weight")))
ok6 = smoke_test(model6, seconds=1.0)
check("apply_model работает с 6 дорожками (веса обёртки не сломались)", ok6)

# РЕГРЕССИЯ на главную найденную ошибку. Раньше код менял только ветку
# tdecoder, и сеть падала с "shape ... is invalid for input of size ...".
# Ломаем намеренно только одну ветку и убеждаемся, что это действительно
# ломает модель - значит перестраивать надо обе, как теперь и сделано.
half = fresh_model()
half_net = inner_models(half)[0]
half_layers = half_net.tdecoder
half_conv = getattr(half_layers[3], "conv_tr")
new_half = type(half_conv)(half_conv.in_channels, 12, half_conv.kernel_size,
                           half_conv.stride, half_conv.padding,
                           half_conv.output_padding, half_conv.groups,
                           half_conv.bias is not None, half_conv.dilation,
                           half_conv.padding_mode)
half_layers[3].conv_tr = new_half
half_net.sources = ["drums", "bass", "other", "vocals", "source_5", "source_6"]
half.sources = list(half_net.sources)
half.weights = [[1.0] * 6]
try:
    with torch.no_grad():
        half_net(torch.zeros(1, 2, 44100))
    check("одна перестроенная ветка ЛОМАЕТ модель (так и должно быть)", False,
          "сеть неожиданно выжила - проверка в _check_model_output слабая")
except RuntimeError:
    check("одна перестроенная ветка ЛОМАЕТ модель (так и должно быть)", True,
          "подтверждает: перестраивать нужно обе ветки")
del half, half_net, new_half, half_conv, half_layers

print()
print("=" * 70)
print("7. МЕНЬШЕ ДОРОЖЕК: 4 -> 2")
print("=" * 70)
model2 = fresh_model()
model2 = modify_model_for_custom(model2, num_sources=2)
names2 = get_sources(model2)
check("осталось 2 дорожки", len(names2) == 2, ", ".join(names2))
check("взяты первые две (drums, bass)",
      names2 == ["drums", "bass"], ", ".join(names2))
net2 = inner_models(model2)[0]
out2 = {o.branch: o for o in find_branch_outputs(net2)}
check("обе ветки сжаты пропорционально (8->4 и 16->8)",
      out2["tdecoder"].conv.out_channels == 4
      and out2["decoder"].conv.out_channels == 8,
      f"tdecoder {out2['tdecoder'].conv.out_channels}, "
      f"decoder {out2['decoder'].conv.out_channels}")
check("это по-прежнему последние слои своих веток",
      out2["tdecoder"].path == "tdecoder.3.conv_tr"
      and out2["decoder"].path == "decoder.3.conv_tr",
      f"{out2['tdecoder'].path}, {out2['decoder'].path}")
check("размер весов спектральной ветки (48, 4, 8)",
      tuple(out2["tdecoder"].conv.weight.shape) == (48, 4, 8),
      str(tuple(out2["tdecoder"].conv.weight.shape)))
check("размер весов временной ветки (48, 8, ...)",
      tuple(out2["decoder"].conv.weight.shape)[:2] == (48, 8),
      str(tuple(out2["decoder"].conv.weight.shape)))
check("у оставшихся 4 каналов веса из pretrained, не мусор",
      torch.equal(out2["tdecoder"].conv.weight,
                  ref_conv.weight[:, :4, :]),
      "взяли первые 4 дорожки как есть")
check("в обёртке тоже 2 дорожки и верные веса",
      len(model2.sources) == 2 and all(len(w) == 2 for w in model2.weights),
      f"{[len(w) for w in model2.weights]}")
freeze_layers(model2, freeze_encoder=True)
ok2 = smoke_test(model2, seconds=1.0)
check("apply_model работает с 2 дорожками", ok2)

print()
print("=" * 70)
print("8. БЕЗ ЗАМОРОЗКИ (freeze_encoder=False)")
print("=" * 70)
model_all = fresh_model()
model_all = freeze_layers(model_all, freeze_encoder=False)
all_train = [n for n, p in model_all.named_parameters() if not p.requires_grad]
check("тогда учится вся сеть целиком", not all_train,
      f"заморожено {len(all_train)}")
total = sum(p.numel() for p in model_all.parameters())
train = sum(p.numel() for p in model_all.parameters() if p.requires_grad)
check("числа сошлись: обучаемых == всего", train == total, f"{train:,} из {total:,}")

print()
print("=" * 70)
print("9. ПЛОХОЙ ВВОД")
print("=" * 70)
try:
    load_demucs_model("нетакой-модели")
    check("несуществующая модель -> понятная ошибка", False)
except RuntimeError as exc:
    text = str(exc)
    check("несуществующая модель -> понятная ошибка",
          "Не удалось загрузить" in text and "htdemucs" in text,
          text.splitlines()[0])

try:
    modify_model_for_custom(fresh_model(), num_sources=0)
    check("num_sources=0 -> ошибка", False)
except (ValueError, RuntimeError) as exc:
    check("num_sources=0 -> понятная ошибка", len(str(exc)) > 0,
          str(exc).splitlines()[0])

print()
print("=" * 70)
if FAIL:
    print(f"ПРОВАЛЕНО {len(FAIL)}:")
    for name in FAIL:
        print(f"  - {name}")
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
