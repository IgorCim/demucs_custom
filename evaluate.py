# -*- coding: utf-8 -*-
"""
demucs_custom / evaluate.py
===========================

ЧТО ДЕЛАЕТ ЭТОТ ФАЙЛ (задача):
    Измеряет, насколько хорошо наша дообученная модель разделяет музыку.
    Без цифр мы не поймём, стало ли лучше после обучения или мы испортили модель.

ГЛАВНЫЕ МЕТРИКИ:
    SDR  (Signal-to-Distortion Ratio) — «отношение сигнал/искажение».
         Главная метрика в мире разделения музыки. Измеряется в дБ.
         Больше = лучше. У Demucs (htdemucs) на хороших данных обычно ~7-8 дБ.

    SI-SNR (Scale-Invariant SNR) — похожая метрика, не зависит от общей
         громкости. Тоже в дБ, больше = лучше.

    Pearson r — корреляция между тем, что выдала модель, и эталоном.
         Чем ближе к 1.0, тем лучше.

КАК ЭТО РАБОТАЕТ:
    1. Берём эталонные дорожки из custom_dataset (vocals/drums/bass/other).
    2. Пропускаем микс через inference.py — получаем предсказание модели.
    3. Считаем метрики между предсказанием и эталоном.
    4. Выводим таблицу и сохраняем результат в JSON.

ЗАПУСК:
    python evaluate.py --dry-run     # только план
    python evaluate.py               # посчитать метрики
    python evaluate.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config


# ---------------------------------------------------------------------------
# МЕТРИКИ
# ---------------------------------------------------------------------------

def compute_sdr(estimate, reference, eps: float = 1e-8) -> float:
    """
    Signal-to-Distortion Ratio (в дБ).

    Идея: представим эталон как сумму «то, что модель уловила» + «ошибка».
    Тогда SDR = 10 * log10(сила_сигнала / сила_ошибки).

    estimate  — numpy-массив, то что выдала модель
    reference — numpy-массив, эталон (настоящая дорожка)
    """
    # TODO: реализовать SDR
    # Подсказка:
    #   signal = estimate - (ошибка)  # проекция reference на estimate
    #   проще: noise = reference - estimate
    #           sdr = 10*log10(sum(reference**2) / (sum(noise**2) + eps))
    raise NotImplementedError("TODO: compute_sdr() ещё не написан")


def compute_sisnr(estimate, reference, eps: float = 1e-8) -> float:
    """
    Scale-Invariant SNR (в дБ). Похожа на SDR, но устойчива к изменению
    общей громкости: если модель выдала дорожку в полтора раза тише,
    метрика этого не сочтёт за ошибку.
    """
    # TODO: реализовать SI-SNR
    # Подсказка:
    #   scale = sum(estimate*reference) / (sum(estimate**2) + eps)
    #   e_target = scale * reference; e_noise = estimate - e_target
    #   sisnr = 10*log10(sum(e_target**2) / (sum(e_noise**2) + eps))
    raise NotImplementedError("TODO: compute_sisnr() ещё не написан")


def compute_pearson(estimate, reference, eps: float = 1e-8) -> float:
    """
    Коэффициент корреляции Пирсона между предсказанием и эталоном.
    Значение от -1 до 1; для хорошей модели ожидаем 0.7-0.95.
    """
    # TODO: реализовать корреляцию (numpy.corrcoef) или вручную
    raise NotImplementedError("TODO: compute_pearson() ещё не написан")


# ---------------------------------------------------------------------------
# ОЦЕНКА
# ---------------------------------------------------------------------------

def evaluate_source(
    estimate,
    reference,
) -> Dict[str, float]:
    """
    Считает все метрики для ОДНОЙ дорожки.
    Возвращает словарь: {"sdr_db": ..., "sisnr_db": ..., "pearson": ...}
    """
    return {
        # TODO: подставить реальные вызовы функций выше
        "sdr_db": 0.0,
        "sisnr_db": 0.0,
        "pearson": 0.0,
    }


def evaluate_model(limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Оценивает модель на проверочной выборке.

    limit — ограничить количество треков (например 5, чтобы не ждать час).
    dry_run — только показать план.

    Возвращает словарь с результатами (и его же сохраняет в JSON).
    """
    print("=" * 70)
    print(" demucs_custom / ОЦЕНКА КАЧЕСТВА")
    print("=" * 70)

    model_path = config.output_model_path_abs
    print(f"Модель        : {model_path}")
    print(f"Данные        : {config.custom_data_path_abs}")
    print(f"Дорожки       : {', '.join(config.sources)}")
    print(f"Устройство    : {config.device}")

    if not model_path.exists():
        print()
        print(f"[!] Модель не найдена: {model_path}")
        print("    Сначала обучи модель: python train_custom.py")
        return {}

    val_dir = config.custom_data_path_abs / "mixture"
    n_files = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
    print(f"Файлов для проверки: {n_files}")

    if n_files == 0:
        print()
        print("[!] Проверочные данные не найдены, оценивать нечего.")
        return {}

    if limit:
        print(f"Ограничение: первые {limit} треков")

    if dry_run:
        print()
        print("Режим --dry-run: метрики не считались, показан только план.")
        return {}

    # TODO: полный цикл оценки:
    #   1. Загрузить модель (inference.load_model)
    #   2. Для каждого трека из mixture:
    #        - предсказать дорожки через inference.separate
    #        - сравнить с эталоном из папок vocals/drums/bass/other
    #        - посчитать SDR / SI-SNR / корреляцию
    #   3. Усреднить метрики по всем трекам
    #   4. Напечатать таблицу и сохранить в evaluate_report.json
    print()
    print("[!] Это заглушка: метрики ещё не считаются.")
    return {}


def format_report(results: Dict[str, Any]) -> str:
    """
    Превращает результаты в аккуратную таблицу для вывода в консоль
    и для записи в лог.
    """
    if not results:
        return "(нет результатов)"

    # TODO: сделать красивую таблицу по дорожкам
    return json.dumps(results, ensure_ascii=False, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Разбирает ключи командной строки."""
    parser = argparse.ArgumentParser(
        description="demucs_custom: оценка качества модели",
    )
    parser.add_argument("--model", default=None,
                        help="путь к модели (.pth)")
    parser.add_argument("--limit", type=int, default=None,
                        help="сколько треков проверить (например 5)")
    parser.add_argument("--out", default=None,
                        help="куда сохранить отчёт (JSON)")
    parser.add_argument("--dry-run", action="store_true",
                        help="только показать план")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа: разбор аргументов и запуск оценки."""
    args = parse_args(argv)

    if args.model:
        config.output_model_path = args.model

    results = evaluate_model(limit=args.limit, dry_run=args.dry_run)
    if not results:
        return 1

    print(format_report(results))

    if args.out:
        # TODO: сохранить отчёт в JSON (json.dump)
        print(f"Отчёт сохранён: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
