# -*- coding: utf-8 -*-
"""
demucs_custom / config.py
=========================

ЕДИНЫЙ ФАЙЛ КОНФИГУРАЦИИ проекта.

Зачем он нужен:
    Чтобы не искать числа по всему коду. Все гиперпараметры (sample rate,
    размер батча, скорость обучения и т.д.) хранятся здесь, в одном месте.
    Захотел поменять скорость обучения — открыл config.py, поправил одно
    число, и это изменение подхватили все скрипты сразу.

Как пользоваться (из любого другого файла проекта):
    from config import config
    print(config.sample_rate)              # 44100
    print(config.output_model_path)        # ./models/custom_demucs.pth

Проверить, что файл в порядке (должно вывести таблицу настроек):
    python config.py

ВАЖНО:
    В этом файле НЕТ тяжёлых импортов (torch, demucs и т.д.).
    Благодаря этому `python config.py` работает даже до установки библиотек.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# СЛУЖЕБНЫЕ КОНСТАНТЫ
# ---------------------------------------------------------------------------

# PROJECT_ROOT — абсолютный путь к папке проекта (там, где лежит этот файл).
# Нужен для того, чтобы относительные пути ("./models/...") работали
# при запуске скрипта из любой директории, а не только из корня проекта.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# SOURCES — список дорожек (источников), которые мы хотим получить на выходе.
# Порядок важен: он соответствует порядку выходов нейросети Demucs.
#   vocals — вокал (голос)
#   drums  — барабаны/перкуссия
#   bass   — бас-гитара и низкие частоты
#   other  — всё остальное (гитары, клавиши, эффекты, шум)
SOURCES: List[str] = ["vocals", "drums", "bass", "other"]

# Модели, которые официально поддерживает библиотека demucs.
# "htdemucs" — Hybrid Transformer Demucs, самая качественная из четырёх,
#             умеет отделять 4 дорожки (vocals/drums/bass/other).
# Подробнее: https://github.com/facebookresearch/demucs
SUPPORTED_DEMUCS_MODELS: List[str] = [
    "htdemucs",   # 4 дорожки, гибридная трансформерная модель (наша база)
    "htdemucs_ft",# 4 дорожки, версия дообученная авторами на большем датасете
    "htdemucs_6s",# 6 дорожек (добавляются гитара и фортепиано)
    "demucs",     # 4 дорожки, старая RNN-архитектура (быстрее, хуже качество)
]

# Допустимые устройства вычислений.
#   "cpu"  — работает везде, но медленно (наш стартовый вариант)
#   "cuda" — видеокарта NVIDIA, в десятки раз быстрее
#   "mps"  — чипы Apple Silicon (MacBook с M1/M2/M3)
ALLOWED_DEVICES: List[str] = ["cpu", "cuda", "mps"]


# ---------------------------------------------------------------------------
# КЛАСС CONFIG — ВСЕ ГИПЕРПАРАМЕТРЫ ПРОЕКТА
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Все настройки проекта в одном месте.

    Используется как обычный dataclass: можно создать копию с изменениями
    для эксперимента, не трогая основной конфиг.

        cfg = Config(device="cuda", epochs=10)   # временный эксперимент
    """

    # ------------------ МОДЕЛЬ -------------------------------------------
    # Базовая архитектура Demucs, которую мы берём и дообучаем на своих данных.
    # "htdemucs" — стандартный выбор: 4 дорожки, хорошее качество, MIT-лицензия.
    demucs_model: str = "htdemucs"

    # Сколько дорожек (источников) мы получаем на выходе: 4 = vocals/drums/bass/other.
    # Должно совпадать с длиной списка self.sources (проверяется в validate()).
    num_sources: int = 4

    # Список названий дорожек. НЕ МЕНЯЙ порядок — он связан с обучением модели.
    # Если будешь менять — начни обучение с нуля, иначе дорожки перепутаются.
    #
    # ВАЖНО (проверено 26.09.2026, check_demucs.py): у самой модели htdemucs
    # порядок каналов ДРУГОЙ - drums, bass, other, vocals. Этот список - про
    # то, как мы называем файлы у себя, а настоящие названия всегда бери
    # у модели: model.sources. Иначе вокал сохранится под именем "drums".
    sources: List[str] = field(default_factory=lambda: list(SOURCES))

    # Путь к файлу весов (pretrained), с которого мы стартуем.
    # None = "взять из библиотеки demucs автоматически (скачается сама)".
    # Если у тебя уже есть свой .pth — впиши сюда полный путь, например:
    #   pretrained_model_path = r"C:\Users\User\models\htdemucs.th"
    pretrained_model_path: Optional[str] = None

    # Путь, куда сохраняем результат дообучения (наша финальная модель).
    output_model_path: str = "./models/custom_demucs.pth"

    # ------------------ АУДИО --------------------------------------------
    # Частота дискретизации (Гц). 44100 — стандарт CD-качества.
    # Менять нельзя "по желанию": модель обучалась на 44100, при другой
    # частоте Demucs не сможет корректно обработать звук.
    sample_rate: int = 44100

    # Размер одного обучающего кусочка аудио в СЕМПЛАХ (не в секундах!).
    # 480000 семплов = 480000 / 44100 = 10.88 секунды звука.
    # Demucs внутри использует блоки по этому размеру — так устроен STFT-стек.
    segment_length: int = 480000

    # Сколько миллисекунд оставлять "зазором" между соседними сегментами
    # при нарезке длинных песен (убирает щелчки на стыках).
    segment_padding: int = 0

    # ------------------ ОБУЧЕНИЕ -----------------------------------------
    # Размер батча — сколько кусочков за раз подаём в модель.
    # 8 — это 8 * 4 дорожки * 10.88 сек, примерно 348 секунд аудио за шаг.
    # На слабом CPU лучше поставить 2-4, на видеокарте можно 16-32.
    # Если не хватает памяти — уменьшай, ошибка будет вида "out of memory".
    batch_size: int = 8

    # Сколько полных проходов по всему датасету делаем за одно обучение.
    # 50 — небольшое число: этого хватает, чтобы "подружить" модель с нашими
    # стилями музыки, но НЕ достаточно, чтобы переобучить её.
    # Переобучение = модель запомнила тренировочные треки и ломается на новых.
    epochs: int = 50

    # Скорость обучения. Маленькое значение = делаем маленькие аккуратные шаги.
    # 0.0001 (1e-4) выбран потому, что мы дообучаем УЖЕ обученную модель:
    # большая скорость быстро испортила бы хорошие веса.
    # Для обучения с нуля (с random weights) обычно ставят 0.001.
    learning_rate: float = 0.0001

    # Весовой коэффициент затухания L2 (weight decay).
    # Небольшое значение "сжимает" веса и снижает переобучение.
    weight_decay: float = 0.0001

    # Количество потоков для чтения данных.
    # 4 — разумно для большинства ПК (8 ядер). Больше 8 обычно бесполезно.
    # Если система тормозит — поставь 2.
    num_workers: int = 4

    # Нужны ли "горячие" (перемешанные) данные каждый раз.
    shuffle: bool = True

    # Проверочная выборка для оценки качества: доля от всех данных.
    # 0.1 = 10% треков не учатся в обучении, а только проверяют модель.
    validation_split: float = 0.1

    # Сохранять промежуточные чекпойнты (чтобы не потерять обучение при сбое).
    save_checkpoints: bool = True

    # Как часто сохранять чекпойнт (в эпохах).
    checkpoint_interval: int = 5

    # Останавливать ли обучение, если качество перестало улучшаться.
    early_stopping: bool = True

    # Сколько эпох терпим без улучшения, прежде чем остановиться.
    early_stopping_patience: int = 10

    # Градиентный клиппинг — обрезает слишком большие градиенты,
    # защищает от "взрыва" обучения (NaN в весах).
    grad_clip_norm: float = 1.0

    # Фиксированное "зерно" генератора случайных чисел.
    # Нужно для воспроизводимости: с тем же seed результат обучения будет
    # таким же. 42 — классическое значение.
    seed: int = 42

    # ------------------ ДАННЫЕ -------------------------------------------
    # Папка с нашими легальными обучающими данными.
    # Ожидаемая структура внутри (создаст data_prep.py):
    #   custom_dataset/
    #     - mixture/    — полные миксы (то, что мы хотим разделить)
    #     - vocals/     — только вокал
    #     - drums/      — только барабаны
    #     - bass/       — только бас
    #     - other/      — остальное
    custom_data_path: str = "./custom_dataset"

    # Расширения файлов, которые считаем аудио.
    audio_extensions: List[str] = field(
        default_factory=lambda: [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
    )

    # Минимальная длительность трека в секундах. Короче — выбрасываем.
    min_track_duration: float = 10.0

    # Нормализация громкости (привести все треки к одной громкости),
    # чтобы модель не "училась" на тихих и на громких записях.
    normalize_audio: bool = True

    # Целевая громкость в dBFS при нормализации. -20 dBFS — безопасное значение.
    target_dbfs: float = -20.0

    # Делить ли данные на части (train/val) и сохранять списки в JSON.
    save_dataset_manifest: bool = True

    # ------------------ ВЫЧИСЛЕНИЯ ---------------------------------------
    # Устройство вычислений: "cpu" / "cuda" / "mps".
    # Ставим "cpu", потому что на первом этапе нет настроенной видеокарты.
    # Когда появится NVIDIA с CUDA — поменяй на "cuda" (одна строка).
    device: str = "cpu"

    # Использовать ли смешанную точность (float16) на видеокарте.
    # Ускоряет обучение почти вдвое, но на CPU работать не будет.
    use_amp: bool = False

    # Сколько памяти видеокарты можно занять (ГБ). Только для cuda.
    gpu_memory_fraction: float = 0.8

    # Заранее "прогреть" CUDA-ядра (быстрее первый шаг обучения).
    enable_cudnn_benchmark: bool = True

    # Фиксировать случайные числа для воспроизводимости на разных устройствах.
    deterministic: bool = False

    # ------------------ ВЫВОД (inference) --------------------------------
    # Папка, куда складываем разделённые дорожки новых песен.
    output_dir: str = "./output"

    # Список дорожек, которые сохранять по умолчанию (пусто = все дорожки).
    # Пример: ["vocals"] — только вокал.
    keep_sources: List[str] = field(default_factory=list)

    # Формат выходного аудио. "wav" — без потерь и большие, "mp3" — маленький.
    output_format: str = "wav"

    # Качество для сжатого формата (битрейт в кбит/с). 192 — хороший компромисс.
    output_bitrate: str = "192k"

    # Перекрытие соседних окон при разделении. 0.25 — стандарт Demucs.
    # Меньше — быстрее, но возможны артефакты на стыках.
    overlap: float = 0.25

    # Сколько раз повторять «сдвиг» по времени (0 = точное совпадение с началом).
    shifts: int = 0

    # Разделять ли стерео на 2 канала (True) или работать в моно (False).
    stereo: bool = True

    # Максимальный размер загружаемого файла (МБ) — защита от перегрузки.
    max_file_size_mb: int = 200

    # ------------------ СЕРВИС (API / бот) -------------------------------
    # Хост и порт FastAPI-сервера.
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # Токен Telegram-бота. НИКОГДА не пиши токен сюда и не коммить его в git.
    # Будет читаться из переменной окружения TELEGRAM_BOT_TOKEN.
    telegram_bot_token_env: str = "TELEGRAM_BOT_TOKEN"

    # Максимальный размер файла, который бот/API примет (МБ).
    max_upload_mb: int = 200

    # Таймаут обработки одного файла в секундах (защита от зависаний).
    processing_timeout: int = 1800

    # Сколько файлов бот обрабатывает одновременно.
    max_concurrent_jobs: int = 1

    # ------------------ ЛОГИ И МОНИТОРИНГ --------------------------------
    # Папка для логов.
    log_dir: str = "./logs"

    # Уровень логирования: DEBUG / INFO / WARNING / ERROR.
    log_level: str = "INFO"

    # Максимальный размер одного файла лога в МБ (потом ротируется).
    log_max_size_mb: int = 10

    # Сколько старых файлов логов хранить.
    log_backup_count: int = 5

    # Как часто проверять "живость" сервиса, секунды.
    monitor_interval: int = 60

    # ================================================================
    # СЛУЖЕБНЫЕ МЕТОДЫ (ниже — не гиперпараметры, а вспомогательные функции)
    # ================================================================

    @property
    def project_root(self) -> Path:
        """Абсолютный путь к корню проекта."""
        return PROJECT_ROOT

    @property
    def segment_seconds(self) -> float:
        """Длительность одного сегмента в секундах (для логов и README)."""
        return self.segment_length / self.sample_rate

    @property
    def custom_data_path_abs(self) -> Path:
        """custom_data_path, но всегда абсолютный (не зависит от места запуска)."""
        return self._abs(self.custom_data_path)

    @property
    def output_model_path_abs(self) -> Path:
        """output_model_path как абсолютный путь."""
        return self._abs(self.output_model_path)

    @property
    def output_dir_abs(self) -> Path:
        """output_dir как абсолютный путь."""
        return self._abs(self.output_dir)

    @property
    def log_dir_abs(self) -> Path:
        """log_dir как абсолютный путь."""
        return self._abs(self.log_dir)

    @staticmethod
    def _abs(path_like: str) -> Path:
        """
        Превращает "./models/x.pth" в абсолютный путь от корня проекта.
        Абсолютные пути (C:\\... или /home/...) возвращает без изменений.
        """
        p = Path(path_like)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Все поля конфига в виде обычного словаря (для логов и JSON)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def ensure_directories(self) -> List[str]:
        """
        Создаёт все нужные рабочие папки, если их ещё нет.
        Возвращает список созданных папок (удобно для логов).
        """
        created: List[str] = []
        for directory in (
            self.custom_data_path_abs,
            self.output_model_path_abs.parent,
            self.output_dir_abs,
            self.log_dir_abs,
        ):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(str(directory))
        return created

    def validate(self) -> List[str]:
        """
        Мягкая проверка конфига: возвращает список ПРОБЛЕМ (строки).
        Пустой список = всё в порядке.
        Исключения не бросает — вызывающий код сам решает, что делать.

        ЭТО НЕ ВАЛИДАЦИЯ ТИПА (для этого есть аннотации типов),
        а проверка ЗАЧЕМОВОСТИ значений: чтобы мы не обучали модель
        с sample_rate = -1 и не гадали потом, почему получился мусор.
        """
        problems: List[str] = []

        # 1. Модель должна быть из списка поддерживаемых.
        if self.demucs_model not in SUPPORTED_DEMUCS_MODELS:
            problems.append(
                f"demucs_model='{self.demucs_model}' нет такого в demucs. "
                f"Доступно: {', '.join(SUPPORTED_DEMUCS_MODELS)}"
            )

        # 2. Число дорожек должно совпадать с длиной списка sources.
        if self.num_sources != len(self.sources):
            problems.append(
                f"num_sources={self.num_sources}, а в sources {len(self.sources)} "
                f"элементов: {self.sources}. Они должны совпадать."
            )
        if len(set(self.sources)) != len(self.sources):
            problems.append(f"В sources есть повторы: {self.sources}")

        # 3. Частота дискретизации: 8000 <= sr <= 192000.
        if not (8000 <= self.sample_rate <= 192000):
            problems.append(
                f"sample_rate={self.sample_rate} вне диапазона 8000..192000"
            )

        # 4. Сегмент должен быть разумной длины (1..60 секунд).
        seg_sec = self.segment_seconds
        if not (1.0 <= seg_sec <= 60.0):
            problems.append(
                f"segment_length={self.segment_length} это {seg_sec:.2f} сек, "
                "ожидаем 1..60 сек"
            )

        # 5. Размер батча и число эпох — положительные целые.
        if self.batch_size < 1:
            problems.append(f"batch_size={self.batch_size} должен быть >= 1")
        if self.epochs < 1:
            problems.append(f"epochs={self.epochs} должен быть >= 1")

        # 6. Скорость обучения: строго больше нуля и не запредельная.
        if not (0.0 < self.learning_rate <= 1.0):
            problems.append(
                f"learning_rate={self.learning_rate} должен быть в (0, 1]"
            )

        # 7. Число воркеров не может быть отрицательным.
        if self.num_workers < 0:
            problems.append(f"num_workers={self.num_workers} не может быть < 0")

        # 8. Устройство вычислений из разрешённого списка.
        if self.device not in ALLOWED_DEVICES:
            problems.append(
                f"device='{self.device}' не поддерживается. "
                f"Варианты: {', '.join(ALLOWED_DEVICES)}"
            )

        # 9. Доля проверочной выборки строго между 0 и 1.
        if not (0.0 < self.validation_split < 1.0):
            problems.append(
                f"validation_split={self.validation_split} должна быть в (0, 1)"
            )

        # 10. Перекрытие при разделении: 0 < overlap < 1.
        if not (0.0 <= self.overlap < 1.0):
            problems.append(f"overlap={self.overlap} должен быть в [0, 1)")

        # 11. Если модель дообучаем (pretrained путь задан) — файл должен есть.
        if self.pretrained_model_path is not None:
            if not self._abs(self.pretrained_model_path).exists():
                problems.append(
                    f"pretrained_model_path не найден: {self.pretrained_model_path}"
                )

        # 12. Нормализация невозможна на CPU, если включён AMP.
        if self.use_amp and self.device == "cpu":
            problems.append("use_amp=True не работает на device='cpu'")

        return problems

    def describe(self) -> List[str]:
        """
        Готовит человекочитаемое описание конфига (по одной строке на поле).
        Используется для вывода в консоль и записи в лог при старте обучения.
        """
        # Русские подписи для читаемого вывода.
        labels: Dict[str, str] = {
            "demucs_model": "базовая модель Demucs",
            "num_sources": "количество дорожек",
            "sources": "названия дорожек",
            "pretrained_model_path": "путь к pretrained-весам",
            "output_model_path": "куда сохраняем модель",
            "sample_rate": "частота дискретизации (Гц)",
            "segment_length": "длина сегмента (семплов)",
            "segment_padding": "зазор между сегментами (мс)",
            "batch_size": "размер батча",
            "epochs": "количество эпох",
            "learning_rate": "скорость обучения",
            "weight_decay": "weight decay (L2)",
            "num_workers": "потоки загрузки данных",
            "shuffle": "перемешивать данные",
            "validation_split": "доля проверочной выборки",
            "save_checkpoints": "сохранять чекпойнты",
            "checkpoint_interval": "интервал чекпойнтов (эпох)",
            "early_stopping": "ранняя остановка",
            "early_stopping_patience": "терпение ранней остановки (эпох)",
            "grad_clip_norm": "клиппинг градиентов",
            "seed": "зерно случайности",
            "custom_data_path": "папка с данными",
            "audio_extensions": "расширения аудиофайлов",
            "min_track_duration": "минимальная длительность трека (сек)",
            "normalize_audio": "нормализовать громкость",
            "target_dbfs": "целевая громкость (dBFS)",
            "save_dataset_manifest": "сохранять манифест датасета",
            "device": "устройство вычислений",
            "use_amp": "смешанная точность (AMP)",
            "gpu_memory_fraction": "доля памяти GPU",
            "enable_cudnn_benchmark": "cudnn.benchmark",
            "deterministic": "детерминированный режим",
            "output_dir": "папка с результатами разделения",
            "keep_sources": "какие дорожки сохранять",
            "output_format": "формат выходного файла",
            "output_bitrate": "битрейт (для сжатых форматов)",
            "overlap": "перекрытие окон при разделении",
            "shifts": "случайные сдвиги (shifts)",
            "stereo": "стерео-режим",
            "max_file_size_mb": "макс. размер файла (МБ)",
            "api_host": "хост API",
            "api_port": "порт API",
            "telegram_bot_token_env": "переменная окружения с токеном бота",
            "max_upload_mb": "макс. размер загрузки (МБ)",
            "processing_timeout": "таймаут обработки (сек)",
            "max_concurrent_jobs": "одновременных задач",
            "log_dir": "папка с логами",
            "log_level": "уровень логирования",
            "log_max_size_mb": "макс. размер лог-файла (МБ)",
            "log_backup_count": "сколько логов хранить",
            "monitor_interval": "интервал мониторинга (сек)",
        }

        lines: List[str] = []
        for f in fields(self):
            # Служебные свойства и методы в этот список не попадают —
            # здесь только обычные поля (гиперпараметры).
            label = labels.get(f.name, f.name)
            value = getattr(self, f.name)
            if value is None:
                value_shown = "None  (будет скачано автоматически)"
            else:
                value_shown = repr(value)
            lines.append(f"  {f.name:<24} = {value_shown:<38} # {label}")
        return lines

    def summary(self) -> str:
        """Короткая сводка для подтверждения, что конфиг загрузился."""
        return (
            f"Модель: {self.demucs_model} | Дорожки: {', '.join(self.sources)} | "
            f"{self.sample_rate} Гц | сегмент {self.segment_length} "
            f"({self.segment_seconds:.2f} сек) | батч {self.batch_size} | "
            f"эпох {self.epochs} | lr {self.learning_rate} | "
            f"устройство {self.device}"
        )


# ---------------------------------------------------------------------------
# ЕДИНЫЙ ЭКЗЕМПЛЯР КОНФИГА
# ---------------------------------------------------------------------------

# Создаём ОДИН объект Config на всё приложение.
# Все остальные файлы делают: from config import config
# и получают одни и те же настройки.
config = Config()


def get_config() -> Config:
    """
    Возвращает глобальный конфиг.
    Используй её вместо прямого `from config import config`, если нужен
    единый стиль доступа в коде.
    """
    return config


def reload_config() -> Config:
    """
    Пересоздаёт конфиг с нуля (сбрасывает все изменения, сделанные в коде).
    Полезно в тестах: можно задать cfg.device = "cuda", а потом вернуть всё
    обратно одной командой, не перезапуская скрипт.
    """
    global config
    config = Config()
    return config


def resolve_device(device: Optional[str] = None) -> str:
    """
    Определяет реальное устройство вычислений.

    Логика:
      1. Если устройство явно передали — берём его.
      2. Иначе пробуем CUDA (видеокарта NVIDIA).
      3. Иначе пробуем MPS (чипы Apple).
      4. Иначе — CPU (работает всегда).

    torch импортируется ЗДЕСЬ, а не вверху файла — чтобы `python config.py`
    работал даже без установленного torch.
    """
    wanted = (device or config.device or "cpu").lower()

    try:
        import torch  # type: ignore
    except ImportError:
        # torch не установлен — считаем, что работаем на CPU.
        return "cpu"

    if wanted == "cuda" and torch.cuda.is_available():
        return "cuda"
    if wanted == "mps" and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"
    if wanted != "cpu" and wanted not in ALLOWED_DEVICES:
        return "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# ПРОВЕРКА ПРИ ЗАПУСКЕ ЧЕРЕЗ `python config.py`
# ---------------------------------------------------------------------------

def _setup_console() -> None:
    """
    Переводит вывод консоли в UTF-8 с заменой непечатных символов.

    Зачем: в Windows консоль по умолчанию использует кодировку cp866/cp1251,
    где нет всех букв. Без этой настройки печать русских букв/эмодзи упала бы
    с ошибкой UnicodeEncodeError. С "errors=replace" она не упадёт НИКОГДА —
    непонятный символ просто заменится на "?".
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Уже настроен или поток закрыт — не критично, продолжаем.
                pass


def _print_report() -> None:
    """Печатает полный отчёт о конфигурации. Используется только при запуске."""
    _setup_console()

    line = "=" * 78
    print(line)
    print(" demucs_custom — КОНФИГУРАЦИЯ ПРОЕКТА")
    print(line)
    print(f" Корень проекта : {PROJECT_ROOT}")
    print(f" Версия Python : {sys.version.split()[0]}")
    print("-" * 78)
    print(" ВСЕ ПАРАМЕТРЫ:")
    print("-" * 78)
    for row in config.describe():
        print(row)

    print("-" * 78)
    print(" ПРОВЕРКА:")
    print("-" * 78)

    # Реальное устройство вычислений (если torch уже установлен).
    actual_device = resolve_device()
    print(f"  Устройство по конфигу : {config.device}")
    print(f"  Устройство реально     : {actual_device}")
    if config.device == "cpu" and actual_device == "cuda":
        print("  [!] Доступна видеокарта! Можно поставить device='cuda' в config.py")

    # Проверка значений: предупреждения не мешают, но их надо видеть.
    problems = config.validate()
    if problems:
        print(f"  Найдено проблем: {len(problems)}")
        for p in problems:
            print(f"    [!] {p}")
    else:
        print("  Все параметры корректны. Проблем не найдено.")

    print("-" * 78)
    print(f" ИТОГ: {config.summary()}")
    print(line)
    print(" Готово. Можно запускать остальные скрипты проекта.")
    print(line)


if __name__ == "__main__":
    _print_report()
