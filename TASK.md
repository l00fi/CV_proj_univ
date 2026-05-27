# Задание: детекция игральных карт (YOLOv8)

## Постановка задачи

Разработать end-to-end пайплайн для **обнаружения и классификации игральных карт** на фотографиях покерных комбинаций. Датасет подготовлен на платформе [Roboflow](https://roboflow.com) и экспортирован в формате **YOLO**.

### Данные

| Разбиение | Изображений | Описание |
|-----------|-------------|----------|
| `train`   | 109         | Обучающая выборка |
| `test`    | 31          | Тестовая выборка (используется для валидации и инференса) |

- **Формат разметки:** YOLO (нормализованные `class_id cx cy w h`)
- **Количество классов:** 52 (все карты стандартной колоды: ранг × масть)
- **Имена классов:** `10C` … `AS` (ранг + масть: C=clubs, D=diamonds, H=hearts, S=spades)

### Цель модели

На вход — изображение стола/руки с картами. На выход — bounding boxes с метками конкретных карт (например, `AH`, `KS`).

---

## Модель и стек

| Компонент | Технология |
|-----------|------------|
| Архитектура | **YOLOv8** (Ultralytics, PyTorch backend) |
| Обучение | Ultralytics API (`model.train`) |
| Аугментации | YOLO built-in (mosaic, mixup, …) + Albumentations on-the-fly |
| Трекинг экспериментов | MLflow |
| Мониторинг | CPU/RAM/GPU sampling, Prometheus Pushgateway, Grafana |
| Отчёты | JSON / Markdown / Prometheus → `runs/reports/` |
| Окружение | Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) |
| Деплой | Docker Compose (пайплайн + MLflow + observability) |

Базовая модель: `yolov8n.pt` (nano). При наличии GPU — `yolov8s/m/l`.

---

## Реализованный пайплайн

Одна команда **`train`** выполняет полный цикл:

```
train → validate → infer → report
```

| Шаг | Модуль | Что делает |
|-----|--------|------------|
| 1. Train | `poker_yolo/train.py` | Обучение YOLOv8, MLflow callbacks, мониторинг ресурсов |
| 2. Validate | `poker_yolo/validate.py` | mAP, Precision, Recall, F1 на test split |
| 3. Infer | `poker_yolo/infer.py` | Предсказания на `infer.source` (по умолчанию `dataset/test/images`) |
| 4. Report | `poker_yolo/reporting.py` | JSON/MD/Prometheus, 3 preview-изображения, production KPI |

Отдельные команды `validate` и `infer` доступны для повторного запуска на уже обученной модели.

---

## Метрики

### Основные (object detection)

| Метрика | Описание | Целевое значение (baseline) |
|---------|----------|----------------------------|
| **mAP@0.5** | Mean Average Precision при IoU=0.5 | ≥ 0.85 |
| **mAP@0.5:0.95** | COCO-style mAP (IoU 0.5–0.95) | ≥ 0.60 |
| **Precision** | TP / (TP + FP) | ≥ 0.90 |
| **Recall** | TP / (TP + FN) | ≥ 0.85 |
| **F1** | Гармоническое среднее Precision и Recall | ≥ 0.87 |

### Дополнительные (реализованы в отчётах)

| Метрика | Где смотреть |
|---------|--------------|
| **mAP per class** | MLflow artifacts |
| **Inference latency / FPS** | `runs/reports/latest.md`, MLflow |
| **Model size (MB)** | Отчёт, Grafana |
| **CPU/RAM/GPU (train/val)** | Отчёт, Grafana |
| **Augmentation ratio** | Отчёт (synthetic/real), Grafana |
| **Pipeline duration** | Отчёт (train + val + infer) |

### Критерии приёмки

1. mAP@0.5 на test ≥ 0.80 — минимально приемлемый результат
2. mAP@0.5 ≥ 0.90 — хороший результат для данного объёма данных
3. Нет классов с mAP < 0.50 (иначе — доразметка / аугментации)

---

## Способы улучшения качества

### 1. Данные

- **Увеличить датасет:** 140 изображений мало для 52 классов; целевой объём — 500+
- **Балансировка классов:** проверить распределение карт, добавить редкие классы
- **Hard negative mining:** кадры без карт, перекрытия, блики
- **Проверка разметки:** ручной аудит bbox на train/test

### 2. Аугментации

Настраиваются в `configs/*.yaml`, секция `augmentations`:

- YOLO: mosaic, mixup, copy-paste, cutmix, HSV, flip, rotate, scale
- Albumentations: blur, noise, brightness/contrast, compression и др.

### 3. Архитектура и обучение

- Переход `yolov8n → yolov8s/m` при наличии GPU
- Увеличить `epochs` (100–300), early stopping по val mAP (`patience`)
- Подбор `imgsz` (640 → 1280 для мелких карт)
- Тюнинг `lr0`, `lrf`, `warmup_epochs`, `weight_decay`

### 4. Постобработка

- Подбор `conf` / `iou` в секциях `validate` и `infer` конфига
- Test-Time Augmentation (TTA) для финальной оценки
- Ensemble нескольких чекпоинтов

### 5. Инфраструктура экспериментов

- MLflow: сравнение run'ов по mAP и гиперпараметрам
- Grafana: дашборд с метриками, ресурсами и preview-изображениями
- Структурированные логи: `runs/logs/poker-yolo.jsonl`

---

## Структура проекта

```
CV_proj/
├── poker_yolo/              # Python-пакет пайплайна
│   ├── cli.py               # train / validate / infer
│   ├── train.py             # обучение YOLOv8
│   ├── validate.py          # валидация на test
│   ├── infer.py             # инференс
│   ├── augmentations.py     # YOLO + Albumentations
│   ├── monitoring.py        # CPU/GPU/RAM, aug stats
│   ├── predictions.py       # 3 preview-изображения
│   ├── reporting.py         # JSON / MD / Prometheus
│   ├── callbacks.py         # MLflow epoch callbacks
│   ├── mlflow_utils.py      # MLflow integration
│   ├── logging_config.py    # JSONL structured logs
│   └── config.py            # YAML config loader
├── configs/
│   ├── default.yaml         # 50 эпох, полные аугментации
│   ├── local.yaml           # 10 эпох, MLflow localhost
│   └── smoke.yaml           # 3 эпохи, CPU, smoke test
├── dataset/                 # Roboflow export (train/test)
├── observability/           # Prometheus, Grafana, nginx
├── scripts/                 # entrypoint.sh
├── tests/                   # pytest (61 тест)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── README.md                # руководство пользователя
└── TASK.md                  # этот документ
```

---

## Конфигурации

| Файл | Эпохи | Device | MLflow URI | Назначение |
|------|-------|--------|------------|------------|
| `configs/smoke.yaml` | 3 | cpu | localhost:5000 | Быстрая проверка пайплайна |
| `configs/local.yaml` | 10 | auto | localhost:5000 | Локальная разработка |
| `configs/default.yaml` | 50 | auto | mlflow:5000 (Docker) | Полное обучение |

Ключевые секции YAML: `data`, `model`, `train`, `augmentations`, `validate`, `infer`, `mlflow`, `reporting`.

---

## Запуск

### Локально

```bash
uv sync
docker compose up -d mlflow
docker compose --profile observability up -d   # опционально

# Полный пайплайн
uv run poker-yolo --config configs/local.yaml train

# Smoke test (~5 мин на CPU)
uv run poker-yolo --config configs/smoke.yaml train

# Полное обучение (50 эпох)
uv run poker-yolo --config configs/default.yaml train
```

### Опции команды `train`

```bash
--skip-infer              # только train + validate
--infer-source PATH       # другая папка для инференса
--no-save                 # не сохранять аннотированные картинки
```

### Отдельные шаги

```bash
uv run poker-yolo --config configs/local.yaml validate \
  --weights runs/detect/runs/train/poker_cards/weights/best.pt

uv run poker-yolo --config configs/local.yaml infer \
  --weights runs/detect/runs/train/poker_cards/weights/best.pt \
  --source dataset/test/images
```

### Docker

```bash
docker compose up -d mlflow
docker compose --profile observability up -d
docker compose build poker-yolo
docker compose run --rm poker-yolo train --config configs/default.yaml
```

---

## Артефакты и просмотр результатов

| Артефакт | Путь |
|----------|------|
| Веса | `runs/detect/runs/train/<name>/weights/best.pt` |
| Отчёт | `runs/reports/latest.md` |
| Preview | `runs/reports/preview/sample_{0,1,2}.jpg` |
| Инференс | `runs/infer/pred_<timestamp>/` |
| Логи | `runs/logs/poker-yolo.jsonl` |

| Сервис | URL |
|--------|-----|
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3001 (admin / admin) |
| Preview nginx | http://localhost:8088/preview/ |
| Prometheus | http://localhost:9090 |

---

## Переменные окружения

| Переменная | Описание |
|------------|----------|
| `MLFLOW_TRACKING_URI` | URI MLflow (перекрывает YAML) |
| `PROMETHEUS_PUSHGATEWAY_URL` | Pushgateway для Grafana |
| `REPORTS_BASE_URL` | Base URL для preview-ссылок в отчётах |

---

## Тесты

```bash
uv sync --group dev
uv run pytest
```

Подробное руководство — в [README.md](README.md).

### Определение комбинации на фото

```bash
uv run poker-hand --config configs/local.yaml
uv run poker-hand --image dataset/test/images/test_1.jpg --weights runs/detect/runs/train/poker_cards/weights/best.pt
```

Скрипт берёт случайное test-изображение (или `--image`), детектирует 5 карт и выводит название комбинации либо сообщает, что комбинацию определить нельзя.

