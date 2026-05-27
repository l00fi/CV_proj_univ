# Задание: детекция игральных карт (YOLOv8)

## Постановка задачи

Разработать end-to-end пайплайн для **обнаружения и классификации игральных карт** на фотографиях покерных комбинаций. Датасет подготовлен на платформе [Roboflow](https://roboflow.com) и экспортирован в формате **YOLO**.

### Данные

| Разбиение | Изображений | Описание |
|-----------|-------------|----------|
| `train`   | 109         | Обучающая выборка |
| `test`    | 31          | Тестовая выборка (используется также для валидации) |

- **Формат разметки:** YOLO (нормализованные `class_id cx cy w h`)
- **Количество классов:** 52 (все карты стандартной колоды: ранг × масть)
- **Имена классов:** `2C` … `AS` (ранг + масть: C=clubs, D=diamonds, H=hearts, S=spades)

### Цель модели

На вход — изображение стола/руки с картами. На выход — bounding boxes с метками конкретных карт (например, `AH`, `KS`).

---

## Модель и стек

| Компонент | Технология |
|-----------|------------|
| Архитектура | **YOLOv8** (Ultralytics, PyTorch backend) |
| Фреймворк обучения | PyTorch + PyTorch Lightning (оркестрация) |
| Трекинг экспериментов | MLflow |
| Окружение | Python 3.11, `uv` |
| Деплой | Docker (один контейнер: train / validate / infer) |

Базовая модель по умолчанию: `yolov8n.pt` (nano — быстрый baseline). При наличии GPU можно перейти на `yolov8s/m/l`.

---

## Метрики

### Основные (object detection)

| Метрика | Описание | Целевое значение (baseline) |
|---------|----------|----------------------------|
| **mAP@0.5** | Mean Average Precision при IoU=0.5 | ≥ 0.85 |
| **mAP@0.5:0.95** | COCO-style mAP (IoU 0.5–0.95, шаг 0.05) | ≥ 0.60 |
| **Precision** | TP / (TP + FP) | ≥ 0.90 |
| **Recall** | TP / (TP + FN) | ≥ 0.85 |
| **F1** | Гармоническое среднее Precision и Recall | ≥ 0.87 |

### Дополнительные

| Метрика | Описание |
|---------|----------|
| **mAP per class** | mAP по каждой из 52 карт — выявление редких/сложных классов |
| **Inference latency** | Время инференса на одно изображение (ms) |
| **FPS** | Кадров в секунду при batch=1 |
| **Model size** | Размер весов (.pt), MB |

### Критерии приёмки

1. mAP@0.5 на test ≥ 0.80 (минимально приемлемый результат)
2. mAP@0.5 ≥ 0.90 — хороший результат для данного объёма данных
3. Нет классов с mAP < 0.50 (иначе — доразметка / аугментации)

---

## Способы улучшения качества

### 1. Данные

- **Увеличить датасет:** 140 изображений мало для 52 классов; целевой объём — 500+ изображений
- **Балансировка классов:** проверить распределение карт, добавить примеры редких классов
- **Hard negative mining:** добавить кадры без карт, частично перекрытых карт, бликов
- **Проверка разметки:** ручной аудит bbox на train/test

### 2. Аугментации (Roboflow / Albumentations)

- Mosaic, MixUp, Copy-Paste (стандарт YOLOv8)
- HSV jitter, flip, rotate (±15°), scale (0.5–1.5)
- Blur, noise, JPEG compression — имитация камеры телефона
- **Cutout** на областях стола — устойчивость к перекрытиям

### 3. Архитектура и обучение

- Переход `yolov8n → yolov8s/m` при наличии GPU
- Увеличить `epochs` (100–300), early stopping по val mAP
- Подбор `imgsz` (640 → 1280 для мелких карт)
- Тюнинг `lr0`, `lrf`, `warmup_epochs`, `weight_decay`
- **Transfer learning:** заморозка backbone на первых N эпохах
- **Fine-tune** с меньшим lr после основного обучения

### 4. Постобработка

- Подбор `conf` threshold (0.25 → 0.40) и `iou` NMS (0.45 → 0.60)
- Test-Time Augmentation (TTA) для финальной оценки
- Ensemble нескольких чекпоинтов

### 5. Инфраструктура экспериментов

- MLflow: сравнение run'ов по mAP, гиперпараметрам, версиям данных
- Логирование confusion matrix и примеров предсказаний как artifacts
- Версионирование датасета через DVC или MLflow datasets

---

## Этапы реализации

```
[1] Подготовка data.yaml и проверка датасета
         ↓
[2] Baseline-обучение YOLOv8n (50 epochs)
         ↓
[3] Валидация на test, логирование метрик в MLflow
         ↓
[4] Инференс на новых изображениях
         ↓
[5] Итерации: аугментации, гиперпараметры, большая модель
```

---

## Структура проекта

```
CV_proj/
├── dataset/              # Roboflow export (train/test)
├── configs/
│   └── default.yaml      # Гиперпараметры обучения
├── src/poker_yolo/       # Исходный код
├── TASK.md               # Этот документ
├── pyproject.toml        # Зависимости (uv)
├── Dockerfile            # Train / validate / infer
└── docker-compose.yml    # App + MLflow server
```

## Запуск

```bash
# Локально (uv)
uv sync
docker compose up -d mlflow

# Полный пайплайн: train → validate → infer → report
uv run poker-yolo --config configs/local.yaml train

# Smoke test (3 эпохи)
uv run poker-yolo --config configs/smoke.yaml train

# Отдельные шаги (если модель уже обучена)
uv run poker-yolo --config configs/local.yaml validate --weights runs/detect/runs/train/poker_cards/weights/best.pt
uv run poker-yolo --config configs/local.yaml infer --weights runs/detect/runs/train/poker_cards/weights/best.pt --source dataset/test/images

# Docker
docker compose run --rm poker-yolo train --config configs/default.yaml
```

MLflow UI: http://localhost:5000
