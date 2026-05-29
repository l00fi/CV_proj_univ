# Справочник метрик Poker YOLO

Полный перечень метрик и полей, которые появляются в **MLflow**, **Grafana** и **HTML-отчёте**. Источник правды для числовых значений после пайплайна — `runs/reports/latest.json`; для Pushgateway — `runs/reports/latest.prom`; для Infinity-панелей — `runs/reports/grafana/*.json`.

См. также: [README.md](../README.md) (запуск), [TASK.md](TASK.md) (постановка задачи).

---

## Содержание

1. [Общие соглашения](#общие-соглашения)
2. [MLflow](#mlflow)
3. [Grafana](#grafana)
4. [HTML-отчёт](#html-отчёт)
5. [Сводная таблица имён](#сводная-таблица-имён)

---

## Общие соглашения

| Контекст | Формат имени | Пример |
|----------|--------------|--------|
| JSON / HTML / Markdown | `snake_case`, префиксы `train_`, `val_`, `infer_`, `hands_benchmark_` | `val_top1` |
| MLflow (итог фазы) | как в JSON или с префиксом фазы в `log_ultralytics_results` | `train_metrics_accuracy_top1` |
| MLflow (по эпохам) | `epoch_<sanitized_trainer_key>` | `epoch_metrics_accuracy_top1` |
| Prometheus / Grafana (instant) | `poker_yolo_<sanitized_json_key>` | `poker_yolo_val_top1` |

**Санитизация для Prometheus:** символы `-./()` и пробелы → `_`, нижний регистр; ведущая цифра → префикс `m_`.

**Labels на всех `poker_yolo_*` gauge (кроме специальных серий):** `phase`, `run_id`, `status`, иногда `config`.

**10 классов комбинаций** (порядок в confusion / outcomes / `COMBO_CLASSES`):

`high_card`, `pair`, `two_pair`, `three_of_a_kind`, `straight`, `flush`, `full_house`, `four_of_a_kind`, `straight_flush`, `royal_flush`.

**Важно:** `val_*` считаются на **Kaggle MRPH classify val** (одна карта → класс комбинации). `hands_benchmark_*` — на **`dataset/test/images`** (полный кадр стола, GT из YOLO-меток карт через `evaluate_hand`). Это разные задачи; значения не обязаны совпадать.

---

## MLflow

Каждая фаза (`train`, `validate`, `infer`) открывает **отдельный MLflow run** (`mlflow_phase`). Тег: `project=poker-yolo`.

### Параметры (`log_params`)

Записываются при старте фазы train через `log_config()`; дополнительно в validate/infer:

| Параметр | Фаза | Описание |
|----------|------|----------|
| `model_weights` | train | Стартовые веса, напр. `yolov8n-cls.pt` |
| `imgsz` | train | Размер входа |
| `epochs` | train | Число эпох |
| `batch` | train | Размер батча |
| `patience` | train | Early stopping patience |
| `lr0` | train | Начальный learning rate |
| `lrf` | train | Финальный LR factor |
| `weight_decay` | train | L2 regularization |
| `warmup_epochs` | train | Warmup epochs |
| `data_yaml` | train | Путь к YAML датасета |
| `aug_enabled` | train | Флаг аугментаций в YAML |
| `mosaic`, `mixup`, `copy_paste`, `cutmix`, `fliplr`, `flipud`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `hsv_h`, `hsv_s`, `hsv_v` | train | Параметры YOLO-ауг из конфига |
| `aug_alb_<name>` | train | Вероятность Albumentations-трансформа `<name>` |
| `weights` | validate, infer | Путь к используемому `.pt` |
| `source` | infer | Путь/метка источника изображений |

### Метрики по эпохам (только `train`)

Колбэки `on_train_epoch_end` / `on_fit_epoch_end` → `mlflow.log_metrics(..., step=epoch)`.

| Шаблон имени | Описание |
|--------------|----------|
| `epoch_<key>` | Каждая числовая метрика из `trainer.metrics` Ultralytics на конце эпохи. `<key>` — имя после санитизации: `/` → `_`, скобки и пробелы → `_`. |

**Типичные ключи Ultralytics classify** (точный набор зависит от версии):

| Исходный ключ trainer | Пример MLflow имени | Смысл |
|----------------------|---------------------|--------|
| `metrics/accuracy_top1` | `epoch_metrics_accuracy_top1` | Top-1 accuracy за эпоху |
| `metrics/accuracy_top5` | `epoch_metrics_accuracy_top5` | Top-5 accuracy за эпоху |
| `train/loss` | `epoch_train_loss` | Train loss |
| `val/loss` | `epoch_val_loss` | Val loss |
| `lr/pg0`, `lr/pg1`, … | `epoch_lr_pg0`, … | Learning rate param groups |

Полный список смотрите в UI MLflow → Metrics с фильтром `epoch_`.

### Итоговые метрики фазы `train`

| MLflow метрика | JSON-аналог | Описание |
|----------------|-------------|----------|
| `train_metrics_accuracy_top1` | `train_top1` | Top-1 accuracy по итогам train (из `results_dict`) |
| `train_metrics_accuracy_top5` | `train_top5` | Top-5 accuracy |
| `train_val_loss` или `train_train_loss` | `train_loss` | Loss (зависит от ключей Ultralytics) |
| `train_fitness` | `train_fitness` | Fitness Ultralytics |
| `train_duration_sec` | `train_duration_sec` | Длительность train, с |
| `train_cpu_avg_pct`, … | `train_*` ресурсы | См. [ресурсы](#ресурсы-resourcemonitor) с префиксом `train_` |

Дополнительно логируются **все** пары из `results.results_dict` с префиксом `train_` и заменой `/` на `_` (`metrics_for_mlflow`).

### Итоговые метрики фазы `validate`

| MLflow метрика | JSON-аналог | Описание |
|----------------|-------------|----------|
| `val_metrics_accuracy_top1` | `val_top1` | Top-1 на val split MRPH |
| `val_metrics_accuracy_top5` | `val_top5` | Top-5 на val |
| `val_val_loss` | `val_loss` | Validation loss |
| `val_fitness` | `val_fitness` | Fitness |
| `top1`, `top5`, `loss`, `fitness` | — | Дубли без префикса через `log_metrics(metrics)` |
| `val_duration_sec` | `val_duration_sec` | Длительность validate, с |
| `val_*` ресурсы | `val_cpu_avg_pct`, … | Мониторинг фазы val |

Артефакт: `validation_summary.json` (split, weights, метрики).

### Итоговые метрики фазы `infer`

| MLflow метрика | JSON-аналог | Описание |
|----------------|-------------|----------|
| `infer_latency_ms` | `infer_latency_ms` | Среднее время на кадр, мс |
| `infer_fps` | `infer_fps` | Кадров в секунду |
| `infer_images` | `infer_images` | Число кадров |
| `infer_top1_conf_avg` | `infer_top1_conf_avg` | Средняя уверенность top-1 |
| `infer_top1_class_<id>` | `infer_top1_class_<id>` | Счётчик top-1 по class id |

Артефакты: каталог `predictions/` (аннотированные кадры), параметры `weights`, `source`.

### Артефакты MLflow

| Путь в UI | Когда |
|-----------|--------|
| `weights/best.pt` | После train |
| `training/results.csv` | После train, если есть CSV |
| `predictions/` | После infer |
| `validation_summary.json` | После validate |

---

## Grafana

Три дашборда + JSON на report-server. Метрики делятся на **Prometheus (Pushgateway)** и **Infinity JSON**.

### Источники данных

| Источник | Файлы / API | Дашборды |
|----------|-------------|----------|
| Prometheus instant query | `latest.prom` → Pushgateway | Training, Inference (KPI) |
| Infinity → report-server | `runs/reports/grafana/*.json`, `${report_server}/preview/` | Curves, Inference (графики/таблицы) |

Запросы Prometheus: `max by () (last_over_time(poker_yolo_<metric>{status="success"}[7d]))` — последний успешный прогон за 7 дней.

---

### Дашборд «Training curves» (`poker-yolo-curves`)

**Файл:** `training_curves.json`  
**URL Infinity:** `http://report-server:8088/grafana/training_curves.json`

#### Панель: Loss (train vs validation)

| Поле JSON | Тип | Описание |
|-----------|-----|----------|
| `epoch` | number | Номер эпохи (ось X в UI) |
| `time` | timestamp (RFC3339) | Синтетическое время для timeseries-панели Grafana |
| `train_loss` | number | `train/loss` из `results.csv` |
| `val_loss` | number | `val/loss` из `results.csv` |

#### Панель: Accuracy (MRPH val split)

| Поле JSON | Тип | Описание |
|-----------|-----|----------|
| `epoch` | number | Номер эпохи |
| `time` | timestamp | Синтетическое время |
| `top1` | number | `metrics/accuracy_top1` (0…1) |
| `top5` | number | `metrics/accuracy_top5` (0…1) |

---

### Дашборд «Training metrics» (`poker-yolo-training`)

#### Prometheus-панели

| Метрика Prometheus | JSON-ключ | Описание |
|--------------------|-----------|----------|
| `poker_yolo_train_cpu_avg_pct` | `train_cpu_avg_pct` | Средняя загрузка CPU во время train, % |
| `poker_yolo_val_cpu_avg_pct` | `val_cpu_avg_pct` | Средняя загрузка CPU во время validate, % |
| `poker_yolo_train_ram_avg_mb` | `train_ram_avg_mb` | Средняя RAM при train, МБ |
| `poker_yolo_val_ram_avg_mb` | `val_ram_avg_mb` | Средняя RAM при validate, МБ |
| `poker_yolo_train_duration_sec` | `train_duration_sec` | Длительность train, с |
| `poker_yolo_pipeline_duration_sec` | `pipeline_duration_sec` | train + val + infer, с |
| `poker_yolo_val_top1` | `val_top1` | Top-1 accuracy MRPH val (0…1; gauge может показывать как долю) |
| `poker_yolo_val_top5` | `val_top5` | Top-5 accuracy MRPH val |

**Панели «MRPH validation top-1» и «MRPH val metrics»** дублируют `poker_yolo_val_top1` / `poker_yolo_val_top5` в разных визуализациях (gauge / stat).

#### Метрики в Pushgateway, но без отдельной панели на training-дашборде

Экспортируются из `prepare_export_metrics()` и доступны в Prometheus при необходимости кастомных панелей:

| Метрика Prometheus | Описание |
|--------------------|----------|
| `poker_yolo_train_cpu_peak_pct` | Пик CPU train |
| `poker_yolo_train_ram_peak_mb` | Пик RAM train |
| `poker_yolo_train_gpu_util_avg_pct` | Средняя загрузка GPU, % |
| `poker_yolo_train_gpu_util_peak_pct` | Пик GPU, % |
| `poker_yolo_train_gpu_mem_peak_mb` | Пик VRAM, МБ |
| `poker_yolo_train_resource_samples` | Число сэмплов монитора train |
| `poker_yolo_val_*` | Аналоги peak/gpu/samples для val |
| `poker_yolo_infer_*` | Все infer-метрики из JSON |
| `poker_yolo_hands_benchmark_*` | Все benchmark-метрики |
| `poker_yolo_model_size_mb` | Размер `best.pt` |
| `poker_yolo_run_duration_seconds` | Длительность всего run отчёта |
| `poker_yolo_run_info` | Константа `1` (маркер последнего run) |

#### Серии по эпохам (Prometheus, опционально в `.prom`)

| Метрика | Labels | Описание |
|---------|--------|----------|
| `poker_yolo_train_epoch_metric` | `epoch`, `series` | `series` ∈ `train_loss`, `val_loss`, `top1`, `top5` |

---

### Дашборд «Benchmark & Inference» (`poker-yolo-inference`)

#### Prometheus stat-панели

| Метрика Prometheus | JSON-ключ | Панель | Описание |
|--------------------|-----------|--------|----------|
| `poker_yolo_hands_benchmark_top1_conf_avg` | `hands_benchmark_top1_conf_avg` | Avg prediction confidence | Средняя уверенность top-1 на benchmark |
| `poker_yolo_hands_benchmark_accuracy` | `hands_benchmark_accuracy` | Benchmark accuracy | Доля верных комбинаций среди оценённых кадров |
| `poker_yolo_hands_benchmark_correct_total` | `hands_benchmark_correct_total` | Benchmark correct / incorrect | Число верных top-1 |
| `poker_yolo_hands_benchmark_incorrect_total` | `hands_benchmark_incorrect_total` | то же | Число ошибок top-1 |
| `poker_yolo_hands_benchmark_evaluated_images` | `hands_benchmark_evaluated_images` | то же | Кадры с GT и предсказанием |

#### Infinity: `benchmark_outcomes.json`

**Панель:** Predictions by true hand (correct vs incorrect) — stacked bar.

| Поле | Описание |
|------|----------|
| `hand_class` | Истинная комбинация (ось X) |
| `hand_index` | Индекс 0…9 в `COMBO_CLASSES` |
| `correct` | Число верных top-1 для этой руки |
| `incorrect` | Число неверных top-1 |

#### Infinity: `confusion_matrix.json`

**Панель:** Confusion matrix (true hand → predicted) — цветная таблица.

| Поле | Описание |
|------|----------|
| `true_class` | Истинная комбинация (строка) |
| `true_index` | Индекс истинного класса |
| `pred_class` | Предсказанная комбинация (столбец) |
| `pred_index` | Индекс предсказанного класса |
| `count` | Число кадров с парой (true, pred) |

#### Превью (панели Sample 0…N)

Не метрики JSON — URL `${report_server}/preview/sample_<i>.jpg`. Соответствуют `predictions[]` в отчёте.

#### Prometheus-серии benchmark (в `.prom`, без отдельной панели)

| Метрика | Labels | Описание |
|---------|--------|----------|
| `poker_yolo_hands_benchmark_outcome` | `hand_class`, `outcome`, `hand_index` | `outcome` = `correct` \| `incorrect` |
| `poker_yolo_hands_confusion` | `true_class`, `pred_class`, `true_index`, `pred_index` | Счётчик ячейки матрицы |
| `poker_yolo_preview_sample_predictions` | `sample_index` | `predictions_count` на превью |
| `poker_yolo_preview_sample_top_confidence` | `sample_index`, `class_name` | Уверенность top-1 на превью |

---

## HTML-отчёт

Файл: `runs/reports/latest.html` (модуль `poker_yolo/html_report.py`). Данные — тот же `latest.json`, что и для MD/Prom/Grafana.

### Шапка и обзор (KPI)

| Поле | Описание |
|------|----------|
| `run_id`, `phase`, `config_name`, `status`, `error` | Метаданные прогона |
| `started_at`, `finished_at`, `duration_sec` | Время и длительность |
| KPI-карточки | Приоритет: `hands_benchmark_accuracy`, `val_top1`, `val_top5`, `train_duration_sec`, `pipeline_duration_sec`, остальные `hands_benchmark_*` |

### Датасет (`dataset_stats`)

| Поле | Описание |
|------|----------|
| `train_images` | Файлов в train (Kaggle) |
| `val_images` | Файлов в val |
| `test_images` | Файлов в test split Kaggle |
| `num_classes` | Число классов в `data.yaml` |
| `class_names` | Имена классов (в JSON; в HTML — косвенно через число классов) |
| `benchmark_test_images` | Число изображений hands benchmark |
| `benchmark_source` | Путь-метка, напр. `dataset/test/images` |

### Параметры и аугментации

| Блок | Поля | Описание |
|------|------|----------|
| `params` | `model_weights`, `task`, `imgsz`, `epochs`, `batch`, `device`, `data_yaml`, `aug_enabled`, `mosaic`, `mixup`, `infer_source` | Снимок конфига на старт |
| `augmentations_summary` | `train_images_real`, `augmentations_enabled`, `note`, `synthetic_to_real_ratio`, `estimated_augmented_views_per_epoch`, `yolo_probabilities`, `albumentations_probabilities` | См. [MLflow параметры](#параметры-log_params); для classify `note` про встроенные ауги Ultralytics |

### Кривые обучения (Chart.js)

Те же поля, что в [training_curves.json](#дашборд-training-curves-poker-yolo-curves): `epoch`, `train_loss`, `val_loss`, `top1`, `top5`.

| Метрика | Описание |
|---------|----------|
| `train_loss` | Средний/агрегированный loss на train по эпохе |
| `val_loss` | Loss на validation по эпохе |
| `top1` | Val top-1 accuracy по эпохе |
| `top5` | Val top-5 accuracy по эпохе |

### Метрики обучения (`train_*`)

| Метрика | Описание |
|---------|----------|
| `train_top1` | Итоговая top-1 accuracy после train |
| `train_top5` | Итоговая top-5 accuracy |
| `train_fitness` | Fitness Ultralytics |
| `train_loss` | Итоговый loss (если экспортирован) |
| `train_duration_sec` | Длительность train |

### Метрики валидации (`val_*`)

| Метрика | Описание |
|---------|----------|
| `val_top1` | Top-1 на MRPH val — **главная offline-метрика классификатора** |
| `val_top5` | Top-5 на val |
| `val_fitness` | Fitness после val |
| `val_loss` | Validation loss |
| `val_duration_sec` | Длительность validate |

### Метрики инференса (`infer_*`)

| Метрика | Формула / описание |
|---------|-------------------|
| `infer_images` | N — число кадров |
| `infer_duration_sec` | T — wall-clock predict |
| `infer_latency_ms` | 1000 × T / N |
| `infer_fps` | N / T |
| `infer_top1_conf_avg` | Среднее `probs.top1conf` |
| `infer_top1_class_<id>` | Сколько раз top-1 = класс id (0…9) |

Без сравнения с GT; для качества на столе см. `hands_benchmark_*`.

### Ресурсы (`<phase>_…`)

| Метрика | Описание |
|---------|----------|
| `<phase>_cpu_avg_pct` | Средний CPU, % |
| `<phase>_cpu_peak_pct` | Пик CPU, % |
| `<phase>_ram_avg_mb` | Средняя RAM, МБ |
| `<phase>_ram_peak_mb` | Пик RAM, МБ |
| `<phase>_gpu_util_avg_pct` | Средняя загрузка GPU (NVML), % |
| `<phase>_gpu_util_peak_pct` | Пик GPU, % |
| `<phase>_gpu_mem_peak_mb` | Пик VRAM (PyTorch), МБ |
| `<phase>_resource_samples` | Число замеров (шаг ~2 с) |

`<phase>` ∈ `train`, `val`.

### Production KPI (`production`)

| Метрика | Описание |
|---------|----------|
| `pipeline_duration_sec` | Сумма train + val + infer |
| `model_size_mb` | Размер `best.pt` |
| `train_images`, `test_images`, `num_classes` | Копия статистики датасета |
| `synthetic_to_real_ratio` | Эвристика аугментаций |
| `estimated_augmented_views_per_epoch` | Оценка «видов» за эпоху |
| + все `val_*`, `infer_*`, ресурсы, `hands_benchmark_*` | Дубли для сводных таблиц |

### Hands benchmark (`hands_benchmark_*`)

| Метрика | Описание |
|---------|----------|
| `hands_benchmark_images` | Всего кадров в прогоне |
| `hands_benchmark_images_with_predictions` | Кадры с непустым top-k |
| `hands_benchmark_prediction_rate` | Доля кадров с ответом модели |
| `hands_benchmark_evaluated_images` | Кадры с GT и pred (для accuracy) |
| `hands_benchmark_correct_total` | Верные top-1 |
| `hands_benchmark_incorrect_total` | Ошибки top-1 |
| `hands_benchmark_accuracy` | correct / evaluated |
| `hands_benchmark_top1_conf_avg` | Средняя уверенность top-1 (benchmark) |
| `hands_benchmark_unique_classes` | Число уникальных предсказанных классов |
| `hands_benchmark_infer_conf` | Порог `infer.conf` из конфига |

**Outcomes by hand** (график): для каждой `true_combo` — `correct`, `incorrect`.

**Confusion matrix** (таблица): строка = истина, столбец = pred, значение = `count`.

### Примеры предсказаний (`predictions[]`)

| Поле | Описание |
|------|----------|
| `index` | Индекс превью |
| `source_image` | Путь к исходному файлу |
| `source_dataset` | Метка набора |
| `preview_image` / `preview_url` | Локальный JPG / URL report-server |
| `predictions_count` | Размер top-k |
| `top_classes[]` | `{class_id, class_name, confidence}` |
| `true_combo` | GT комбинация |
| `predicted_combo` | Top-1 класс |
| `predicted_confidence` | Уверенность top-1 |
| `prediction_correct` | Сравнение pred с GT |

### Журнал событий (`events[]`)

| Поле | Описание |
|------|----------|
| `ts` | Время UTC |
| `action` | `pipeline.start`, `train.epoch`, `train.complete`, `validate.complete`, `infer.complete`, `pipeline.finish`, … |
| прочие ключи | Контекст (метрики эпохи, пути, ошибки) |

### Артефакты (`artifacts`)

| Ключ | Описание |
|------|----------|
| `weights` | Путь к `best.pt` |
| `training_results_csv` | Путь к `results.csv` |
| `inference_output` | Каталог `runs/infer/pred_*` |
| `predictions` | Каталог аннотированных кадров |

---

## Сводная таблица имён

| Смысл | JSON / HTML | MLflow (пример) | Prometheus / Grafana |
|-------|-------------|-----------------|----------------------|
| Val top-1 | `val_top1` | `val_metrics_accuracy_top1`, `top1` | `poker_yolo_val_top1` |
| Val top-5 | `val_top5` | `val_metrics_accuracy_top5`, `top5` | `poker_yolo_val_top5` |
| Train top-1 | `train_top1` | `train_metrics_accuracy_top1` | `poker_yolo_train_top1` * |
| Benchmark accuracy | `hands_benchmark_accuracy` | — (в JSON run train) | `poker_yolo_hands_benchmark_accuracy` |
| Infer latency | `infer_latency_ms` | `infer_latency_ms` | `poker_yolo_infer_latency_ms` |
| Pipeline time | `pipeline_duration_sec` | — | `poker_yolo_pipeline_duration_sec` |
| Epoch val loss (кривая) | `training_curves[].val_loss` | `epoch_val_loss` | `poker_yolo_train_epoch_metric{series="val_loss"}` |
| Outcomes bar | `benchmark_outcomes_by_hand` | — | Infinity `benchmark_outcomes.json` |
| Confusion cell | `benchmark_confusion` | — | Infinity `confusion_matrix.json` |

\* Метрики с префиксом `train_` из JSON экспортируются в Prometheus динамически, если присутствуют в отчёте; на training-дашборде по умолчанию отображаются только перечисленные в [GRAFANA_METRICS](../poker_yolo/observability.py).

---

*Документ сгенерирован по коду `poker_yolo` и provisioning Grafana. При добавлении новых метрик обновляйте этот файл и `GRAFANA_METRICS` в `observability.py`.*
