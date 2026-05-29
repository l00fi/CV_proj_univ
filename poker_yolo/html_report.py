"""Interactive HTML pipeline report from RunReport JSON."""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from poker_yolo.benchmark_grids import normalize_confusion_matrix, normalize_outcomes_by_hand
from poker_yolo.hands import COMBO_CLASSES
from poker_yolo.reporting import RunReport

logger = logging.getLogger(__name__)

_CHART_JS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

_DATASET_KEYS = (
    "train_images",
    "val_images",
    "test_images",
    "num_classes",
    "benchmark_test_images",
    "benchmark_source",
)
_BENCHMARK_KPI_KEYS = (
    "hands_benchmark_images",
    "hands_benchmark_evaluated_images",
    "hands_benchmark_correct_total",
    "hands_benchmark_incorrect_total",
    "hands_benchmark_accuracy",
    "hands_benchmark_top1_conf_avg",
    "hands_benchmark_prediction_rate",
)


def load_report_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def report_from_dict(data: dict[str, Any]) -> RunReport:
    started = datetime.fromisoformat(data["started_at"])
    finished = None
    if data.get("finished_at"):
        finished = datetime.fromisoformat(data["finished_at"])
    report = RunReport(
        run_id=data["run_id"],
        phase=data["phase"],
        started_at=started,
        config_name=data.get("config_name", "unknown"),
    )
    report.status = data.get("status", "unknown")
    report.error = data.get("error")
    report.finished_at = finished
    report.params = dict(data.get("params", {}))
    report.metrics = {k: float(v) for k, v in data.get("metrics", {}).items()}
    report.resources = {k: float(v) for k, v in data.get("resources", {}).items()}
    report.augmentations_summary = dict(data.get("augmentations_summary", {}))
    report.dataset_stats = dict(data.get("dataset_stats", {}))
    report.predictions = list(data.get("predictions", []))
    report.benchmark_class_counts = dict(data.get("benchmark_class_counts", {}))
    report.benchmark_outcomes_by_hand = dict(data.get("benchmark_outcomes_by_hand", {}))
    report.benchmark_confusion = dict(data.get("benchmark_confusion", {}))
    report.training_curves = list(data.get("training_curves", []))
    report.production = {k: float(v) for k, v in data.get("production", {}).items()}
    report.artifacts = dict(data.get("artifacts", {}))
    report.events = list(data.get("events", []))
    return report


def write_html_report(
    report: RunReport | dict[str, Any],
    report_dir: Path,
    *,
    embed_previews: bool = True,
) -> Path:
    """Write ``{run_id}.html`` and ``latest.html`` under *report_dir*."""
    if isinstance(report, dict):
        run_id = report["run_id"]
        payload = report
        model = report_from_dict(report)
    else:
        run_id = report.run_id
        payload = report.to_dict()
        model = report

    report_dir.mkdir(parents=True, exist_ok=True)
    html = render_html(model, payload, report_dir=report_dir, embed_previews=embed_previews)
    html_path = report_dir / f"{run_id}.html"
    latest_path = report_dir / "latest.html"
    html_path.write_text(html, encoding="utf-8")
    latest_path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved: %s", html_path)
    return html_path


def write_html_report_from_json(
    json_path: Path,
    *,
    output: Path | None = None,
    embed_previews: bool = True,
) -> Path:
    """Build HTML from an existing report JSON (standalone use)."""
    data = load_report_json(json_path)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        model = report_from_dict(data)
        html = render_html(model, data, report_dir=output.parent, embed_previews=embed_previews)
        output.write_text(html, encoding="utf-8")
        logger.info("HTML report saved: %s", output)
        return output
    return write_html_report(data, json_path.parent, embed_previews=embed_previews)


def _duration_sec(data: dict[str, Any]) -> float | None:
    raw = data.get("duration_sec")
    if raw is not None:
        return float(raw)
    started = data.get("started_at")
    finished = data.get("finished_at")
    if not started or not finished:
        return None
    t0 = datetime.fromisoformat(started)
    t1 = datetime.fromisoformat(finished)
    return (t1 - t0).total_seconds()


def _group_metrics(metrics: dict[str, float]) -> dict[str, list[tuple[str, float]]]:
    groups: dict[str, list[tuple[str, float]]] = {
        "train": [],
        "val": [],
        "infer": [],
        "other": [],
    }
    for key, value in sorted(metrics.items()):
        if key.startswith("train_"):
            groups["train"].append((key, value))
        elif key.startswith("val_"):
            groups["val"].append((key, value))
        elif key.startswith("infer_"):
            groups["infer"].append((key, value))
        else:
            groups["other"].append((key, value))
    return {k: v for k, v in groups.items() if v}


def _preview_src(
    sample: dict[str, Any],
    report_dir: Path,
    *,
    embed: bool,
) -> str | None:
    local = sample.get("preview_image")
    if local:
        path = Path(local)
        if not path.is_absolute():
            path = report_dir / path
        if path.exists():
            if embed:
                mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"data:{mime};base64,{b64}"
            try:
                rel = path.relative_to(report_dir).as_posix()
                return rel
            except ValueError:
                return path.as_uri()
    url = sample.get("preview_url")
    return str(url) if url else None


def _chart_datasets(curves: list[dict[str, Any]]) -> dict[str, Any]:
    epochs = [int(c.get("epoch", i + 1)) for i, c in enumerate(curves)]
    return {
        "epochs": epochs,
        "train_loss": [c.get("train_loss") for c in curves],
        "val_loss": [c.get("val_loss") for c in curves],
        "top1": [c.get("top1") for c in curves],
        "top5": [c.get("top5") for c in curves],
    }


def _confusion_grid(raw: dict[str, dict[str, int]]) -> list[list[int]]:
    matrix = normalize_confusion_matrix(raw)
    return [[matrix[t][p] for p in COMBO_CLASSES] for t in COMBO_CLASSES]


def _outcomes_chart(raw: dict[str, dict[str, int]]) -> dict[str, list[int]]:
    outcomes = normalize_outcomes_by_hand(raw)
    return {
        "labels": list(COMBO_CLASSES),
        "correct": [outcomes[h]["correct"] for h in COMBO_CLASSES],
        "incorrect": [outcomes[h]["incorrect"] for h in COMBO_CLASSES],
    }


def _fmt_metric(value: float, key: str) -> str:
    if "accuracy" in key or key.endswith("_top1") or key.endswith("_top5") or "map" in key:
        if 0 <= value <= 1:
            return f"{value * 100:.2f}%"
    if key.endswith("_sec") or key.endswith("_seconds") or "duration" in key:
        return f"{value:.1f} s"
    if key.endswith("_mb"):
        return f"{value:.1f} MB"
    if key.endswith("_pct"):
        return f"{value:.1f}%"
    return f"{value:.4f}"


def _table_rows(pairs: list[tuple[str, Any]]) -> str:
    if not pairs:
        return "<tr><td colspan=\"2\" class=\"muted\">—</td></tr>"
    return "".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(_fmt_metric(v, str(k)) if isinstance(v, float) else str(v))}</td></tr>"
        for k, v in pairs
    )


def _kpi_cards(production: dict[str, float], metrics: dict[str, float]) -> str:
    cards: list[tuple[str, str, str]] = []
    priority = [
        ("hands_benchmark_accuracy", production, "Benchmark accuracy"),
        ("val_top1", metrics, "Val top-1"),
        ("val_top5", metrics, "Val top-5"),
        ("train_duration_sec", metrics, "Train duration"),
        ("pipeline_duration_sec", production, "Pipeline duration"),
    ]
    seen: set[str] = set()
    for key, source, label in priority:
        if key in source and key not in seen:
            seen.add(key)
            cards.append((label, _fmt_metric(source[key], key), key))
    for key, value in sorted(production.items()):
        if key.startswith("hands_benchmark_") and key not in seen:
            cards.append((key.replace("_", " "), _fmt_metric(value, key), key))
            seen.add(key)
    if not cards:
        return ""
    items = "".join(
        f'<div class="kpi"><div class="kpi-label">{escape(label)}</div>'
        f'<div class="kpi-value" title="{escape(key)}">{escape(val)}</div></div>'
        for label, val, key in cards[:8]
    )
    return f'<div class="kpi-grid">{items}</div>'


def render_html(
    report: RunReport,
    payload: dict[str, Any],
    *,
    report_dir: Path,
    embed_previews: bool,
) -> str:
    duration = _duration_sec(payload)
    duration_s = f"{duration:.1f} s" if duration is not None else "—"
    status = report.status
    status_class = "ok" if status == "success" else "fail" if status == "failed" else "run"

    chart_data = _chart_datasets(report.training_curves) if report.training_curves else None
    outcomes = _outcomes_chart(report.benchmark_outcomes_by_hand) if report.benchmark_outcomes_by_hand else None
    confusion = _confusion_grid(report.benchmark_confusion) if report.benchmark_confusion else None
    metric_groups = _group_metrics(report.metrics)

    dataset_rows = [(k, report.dataset_stats[k]) for k in _DATASET_KEYS if k in report.dataset_stats]
    benchmark_rows = [
        (k, report.production[k])
        for k in _BENCHMARK_KPI_KEYS
        if k in report.production
    ]
    aug = report.augmentations_summary
    aug_rows: list[tuple[str, Any]] = []
    for key in ("synthetic_to_real_ratio", "train_images_real", "estimated_augmented_views_per_epoch"):
        if key in aug:
            aug_rows.append((key, aug[key]))
    yolo_probs = aug.get("yolo_probabilities", {})
    alb_probs = aug.get("albumentations_probabilities", {})

    preview_blocks = []
    for sample in report.predictions:
        src = _preview_src(sample, report_dir, embed=embed_previews)
        top = sample.get("top_classes") or []
        top_str = ", ".join(
            f"{t.get('class_name', '?')} ({float(t.get('confidence', 0)):.2f})" for t in top[:3]
        ) or "—"
        correct = sample.get("prediction_correct")
        badge = (
            '<span class="badge ok">верно</span>'
            if correct is True
            else '<span class="badge fail">ошибка</span>'
            if correct is False
            else ""
        )
        img_html = (
            f'<img src="{escape(src)}" alt="sample {sample.get("index", 0)}" loading="lazy">'
            if src
            else '<div class="no-img">нет превью</div>'
        )
        preview_blocks.append(
            f'<article class="sample-card">{img_html}'
            f"<h4>Пример {sample.get('index', 0)} {badge}</h4>"
            f"<p><b>Источник:</b> <code>{escape(str(sample.get('source_image', '—')))}</code></p>"
            f"<p><b>Истина:</b> {escape(str(sample.get('true_combo', '—')))} · "
            f"<b>Предсказание:</b> {escape(str(sample.get('predicted_combo', '—')))} "
            f"({float(sample.get('predicted_confidence', 0)):.2f})</p>"
            f"<p><b>Top classes:</b> {escape(top_str)}</p></article>"
        )

    events_html = "".join(
        f'<li><time>{escape(str(e.get("ts", "")))}</time> '
        f'<strong>{escape(str(e.get("action", "")))}</strong> '
        f'<code>{escape(json.dumps({k: v for k, v in e.items() if k not in {"ts", "action"}}, ensure_ascii=False))}</code></li>'
        for e in report.events
    )

    artifacts_html = "".join(
        f'<li><b>{escape(k)}</b>: <code>{escape(str(v))}</code></li>'
        for k, v in sorted(report.artifacts.items())
    ) or "<li class=\"muted\">—</li>"

    confusion_table = ""
    if confusion:
        header = "".join(f"<th>{escape(h)}</th>" for h in COMBO_CLASSES)
        body_rows = []
        max_cell = max((max(row) for row in confusion), default=1) or 1
        for i, true_hand in enumerate(COMBO_CLASSES):
            cells = []
            for j, count in enumerate(confusion[i]):
                intensity = count / max_cell if max_cell else 0
                bg = f"rgba(59, 130, 246, {0.08 + intensity * 0.72})"
                bold = "font-weight:700" if i == j and count else ""
                cells.append(
                    f'<td style="background:{bg};{bold}" title="{escape(true_hand)} → {escape(COMBO_CLASSES[j])}">{count}</td>'
                )
            body_rows.append(f"<tr><th>{escape(true_hand)}</th>{''.join(cells)}</tr>")
        confusion_table = (
            f'<div class="table-wrap"><table class="heatmap"><thead><tr><th>true \\ pred</th>{header}</tr></thead>'
            f"<tbody>{''.join(body_rows)}</tbody></table></div>"
        )

    chart_script = ""
    if chart_data or outcomes:
        chart_script = f"""
<script src="{_CHART_JS}"></script>
<script>
const chartDefaults = {{ responsive: true, maintainAspectRatio: false }};
"""
        if chart_data:
            chart_script += f"""
const curves = {json.dumps(chart_data, ensure_ascii=False)};
const lossCtx = document.getElementById('lossChart');
if (lossCtx && curves.epochs.length) {{
  new Chart(lossCtx, {{
    type: 'line',
    data: {{
      labels: curves.epochs,
      datasets: [
        {{ label: 'train_loss', data: curves.train_loss, borderColor: '#3b82f6', tension: 0.25 }},
        {{ label: 'val_loss', data: curves.val_loss, borderColor: '#f59e0b', tension: 0.25 }},
      ],
    }},
    options: {{
      ...chartDefaults,
      scales: {{ x: {{ title: {{ display: true, text: 'Эпоха' }} }} }},
      plugins: {{ legend: {{ position: 'bottom' }} }},
    }},
  }});
}}
const accCtx = document.getElementById('accChart');
if (accCtx && curves.epochs.length) {{
  new Chart(accCtx, {{
    type: 'line',
    data: {{
      labels: curves.epochs,
      datasets: [
        {{ label: 'top1', data: curves.top1, borderColor: '#10b981', tension: 0.25 }},
        {{ label: 'top5', data: curves.top5, borderColor: '#8b5cf6', tension: 0.25 }},
      ],
    }},
    options: {{
      ...chartDefaults,
      scales: {{ y: {{ min: 0, max: 1 }}, x: {{ title: {{ display: true, text: 'Эпоха' }} }} }},
      plugins: {{ legend: {{ position: 'bottom' }} }},
    }},
  }});
}}
"""
        if outcomes:
            chart_script += f"""
const outcomes = {json.dumps(outcomes, ensure_ascii=False)};
const outCtx = document.getElementById('outcomesChart');
if (outCtx) {{
  new Chart(outCtx, {{
    type: 'bar',
    data: {{
      labels: outcomes.labels,
      datasets: [
        {{ label: 'Correct', data: outcomes.correct, backgroundColor: '#10b981' }},
        {{ label: 'Incorrect', data: outcomes.incorrect, backgroundColor: '#ef4444' }},
      ],
    }},
    options: {{
      ...chartDefaults,
      scales: {{ x: {{ stacked: true }}, y: {{ stacked: true }} }},
      plugins: {{ legend: {{ position: 'bottom' }} }},
    }},
  }});
}}
"""
        chart_script += "</script>"

    metric_sections = ""
    group_titles = {"train": "Обучение", "val": "Валидация", "infer": "Inference", "other": "Прочие"}
    for group, title in group_titles.items():
        rows = metric_groups.get(group, [])
        if rows:
            metric_sections += (
                f"<h3>{title}</h3><table class=\"data-table\">"
                f"<tbody>{_table_rows(rows)}</tbody></table>"
            )

    resource_section = ""
    if report.resources:
        resource_section = (
            f"<h3>Ресурсы (CPU / RAM / GPU)</h3><table class=\"data-table\">"
            f"<tbody>{_table_rows(sorted(report.resources.items()))}</tbody></table>"
        )

    production_section = ""
    if report.production:
        prod_rows = [(k, v) for k, v in sorted(report.production.items()) if k not in _BENCHMARK_KPI_KEYS]
        if prod_rows:
            production_section = (
                f"<h3>Production KPI</h3><table class=\"data-table\">"
                f"<tbody>{_table_rows(prod_rows)}</tbody></table>"
            )

    yolo_table = ""
    if yolo_probs:
        yolo_table = (
            "<h4>YOLO augmentations</h4><table class=\"data-table\"><tbody>"
            + _table_rows(sorted(yolo_probs.items()))
            + "</tbody></table>"
        )
    alb_table = ""
    if alb_probs:
        alb_table = (
            "<h4>Albumentations</h4><table class=\"data-table\"><tbody>"
            + _table_rows(sorted(alb_probs.items()))
            + "</tbody></table>"
        )

    error_block = ""
    if report.error:
        error_block = f'<section class="card error-card" id="error"><h2>Ошибка</h2><pre>{escape(report.error)}</pre></section>'

    curves_section = ""
    if chart_data:
        curves_section = """
<section class="card" id="curves">
  <h2>Кривые обучения</h2>
  <div class="chart-row">
    <div class="chart-box"><canvas id="lossChart"></canvas></div>
    <div class="chart-box"><canvas id="accChart"></canvas></div>
  </div>
</section>"""

    benchmark_section = ""
    if benchmark_rows or outcomes or confusion:
        outcomes_canvas = (
            '<div class="chart-box tall"><canvas id="outcomesChart"></canvas></div>'
            if outcomes
            else ""
        )
        benchmark_section = f"""
<section class="card" id="benchmark">
  <h2>Hands benchmark (dataset/test/images)</h2>
  <p class="muted">Ground truth из YOLO-меток карт через <code>evaluate_hand</code>.</p>
  <table class="data-table"><tbody>{_table_rows(benchmark_rows)}</tbody></table>
  <div class="chart-row">{outcomes_canvas}</div>
  <h3>Confusion matrix (top-1)</h3>
  {confusion_table or '<p class="muted">Нет данных</p>'}
</section>"""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Poker YOLO — {escape(report.phase)} — {escape(report.run_id)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e7ecf3;
      --muted: #94a3b8;
      --accent: #3b82f6;
      --ok: #10b981;
      --fail: #ef4444;
      --border: #2d3a4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    header {{
      padding: 1.5rem 2rem;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    }}
    header h1 {{ margin: 0 0 0.25rem; font-size: 1.5rem; }}
    .meta {{ color: var(--muted); font-size: 0.9rem; }}
    .status {{
      display: inline-block;
      padding: 0.15rem 0.55rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .status.ok {{ background: rgba(16,185,129,0.2); color: var(--ok); }}
    .status.fail {{ background: rgba(239,68,68,0.2); color: var(--fail); }}
    .status.run {{ background: rgba(59,130,246,0.2); color: var(--accent); }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem 1rem;
      padding: 0.75rem 2rem;
      background: rgba(15,20,25,0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 0.9rem;
    }}
    nav a:hover {{ text-decoration: underline; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.25rem;
    }}
    .card h2 {{ margin-top: 0; font-size: 1.15rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0;
    }}
    .kpi {{
      background: rgba(59,130,246,0.08);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
    }}
    .kpi-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-value {{ font-size: 1.35rem; font-weight: 700; margin-top: 0.25rem; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    table.data-table th, table.data-table td {{
      text-align: left;
      padding: 0.4rem 0.6rem;
      border-bottom: 1px solid var(--border);
    }}
    table.data-table th {{ color: var(--muted); font-weight: 500; width: 42%; }}
    .table-wrap {{ overflow-x: auto; }}
    table.heatmap {{ border-collapse: collapse; font-size: 0.72rem; }}
    table.heatmap th, table.heatmap td {{
      border: 1px solid var(--border);
      padding: 0.35rem 0.4rem;
      text-align: center;
      min-width: 2.2rem;
    }}
    table.heatmap th {{ background: #243044; color: var(--muted); }}
    .chart-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }}
    .chart-box {{
      position: relative;
      height: 260px;
      background: rgba(0,0,0,0.15);
      border-radius: 8px;
      padding: 0.5rem;
    }}
    .chart-box.tall {{ height: 320px; }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 1rem;
    }}
    .sample-card img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .sample-card h4 {{ margin: 0.5rem 0 0.25rem; font-size: 0.95rem; }}
    .sample-card p {{ margin: 0.2rem 0; font-size: 0.85rem; }}
    .no-img {{
      height: 160px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #243044;
      border-radius: 8px;
      color: var(--muted);
    }}
    .badge {{
      font-size: 0.7rem;
      padding: 0.1rem 0.4rem;
      border-radius: 4px;
      margin-left: 0.35rem;
    }}
    .badge.ok {{ background: rgba(16,185,129,0.25); color: var(--ok); }}
    .badge.fail {{ background: rgba(239,68,68,0.25); color: var(--fail); }}
    .muted {{ color: var(--muted); }}
    .error-card pre {{
      background: #0b0f14;
      padding: 1rem;
      border-radius: 8px;
      overflow-x: auto;
      color: #fca5a5;
    }}
    ul.timeline {{
      list-style: none;
      padding: 0;
      margin: 0;
      max-height: 320px;
      overflow-y: auto;
    }}
    ul.timeline li {{
      padding: 0.35rem 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.82rem;
    }}
    ul.timeline time {{ color: var(--muted); margin-right: 0.5rem; }}
    code {{ font-size: 0.85em; word-break: break-all; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Poker YOLO — отчёт пайплайна</h1>
    <p class="meta">
      <span class="status {status_class}">{escape(status)}</span>
      · фаза <b>{escape(report.phase)}</b>
      · конфиг <code>{escape(report.config_name)}</code>
      · run <code>{escape(report.run_id)}</code>
    </p>
    <p class="meta">
      Старт: {escape(report.started_at.isoformat())}
      · Финиш: {escape(report.finished_at.isoformat() if report.finished_at else "—")}
      · Длительность: {escape(duration_s)}
    </p>
  </header>
  <nav>
    <a href="#overview">Обзор</a>
    <a href="#dataset">Датасет</a>
    <a href="#params">Параметры</a>
    <a href="#curves">Кривые</a>
    <a href="#metrics">Метрики</a>
    <a href="#benchmark">Benchmark</a>
    <a href="#samples">Примеры</a>
    <a href="#timeline">События</a>
    <a href="#artifacts">Артефакты</a>
  </nav>
  <main>
    {error_block}
    <section class="card" id="overview">
      <h2>Обзор</h2>
      {_kpi_cards(report.production, report.metrics)}
    </section>
    <section class="card" id="dataset">
      <h2>Датасет</h2>
      <table class="data-table"><tbody>{_table_rows(dataset_rows)}</tbody></table>
    </section>
    <section class="card" id="params">
      <h2>Параметры и аугментации</h2>
      <h3>Конфигурация запуска</h3>
      <table class="data-table"><tbody>{_table_rows(sorted(report.params.items()))}</tbody></table>
      <h3>Аугментации</h3>
      <table class="data-table"><tbody>{_table_rows(aug_rows)}</tbody></table>
      {yolo_table}
      {alb_table}
    </section>
    {curves_section}
    <section class="card" id="metrics">
      <h2>Метрики обучения / валидации / inference</h2>
      {metric_sections or '<p class="muted">Нет числовых метрик</p>'}
      {resource_section}
      {production_section}
    </section>
    {benchmark_section}
    <section class="card" id="samples">
      <h2>Примеры предсказаний</h2>
      <div class="samples">{"".join(preview_blocks) or '<p class="muted">Нет превью (запустите inference на benchmark)</p>'}</div>
    </section>
    <section class="card" id="timeline">
      <h2>Журнал событий</h2>
      <ul class="timeline">{events_html or '<li class="muted">—</li>'}</ul>
    </section>
    <section class="card" id="artifacts">
      <h2>Артефакты</h2>
      <ul>{artifacts_html}</ul>
    </section>
  </main>
  <footer>Сгенерировано poker_yolo.html_report · {escape(datetime.now().isoformat(timespec="seconds"))}</footer>
  {chart_script}
</body>
</html>"""
