#!/usr/bin/env python3
"""Split legacy poker-yolo.json into training, curves, and inference dashboards."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "observability" / "grafana" / "provisioning" / "dashboards"
SRC = DASH / "poker-yolo.json"
TRAINING_OUT = DASH / "poker-yolo-training.json"
CURVES_OUT = DASH / "poker-yolo-curves.json"
INFERENCE_OUT = DASH / "poker-yolo-inference.json"

TRAINING_IDS = {100, 20, 21, 1, 2, 3}
CURVES_IDS = {100, 101, 102}
INFERENCE_IDS = {5, 10, 11, 12, 18, 19, 16, 17, 103}

TRAINING_LAYOUT = {
    100: {"h": 2, "w": 24, "x": 0, "y": 0},
    20: {"h": 6, "w": 12, "x": 0, "y": 2},
    21: {"h": 6, "w": 12, "x": 12, "y": 2},
    2: {"h": 6, "w": 8, "x": 0, "y": 8},
    1: {"h": 6, "w": 8, "x": 8, "y": 8},
    3: {"h": 6, "w": 8, "x": 16, "y": 8},
}

CURVES_LAYOUT = {
    100: {"h": 2, "w": 24, "x": 0, "y": 0},
    101: {"h": 10, "w": 12, "x": 0, "y": 2},
    102: {"h": 10, "w": 12, "x": 12, "y": 2},
}

INFERENCE_LAYOUT = {
    5: {"h": 2, "w": 24, "x": 0, "y": 0},
    10: {"h": 10, "w": 8, "x": 0, "y": 2},
    11: {"h": 10, "w": 8, "x": 8, "y": 2},
    12: {"h": 10, "w": 8, "x": 16, "y": 2},
    18: {"h": 4, "w": 8, "x": 0, "y": 12},
    19: {"h": 4, "w": 8, "x": 8, "y": 12},
    16: {"h": 4, "w": 8, "x": 16, "y": 12},
    17: {"h": 10, "w": 24, "x": 0, "y": 16},
    103: {"h": 14, "w": 24, "x": 0, "y": 26},
}

LINK_CURVES = "/d/poker-yolo-curves/poker-yolo-training-curves"
LINK_TRAINING = "/d/poker-yolo-training/poker-yolo-training"
LINK_INFERENCE = "/d/poker-yolo-inference/poker-yolo-benchmark-inference"


def _fix_curve_panel(panel: dict, *, value_fields: list[str]) -> dict:
    """Timeseries panel needs a time field; X axis uses numeric epoch via xField."""
    panel = copy.deepcopy(panel)
    panel.setdefault("options", {})
    panel["options"]["xField"] = "epoch"
    panel["options"].pop("timeField", None)

    defaults = panel.setdefault("fieldConfig", {}).setdefault("defaults", {})
    custom = defaults.setdefault("custom", {})
    custom["axisLabel"] = ""
    hide_viz = {
        "hideFrom": {"legend": True, "tooltip": False, "viz": True},
    }
    overrides = [
        o
        for o in panel["fieldConfig"].setdefault("overrides", [])
        if o.get("matcher", {}).get("options") not in {"epoch", "time"}
    ]
    overrides.extend(
        [
            {
                "matcher": {"id": "byName", "options": "epoch"},
                "properties": [
                    {"id": "displayName", "value": "Эпоха"},
                    {
                        "id": "custom",
                        "value": {
                            "axisLabel": "Эпоха",
                            "axisPlacement": "bottom",
                            "axisSoftMin": 0,
                            **hide_viz,
                        },
                    },
                ],
            },
            {
                "matcher": {"id": "byName", "options": "time"},
                "properties": [{"id": "custom", "value": hide_viz}],
            },
        ]
    )
    panel["fieldConfig"]["overrides"] = overrides

    target = panel["targets"][0]
    target["format"] = "timeseries"
    target["columns"] = [
        {
            "selector": "time",
            "text": "time",
            "type": "timestamp",
            "timestampFormat": "RFC3339",
        },
        {"selector": "epoch", "text": "epoch", "type": "number"},
        *[
            {"selector": name, "text": name, "type": "number"}
            for name in value_fields
        ],
    ]

    conversions = [
        {"destinationType": "time", "targetField": "time"},
        {"destinationType": "number", "targetField": "epoch"},
    ]
    conversions.extend(
        {"destinationType": "number", "targetField": name} for name in value_fields
    )
    panel["transformations"] = [
        {"id": "convertFieldType", "options": {"conversions": conversions}},
        {
            "id": "sortBy",
            "options": {"fields": {}, "sort": [{"desc": False, "field": "epoch"}]},
        },
    ]
    return panel


def _header_html(dashboard: str) -> str:
    if dashboard == "curves":
        return (
            "<h3>Кривые обучения</h3><p>Данные из "
            "<code>runs/reports/grafana/training_curves.json</code>; "
            "ось X — <strong>номер эпохи</strong>.</p>"
            f'<p><a href="{LINK_TRAINING}">→ Метрики обучения (CPU, RAM, длительность)</a> · '
            f'<a href="{LINK_INFERENCE}">→ Benchmark & inference</a></p>'
        )
    if dashboard == "training":
        return (
            "<h3>Метрики обучения</h3><p>CPU/RAM, длительность и итоговые MRPH val — "
            "из Pushgateway (последний успешный train).</p>"
            f'<p><a href="{LINK_CURVES}">→ Кривые loss / accuracy</a> · '
            f'<a href="{LINK_INFERENCE}">→ Benchmark & inference</a></p>'
        )
    return (
        "<h3>Benchmark: <code>dataset/test/images</code></h3>"
        "<p>Ground truth combo from YOLO card labels (<code>evaluate_hand</code>). "
        "Previews: <code>${report_server}/preview/</code>.</p>"
        f'<p><a href="{LINK_TRAINING}">→ Метрики обучения</a> · '
        f'<a href="{LINK_CURVES}">→ Кривые loss / accuracy</a></p>'
    )


def _apply_layout(
    panels: list[dict],
    layout: dict[int, dict],
    *,
    dashboard: str,
) -> list[dict]:
    out = []
    for panel in panels:
        pid = panel.get("id")
        if pid not in layout:
            continue
        panel = copy.deepcopy(panel)
        panel["gridPos"] = layout[pid]
        if pid == 100 or pid == 5:
            panel.setdefault("options", {})
            panel["options"]["content"] = _header_html(dashboard)
        if pid == 101:
            panel = _fix_curve_panel(panel, value_fields=["train_loss", "val_loss"])
        if pid == 102:
            panel = _fix_curve_panel(panel, value_fields=["top1", "top5"])
        out.append(panel)
    return sorted(out, key=lambda p: p["gridPos"]["y"])


def _link(title: str, url: str) -> dict:
    return {
        "asDropdown": False,
        "icon": "external link",
        "includeVars": False,
        "keepTime": True,
        "tags": [],
        "targetBlank": False,
        "title": title,
        "tooltip": "",
        "type": "link",
        "url": url,
    }


def _base_dashboard(panels: list[dict], *, title: str, uid: str, links: list, template: dict) -> dict:
    doc = copy.deepcopy(template)
    doc["panels"] = panels
    doc["title"] = title
    doc["uid"] = uid
    doc["links"] = links
    doc["version"] = 1
    doc["id"] = None
    return doc


def main() -> None:
    if not SRC.exists():
        print(f"No {SRC.name}; dashboards already split (training / curves / inference).")
        return

    doc = json.loads(SRC.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in doc["panels"]}

    training_panels = _apply_layout(
        [by_id[i] for i in TRAINING_IDS if i in by_id],
        TRAINING_LAYOUT,
        dashboard="training",
    )
    curves_panels = _apply_layout(
        [by_id[i] for i in CURVES_IDS if i in by_id],
        CURVES_LAYOUT,
        dashboard="curves",
    )
    inference_panels = _apply_layout(
        [by_id[i] for i in INFERENCE_IDS if i in by_id],
        INFERENCE_LAYOUT,
        dashboard="inference",
    )

    training = _base_dashboard(
        training_panels,
        title="Poker YOLO — Training",
        uid="poker-yolo-training",
        links=[
            _link("Training curves", LINK_CURVES),
            _link("Benchmark & inference", LINK_INFERENCE),
        ],
        template=doc,
    )
    curves = _base_dashboard(
        curves_panels,
        title="Poker YOLO — Training curves",
        uid="poker-yolo-curves",
        links=[
            _link("Training metrics", LINK_TRAINING),
            _link("Benchmark & inference", LINK_INFERENCE),
        ],
        template=doc,
    )
    inference = _base_dashboard(
        inference_panels,
        title="Poker YOLO — Benchmark & Inference",
        uid="poker-yolo-inference",
        links=[
            _link("Training metrics", LINK_TRAINING),
            _link("Training curves", LINK_CURVES),
        ],
        template=doc,
    )

    TRAINING_OUT.write_text(json.dumps(training, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    CURVES_OUT.write_text(json.dumps(curves, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    INFERENCE_OUT.write_text(json.dumps(inference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SRC.unlink()

    print(f"Wrote {TRAINING_OUT.name} ({len(training_panels)} panels)")
    print(f"Wrote {CURVES_OUT.name} ({len(curves_panels)} panels)")
    print(f"Wrote {INFERENCE_OUT.name} ({len(inference_panels)} panels)")
    print(f"Removed legacy {SRC.name}")


if __name__ == "__main__":
    main()
