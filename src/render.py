"""Write the machine-readable outputs and the static results page.

  * results.json  — every per-image record + run summary
  * metrics.json  — the headline quantitative metrics
  * results.html  — a self-contained static page: metrics tables at the top,
                    then per-image side-by-side cards (annotated YOLO image,
                    VLM observations, and the one-line report).
No JS, no framework — plain HTML/CSS.
"""
from __future__ import annotations

import html
import json
import os
from typing import Optional


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def _pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _esc(x: object) -> str:
    return html.escape(str(x))


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def render_results_html(path: str, records: list[dict], metrics: dict, meta: dict) -> None:
    det_label = ("Fine-tuned detector"
                 if (meta.get("detector_backend") or "yolo_world") == "finetuned" else "YOLO-World")
    parts = [_HEAD, _header(meta), _metrics_section(metrics, det_label), _gallery(records), _FOOT]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def _header(meta: dict) -> str:
    rows = [
        ("Dataset", f"{meta.get('dataset_dir')} (split: {meta.get('split')})"),
        ("Images", f"{meta.get('num_shown')} shown of {meta.get('num_total')} evaluated"),
        ("Detector", f"{meta.get('yolo_model') or '—'} ({meta.get('detector_backend') or 'yolo_world'})"),
        ("VLM backend", meta.get("vlm_backend") or "—"),
        ("Device", meta.get("device") or "—"),
        ("Confidence threshold", meta.get("confidence_threshold")),
        ("Generated", meta.get("generated_at")),
    ]
    cells = "".join(
        f"<div class='meta-k'>{_esc(k)}</div><div class='meta-v'>{_esc(v)}</div>"
        for k, v in rows
    )
    return (
        "<header><h1>Construction Safety Inspection — Results</h1>"
        f"<div class='meta-grid'>{cells}</div></header>"
    )


def _metrics_section(metrics: dict, det_label: str = "YOLO-World") -> str:
    return (
        "<section class='metrics'><h2>Quantitative metrics (vs ground-truth labels)</h2>"
        + _yolo_metrics_table(metrics.get("yolo_world", {}), det_label)
        + _vlm_metrics_table(metrics.get("vlm", {}))
        + "<p class='hint'>How to read these: high <b>recall</b> on the violation "
          "classes (NO-Hardhat, NO-Safety Vest) means few missed hazards; high "
          "<b>precision</b> means few false alarms. See the README's "
          "&ldquo;How to read the metrics&rdquo; section to decide whether zero-shot "
          "is good enough or fine-tuning is warranted.</p></section>"
    )


def _yolo_metrics_table(m: dict, det_label: str = "YOLO-World") -> str:
    if not m.get("available"):
        return (f"<h3>{_esc(det_label)} detection metrics</h3>"
                f"<p class='note'>Not available: {_esc(m.get('reason', 'pipeline not run'))}</p>")
    head = ("<tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th>"
            "<th>F1</th><th>AP@50</th><th>TP</th><th>FP</th><th>FN</th></tr>")
    body = ""
    for name, c in m["per_class"].items():
        viol = name.upper().startswith("NO-")
        body += (
            f"<tr class='{ 'violrow' if viol else '' }'><td>{_esc(name)}</td>"
            f"<td>{c['support']}</td><td>{_pct(c['precision'])}</td>"
            f"<td>{_pct(c['recall'])}</td><td>{_pct(c['f1'])}</td>"
            f"<td>{_pct(c['ap50'])}</td><td>{c['tp']}</td><td>{c['fp']}</td>"
            f"<td>{c['fn']}</td></tr>"
        )
    ov = m["overall"]
    foot = (
        f"<tr class='overall'><td>overall (macro)</td><td>{ov['support_total']}</td>"
        f"<td>{_pct(ov['precision_macro'])}</td><td>{_pct(ov['recall_macro'])}</td>"
        f"<td>{_pct(ov['f1_macro'])}</td><td>{_pct(ov['mAP50'])}</td>"
        f"<td colspan='3'>mAP@50 = {_pct(ov['mAP50'])}</td></tr>"
    )
    return (f"<h3>{_esc(det_label)} detection metrics "
            f"(IoU {m['iou_threshold']}, conf ≥ {m['confidence_threshold']})</h3>"
            f"<table class='metric'>{head}{body}{foot}</table>")


def _vlm_metrics_table(m: dict) -> str:
    if not m.get("available"):
        return ("<h3>VLM image-level metrics</h3>"
                f"<p class='note'>Not available: {_esc(m.get('reason', 'pipeline not run'))}</p>")
    head = ("<tr><th>Class</th><th>Support (img)</th><th>Precision</th><th>Recall</th>"
            "<th>F1</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>")
    body = ""
    for name, c in m["per_class"].items():
        body += (
            f"<tr class='violrow'><td>{_esc(name)}</td><td>{c['support_images']}</td>"
            f"<td>{_pct(c['precision'])}</td><td>{_pct(c['recall'])}</td>"
            f"<td>{_pct(c['f1'])}</td><td>{c['tp']}</td><td>{c['fp']}</td>"
            f"<td>{c['fn']}</td><td>{c['tn']}</td></tr>"
        )
    ov = m["overall"]
    foot = (f"<tr class='overall'><td colspan='2'>overall (macro)</td>"
            f"<td>{_pct(ov['precision_macro'])}</td><td>{_pct(ov['recall_macro'])}</td>"
            f"<td>{_pct(ov['f1_macro'])}</td><td colspan='4'></td></tr>")
    return (f"<h3>VLM image-level metrics ({m['num_images']} images)</h3>"
            f"<table class='metric'>{head}{body}{foot}</table>")


def _gallery(records: list[dict]) -> str:
    cards = "".join(_card(r) for r in records)
    return ("<section class='gallery'><h2>Per-image review</h2>"
            f"<div class='grid'>{cards}</div></section>")


def _card(r: dict) -> str:
    img = (f"<img src='{_esc(r['display_image'])}' alt='{_esc(r['image'])}' loading='lazy'>"
           if r.get("display_image") else "")
    report = f"<p class='report'>{_esc(r.get('report', ''))}</p>"

    # YOLO chips
    yw = r.get("yolo_world", {})
    if yw.get("ran"):
        chips = "".join(_chip(d) for d in yw.get("detections", []))
        yolo_html = (f"<div class='col'><h4>YOLO-World ({yw.get('num_detections', 0)})</h4>"
                     f"<div class='chips'>{chips or '<span class=muted>none</span>'}</div></div>")
    else:
        yolo_html = "<div class='col'><h4>YOLO-World</h4><span class='muted'>skipped</span></div>"

    # VLM observations
    vlm = r.get("vlm", {})
    if vlm.get("ran"):
        obs = vlm.get("observations", [])
        if obs:
            items = "".join(
                f"<li><span class='sev sev-{_esc(o.get('severity', 'medium'))}'>"
                f"{_esc(o.get('severity', ''))}</span> "
                f"<b>{_esc(o.get('type', ''))}</b>: {_esc(o.get('description', ''))}</li>"
                for o in obs
            )
            vlm_inner = f"<ul class='obs'>{items}</ul>"
        elif vlm.get("error"):
            vlm_inner = f"<span class='muted'>error: {_esc(vlm['error'])}</span>"
        else:
            vlm_inner = "<span class='muted'>no observations</span>"
        vlm_html = f"<div class='col'><h4>VLM</h4>{vlm_inner}</div>"
    else:
        vlm_html = "<div class='col'><h4>VLM</h4><span class='muted'>skipped</span></div>"

    return (f"<div class='card'><div class='imgwrap'>{img}</div>"
            f"<div class='cap'>{_esc(r['image'])}</div>{report}"
            f"<div class='cols'>{yolo_html}{vlm_html}</div></div>")


def _chip(d: dict) -> str:
    name = d.get("class_name") or d.get("label") or "?"
    viol = str(name).upper().startswith("NO-")
    cls = "chip viol" if viol else ("chip" if d.get("class_name") else "chip un")
    return f"<span class='{cls}'>{_esc(name)} {d.get('confidence', 0):.2f}</span>"


_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construction Safety Inspection — Results</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--ink:#1f2430;--muted:#8a93a3;--red:#d6453d;--orange:#e08a16;--green:#2f9e44;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
header,section{max-width:1180px;margin:0 auto;padding:18px 22px}
h1{margin:0 0 12px;font-size:22px}
h2{font-size:18px;border-bottom:2px solid var(--line);padding-bottom:6px;margin-top:8px}
h3{font-size:15px;margin:18px 0 6px}
h4{margin:0 0 6px;font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.meta-grid{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.meta-k{color:var(--muted)}.meta-v{font-weight:600}
table.metric{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px}
table.metric th,table.metric td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
table.metric th:first-child,table.metric td:first-child{text-align:left}
table.metric th{background:#eef1f5;font-weight:600}
table.metric tr.violrow td:first-child{color:var(--red);font-weight:600}
table.metric tr.overall td{background:#f0f3f7;font-weight:700;border-top:2px solid var(--line)}
.hint{color:#4a5160;background:#eef4ff;border:1px solid #d6e2ff;border-radius:8px;padding:10px 12px}
.note{color:var(--muted);font-style:italic}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column}
.imgwrap{background:#0d1117;border-radius:6px;overflow:hidden;display:flex;justify-content:center}
.card img{max-width:100%;height:auto;display:block}
.cap{font-size:11px;color:var(--muted);margin:6px 0 2px;word-break:break-all}
.report{font-weight:600;margin:4px 0 8px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:auto}
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{font-size:11px;padding:2px 6px;border-radius:10px;background:#e8f5e9;color:#256029;border:1px solid #c7e7cb}
.chip.viol{background:#fdeceb;color:#a3271f;border-color:#f3c3bf}
.chip.un{background:#fff4e0;color:#8a5a00;border-color:#f1d9a8}
.obs{margin:0;padding-left:16px}.obs li{margin-bottom:4px}
.sev{font-size:10px;padding:1px 5px;border-radius:8px;color:#fff;text-transform:uppercase}
.sev-low{background:var(--muted)}.sev-medium{background:var(--orange)}.sev-high{background:var(--red)}
.muted{color:var(--muted)}
footer{max-width:1180px;margin:0 auto;padding:18px 22px;color:var(--muted);font-size:12px}
</style></head><body>"""

_FOOT = "<footer>Static results page — generated by run.py. No tracking, no JavaScript.</footer></body></html>"
