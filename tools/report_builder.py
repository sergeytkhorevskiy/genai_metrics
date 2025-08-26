#!/usr/bin/env python3
"""
Generate an HTML report with charts and top-k worst cases per metric.
Now supports BOTH schemas:
  A) Original flat schema expected before (keys: text, image_fid, image_inception, audio)
  B) "demo_*_model" schema with nested metrics (model_type in {text,image,audio})

Usage:
  python report_builder.py --results /path/to/results.json --out /path/to/report.html [--top_k 5]
"""
from __future__ import annotations
import json, base64
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Utils ---------------------------------

def _save_chart(values: List[float], labels: List[str], out_png: Path, title: str) -> None:
    # sanitize values: replace None/NaN with 0.0
    clean_vals = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]
    plt.figure()
    plt.bar(labels, clean_vals)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()


def _top_k(items: List[Dict[str, Any]], key: str, k: int = 5, reverse: bool = False) -> List[Dict[str, Any]]:
    try:
        return sorted(items, key=lambda x: x.get(key, float('nan')), reverse=reverse)[:k]
    except Exception:
        return items[:k]


def _get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def _first_model(data: Dict[str, Any], model_type: str) -> str | None:
    """Return first key where value.model_type == model_type, else None."""
    for k, v in data.items():
        if isinstance(v, dict) and v.get("model_type") == model_type:
            return k
    return None


def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize various incoming schemas into the flat structure used by the renderer.

    Output keys:
      text: {bleu, sacrebleu, chrf, rouge1_f, rouge2_f, rougeL_f, per_sample?}
      image_fid: float
      image_inception: {mean}
      audio: {snr_mean, stoi_mean, per_sample?}
      _profile?: any
    """
    out: Dict[str, Any] = {}

    # -------------------- TEXT --------------------
    # Prefer an explicit "text" block; else detect a model with model_type=="text".
    text_key = "text" if isinstance(data.get("text"), dict) else _first_model(data, "text")
    if text_key:
        t = data[text_key]
        out["text"] = {
            # Try nested form (bleu.bleu_score / sacrebleu_score / chrf_score), else flat
            "bleu": _get(t, ["bleu", "bleu_score"], t.get("bleu", 0.0) if isinstance(t.get("bleu"), (int, float)) else 0.0),
            "sacrebleu": _get(t, ["bleu", "sacrebleu_score"], t.get("sacrebleu", 0.0) if isinstance(t.get("sacrebleu"), (int, float)) else 0.0),
            "chrf": _get(t, ["bleu", "chrf_score"], t.get("chrf", 0.0) if isinstance(t.get("chrf"), (int, float)) else 0.0),
            # ROUGE nested (rouge.rougeX.fmeasure) or flat rougeX_f
            "rouge1_f": _get(t, ["rouge", "rouge1", "fmeasure"], t.get("rouge1_f", 0.0)),
            "rouge2_f": _get(t, ["rouge", "rouge2", "fmeasure"], t.get("rouge2_f", 0.0)),
            "rougeL_f": _get(t, ["rouge", "rougeL", "fmeasure"], t.get("rougeL_f", 0.0)),
        }
        # Per-sample arrays may appear as per_sample / samples
        per_sample = t.get("per_sample") or t.get("samples") or []
        if isinstance(per_sample, list):
            out["text"]["per_sample"] = per_sample

    # -------------------- IMAGE --------------------
    image_key = _first_model(data, "image")
    if not image_key and isinstance(data.get("image"), dict):
        image_key = "image"
    if image_key:
        i = data[image_key]
        out["image_fid"] = _get(i, ["fid_score"], i.get("image_fid", 0.0)) or 0.0
        inc = _get(i, ["inception_score"], i.get("image_inception", {})) or {}
        out["image_inception"] = {"mean": _get(inc, ["mean"], 0.0)}

    # -------------------- AUDIO --------------------
    audio_key = _first_model(data, "audio")
    if not audio_key and isinstance(data.get("audio"), dict):
        audio_key = "audio"
    if audio_key:
        a = data[audio_key]
        out["audio"] = {
            "snr_mean": _get(a, ["snr", "mean"], a.get("snr_mean", 0.0)) or 0.0,
            "stoi_mean": _get(a, ["stoi", "mean"], a.get("stoi_mean", 0.0)) or 0.0,
        }
        if isinstance(a.get("per_sample"), list):
            out["audio"]["per_sample"] = a["per_sample"]

    # -------------------- Profile passthrough --------------------
    for prof_key in ("_profile", "profile", "profiling"):
        if prof_key in data:
            out["_profile"] = data[prof_key]
            break

    return out


# ---------------------------- Renderer -------------------------------

def build_report(results_json_path: str, out_html_path: str, top_k: int = 5) -> None:
    raw_data = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    data = _normalize(raw_data)

    out_dir = Path(out_html_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    charts: List[tuple[str, Path]] = []

    # Text chart
    if isinstance(data.get("text"), dict):
        t = data["text"]
        vals = [
            float(t.get("bleu", 0.0) or 0.0),
            float(t.get("sacrebleu", 0.0) or 0.0),
            float(t.get("chrf", 0.0) or 0.0),
            float(t.get("rouge1_f", 0.0) or 0.0),
            float(t.get("rouge2_f", 0.0) or 0.0),
            float(t.get("rougeL_f", 0.0) or 0.0),
        ]
        labels = ["BLEU","SacreBLEU","chrF","ROUGE-1","ROUGE-2","ROUGE-L"]
        png = out_dir / "text_metrics.png"
        _save_chart(vals, labels, png, "Text metrics")
        charts.append(("Text metrics", png))

    # Image chart
    if ("image_fid" in data) or ("image_inception" in data):
        vals: List[float] = []
        labels: List[str] = []
        if "image_fid" in data:
            vals.append(float(data.get("image_fid") or 0.0)); labels.append("FID")
        if isinstance(data.get("image_inception"), dict):
            vals.append(float(data["image_inception"].get("mean", 0.0) or 0.0)); labels.append("IS mean")
        if labels:
            png = out_dir / "image_metrics.png"
            _save_chart(vals, labels, png, "Image metrics")
            charts.append(("Image metrics", png))

    # Audio chart
    if isinstance(data.get("audio"), dict):
        a = data["audio"]
        vals = [float(a.get("snr_mean", 0.0) or 0.0), float(a.get("stoi_mean", 0.0) or 0.0)]
        labels = ["SNR (dB)","STOI"]
        png = out_dir / "audio_metrics.png"
        _save_chart(vals, labels, png, "Audio metrics")
        charts.append(("Audio metrics", png))

    # --------------------- Build HTML ---------------------
    html: List[str] = [
        "<html><head><meta charset='utf-8'><title>GenAI Evaluation Report</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;} th{background:#f4f4f4;text-align:left;} code{white-space:pre-wrap;}</style>",
        "</head><body>",
    ]
    html.append("<h1>GenAI Evaluation Report</h1>")

    # Profile
    if data.get("_profile") is not None:
        html.append("<h2>Profiling</h2><pre>")
        html.append(json.dumps(data["_profile"], ensure_ascii=False, indent=2))
        html.append("</pre>")

    # Charts
    if charts:
        html.append("<h2>Metrics</h2>")
        for title, png in charts:
            html.append(f"<h3>{title}</h3>")
            b = Path(png).read_bytes()
            b64 = base64.b64encode(b).decode("ascii")
            html.append(f"<img src='data:image/png;base64,{b64}' alt='{title}' />")

    # Top-k worst cases (TEXT)
    if isinstance(data.get("text"), dict) and isinstance(data["text"].get("per_sample"), list):
        samples = data["text"]["per_sample"]
        if samples:
            html.append("<h2>Text: Top-k worst examples</h2>")
            worst_bleu = _top_k(samples, "sentence_bleu", k=top_k, reverse=False)
            worst_rl = _top_k(samples, "rougeL_f", k=top_k, reverse=False)
            html.append("<h3>Lowest sentence BLEU</h3>")
            html.append("<table><tr><th>#</th><th>Ref</th><th>Cand</th><th>sentBLEU</th><th>ROUGE-L</th><th>Len ratio</th></tr>")
            for s in worst_bleu:
                html.append(
                    f"<tr><td>{s.get('idx','')}</td><td><code>{s.get('reference','')}</code></td><td><code>{s.get('candidate','')}</code></td>"
                    f"<td>{float(s.get('sentence_bleu',0) or 0):.4f}</td><td>{float(s.get('rougeL_f',0) or 0):.4f}</td><td>{float(s.get('len_ratio',0) or 0):.2f}</td></tr>"
                )
            html.append("</table>")

            html.append("<h3>Lowest ROUGE-L</h3>")
            html.append("<table><tr><th>#</th><th>Ref</th><th>Cand</th><th>ROUGE-L</th><th>sentBLEU</th><th>Len ratio</th></tr>")
            for s in worst_rl:
                html.append(
                    f"<tr><td>{s.get('idx','')}</td><td><code>{s.get('reference','')}</code></td><td><code>{s.get('candidate','')}</code></td>"
                    f"<td>{float(s.get('rougeL_f',0) or 0):.4f}</td><td>{float(s.get('sentence_bleu',0) or 0):.4f}</td><td>{float(s.get('len_ratio',0) or 0):.2f}</td></tr>"
                )
            html.append("</table>")

    # Top-k worst cases (AUDIO)
    if isinstance(data.get("audio"), dict) and isinstance(data["audio"].get("per_sample"), list):
        samples = data["audio"]["per_sample"]
        if samples:
            html.append("<h2>Audio: Top-k worst examples</h2>")
            worst_snr = _top_k(samples, "snr", k=top_k, reverse=False)
            worst_stoi = _top_k(samples, "stoi", k=top_k, reverse=False)
            html.append("<h3>Lowest SNR</h3>")
            html.append("<table><tr><th>Ref</th><th>Cand</th><th>SNR (dB)</th><th>STOI</th></tr>")
            for s in worst_snr:
                html.append(f"<tr><td>{s.get('ref','')}</td><td>{s.get('cand','')}</td><td>{float(s.get('snr',0) or 0):.3f}</td><td>{float(s.get('stoi',0) or 0):.3f}</td></tr>")
            html.append("</table>")

            html.append("<h3>Lowest STOI</h3>")
            html.append("<table><tr><th>Ref</th><th>Cand</th><th>STOI</th><th>SNR (dB)</th></tr>")
            for s in worst_stoi:
                html.append(f"<tr><td>{s.get('ref','')}</td><td>{s.get('cand','')}</td><td>{float(s.get('stoi',0) or 0):.3f}</td><td>{float(s.get('snr',0) or 0):.3f}</td></tr>")
            html.append("</table>")

    # Raw JSON (always render original input for debugging)
    html.append("<h2>Raw Results</h2><pre>")
    html.append(Path(results_json_path).read_text(encoding="utf-8"))
    html.append("</pre>")

    html.append("</body></html>")
    Path(out_html_path).write_text("".join(html), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Path to results JSON")
    p.add_argument("--out", required=True, help="Path to output HTML")
    p.add_argument("--top_k", type=int, default=5, help="Top-k failures to show")
    args = p.parse_args()
    build_report(args.results, args.out, top_k=args.top_k)
