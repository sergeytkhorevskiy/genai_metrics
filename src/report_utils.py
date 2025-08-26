
#!/usr/bin/env python3
"""
Report utilities: generate HTML reports with charts.
"""

import json, pathlib
import matplotlib.pyplot as plt
import base64, io

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{b64}'/>"

def generate_html_report(results: dict, out_path: str):
    """Generate HTML report with summary and simple charts."""
    html = ["<html><head><meta charset='utf-8'><title>GenAI Eval Report</title></head><body>"]
    html.append("<h1>GenAI Evaluation Report</h1>")
    html.append("<pre>" + json.dumps(results, indent=2) + "</pre>")

    # Example: bar chart for text scores
    if "text" in results:
        vals = results["text"]
        keys = ["bleu","sacrebleu","rouge1_f","rouge2_f","rougeL_f"]
        fig, ax = plt.subplots()
        ax.bar(keys, [vals.get(k,0) for k in keys])
        ax.set_ylim(0,1)
        ax.set_title("Text metrics")
        html.append(fig_to_base64(fig))

    # Example: audio snr/stoi
    if "audio" in results:
        vals = results["audio"]
        keys = ["snr_mean","stoi_mean"]
        fig, ax = plt.subplots()
        ax.bar(keys, [vals.get(k,0) for k in keys])
        ax.set_title("Audio metrics")
        html.append(fig_to_base64(fig))

    html.append("</body></html>")
    pathlib.Path(out_path).write_text("\n".join(html), encoding="utf-8")
