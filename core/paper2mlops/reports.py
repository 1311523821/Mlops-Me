"""报告生成 — JSON Schema 定义 + HTML 渲染模板。"""

import json
import os
from datetime import datetime


# ── JSON Schema ────────────────────────────────────────────

def build_report_json(
    task_name: str,
    source: dict,
    paper_analysis: dict | None,
    code_analysis: dict | None,
) -> dict:
    """合并 Paper Agent 和 Code Agent 的输出，生成结构化分析报告。"""
    report = {
        "task_name": task_name,
        "source": source,
        "task_type": "classification",
        "model": {"class_name": None, "source_file": None, "init_params": {}, "input_shape": None, "output_shape": None},
        "dataset": {"format": None, "num_classes": None, "class_names": None},
        "loss": {"type": "CrossEntropyLoss", "params": {}},
        "preprocess": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "size": [224, 224], "channels": 3},
        "dependencies": {"pip": [], "local_files": []},
        "confidence": {},
    }

    # 合并论文分析
    if paper_analysis:
        if paper_analysis.get("loss"):
            report["loss"] = paper_analysis["loss"]
        if paper_analysis.get("input_shape"):
            report["model"]["input_shape"] = paper_analysis["input_shape"]["shape"]
            report["confidence"]["input_shape"] = paper_analysis["input_shape"]["source"]
        if paper_analysis.get("preprocess"):
            report["preprocess"].update(paper_analysis["preprocess"])
        if paper_analysis.get("task_type"):
            report["task_type"] = paper_analysis["task_type"]
        report["confidence"]["loss_type"] = "high" if paper_analysis.get("loss") else "low"

    # 合并代码分析
    if code_analysis:
        if code_analysis.get("model"):
            report["model"].update(code_analysis["model"])
            report["confidence"]["model_class"] = "high"
        if code_analysis.get("dataset"):
            report["dataset"].update(code_analysis["dataset"])
        if code_analysis.get("dependencies"):
            report["dependencies"] = code_analysis["dependencies"]
        _infer_task_type_from_code(report, code_analysis)

    # 补充默认置信度
    for key in ["task_type", "input_shape", "output_shape", "preprocess", "num_classes"]:
        if key not in report["confidence"]:
            report["confidence"][key] = "medium" if report.get(key) else "low"

    return report


def _infer_task_type_from_code(report: dict, code_analysis: dict) -> None:
    """从代码分析结果推断任务类型。"""
    loss_type = report.get("loss", {}).get("type", "")
    if loss_type in ("BCEWithLogitsLoss", "DiceLoss", "FocalLoss"):
        report["task_type"] = "segmentation"
        report["confidence"]["task_type"] = "high"
    elif loss_type in ("CTCLoss",):
        report["task_type"] = "speech_recognition"
        report["confidence"]["task_type"] = "medium"


# ── HTML 渲染 ──────────────────────────────────────────────

def render_report_html(report: dict) -> str:
    """将分析报告 JSON 渲染为可交互的 HTML 确认页面。"""
    confidence_badge = {"high": "🟢 高", "medium": "🟡 中", "low": "🔴 低"}
    confidence_color = {"high": "#27ae60", "medium": "#f39c12", "low": "#e74c3c"}

    model = report.get("model", {})
    preprocess = report.get("preprocess", {})
    deps = report.get("dependencies", {})
    conf = report.get("confidence", {})

    # 需要用户特别注意的项（中/低可信度）
    warnings = []
    for key, level in conf.items():
        if level in ("medium", "low"):
            label = {
                "task_type": "任务类型",
                "input_shape": "输入形状",
                "output_shape": "输出形状",
                "model_class": "模型类名",
                "loss_type": "Loss 函数",
                "num_classes": "类别数",
                "preprocess": "预处理参数",
            }.get(key, key)
            warnings.append(f"<b>{label}</b> <span style='color:{confidence_color.get(level)}'>{confidence_badge.get(level)}</span> — Agent 不太确定，请确认")

    warning_html = ""
    if warnings:
        warning_html = f"""
        <div style="background:#fff3cd;padding:12px;border-radius:6px;margin-bottom:16px;font-size:13px">
            <b>⚠️ 需确认的项：</b><br>
            {"<br>".join(f"&nbsp;&nbsp;• {w}" for w in warnings)}
        </div>"""

    pip_list = ", ".join(f"<code>{p}</code>" for p in deps.get("pip", [])) or "无"
    local_list = ", ".join(deps.get("local_files", [])) or "无"

    html = f"""<h2>📋 分析报告：{report['task_name']}</h2>
<p class="subtitle">来源：{" + ".join(f"<a href='{v}'>{k}</a>" for k, v in report.get("source", {}).items() if v)} | 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div style="background:#fff;border:1px solid #ddd;border-radius:8px;padding:20px;font-size:13px;line-height:1.8">
{warning_html}
<table style="width:100%;border-collapse:collapse;margin-bottom:16px">
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9;width:130px"><b>任务类型</b></td><td style="padding:8px;border:1px solid #eee">{report['task_type']} <span style="font-size:11px;color:{confidence_color.get(conf.get('task_type', 'medium'))}">{confidence_badge.get(conf.get('task_type', 'medium'))}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>模型类名</b></td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{model.get('class_name', '未知')}</span> <span style="font-size:11px">{confidence_badge.get(conf.get('model_class', 'medium'))}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>源文件</b></td><td style="padding:8px;border:1px solid #eee"><code>{model.get('source_file', '未知')}</code></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>构造函数参数</b></td><td style="padding:8px;border:1px solid #eee"><code contenteditable="true">{json.dumps(model.get('init_params', {}), ensure_ascii=False)}</code></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>输入形状</b></td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{model.get('input_shape', '未知')}</span> <span style="font-size:11px;color:{confidence_color.get(conf.get('input_shape', 'medium'))}">{confidence_badge.get(conf.get('input_shape', 'medium'))}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>输出形状</b></td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{model.get('output_shape', '未知')}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>Loss 函数</b></td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{report.get('loss', {}).get('type', 'CrossEntropyLoss')}</span> <span style="font-size:11px">{confidence_badge.get(conf.get('loss_type', 'medium'))}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9"><b>类别数</b></td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{report.get('dataset', {}).get('num_classes', '未知')}</span></td></tr>
</table>

<b>预处理参数</b>
<table style="width:100%;border-collapse:collapse;margin-top:8px;margin-bottom:16px">
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9;width:130px">mean</td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{preprocess.get('mean')}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9">std</td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{preprocess.get('std')}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9">resize</td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{preprocess.get('size')}</span></td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9">channels</td><td style="padding:8px;border:1px solid #eee"><span contenteditable="true">{preprocess.get('channels')}</span></td></tr>
</table>

<b>依赖</b>
<table style="width:100%;border-collapse:collapse;margin-top:8px;margin-bottom:16px">
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9;width:130px">pip 包</td><td style="padding:8px;border:1px solid #eee">{pip_list}</td></tr>
<tr><td style="padding:8px;border:1px solid #eee;background:#f9f9f9">本地文件</td><td style="padding:8px;border:1px solid #eee">{local_list}</td></tr>
</table>

<div style="margin-top:16px;display:flex;gap:8px">
  <button class="mock-button" style="background:#27ae60;color:#fff;border:none;padding:10px 20px;font-size:14px;cursor:pointer" onclick="confirmReport('{report['task_name']}')">✓ 确认，开始生成代码</button>
  <button class="mock-button" style="padding:10px 20px;font-size:14px;cursor:pointer" onclick="confirmReport('{report['task_name']}', true)">✎ 修改后确认</button>
</div>
</div>

<script>
const reportData = {json.dumps(report, ensure_ascii=False)};

function confirmReport(taskName, modified) {{
    const edits = {{}};
    if (modified) {{
        document.querySelectorAll('[contenteditable="true"]').forEach(el => {{
            edits[el.parentElement.previousElementSibling?.textContent || el.closest('tr').querySelector('td').textContent] = el.textContent;
        }});
    }}
    fetch('/api/confirm', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{action: 'confirm', task_name: taskName, modified: !!modified, edits: edits, timestamp: Date.now()}})
    }}).then(r => r.json()).then(data => {{
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;min-height:60vh"><h2>✓ 已确认，Agent 正在生成代码...</h2></div>';
    }});
}}
</script>"""
    return html


def render_review_html(review_result: dict) -> str:
    """将代码审查结果渲染为 HTML 页面。"""
    items_html = ""
    all_pass = True
    for item in review_result.get("checks", []):
        icon = "✅" if item["status"] == "pass" else "❌"
        if item["status"] != "pass":
            all_pass = False
        items_html += f"""
        <tr>
            <td style="padding:8px;border:1px solid #eee;text-align:center">{icon}</td>
            <td style="padding:8px;border:1px solid #eee"><b>{item['name']}</b></td>
            <td style="padding:8px;border:1px solid #eee;font-size:12px">{item.get('detail', '')}</td>
        </tr>"""

    status_badge = '<span style="background:#27ae60;color:#fff;padding:4px 10px;border-radius:12px">全部通过</span>' if all_pass else '<span style="background:#e74c3c;color:#fff;padding:4px 10px;border-radius:12px">存在问题</span>'

    return f"""<h2>🔍 代码审查报告：{review_result.get('task_name', '')}</h2>
<p class="subtitle">六项清单独立检查 {status_badge}</p>
<div style="background:#fff;border:1px solid #ddd;border-radius:8px;padding:20px">
<table style="width:100%;border-collapse:collapse">{items_html}</table>
<div style="margin-top:16px;display:flex;gap:8px">
  <button class="mock-button" style="background:#27ae60;color:#fff;border:none;padding:10px 20px;cursor:pointer" onclick="fetch('/api/confirm', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'review_confirm',approved:true,timestamp:Date.now()}})}})">✓ 确认通过</button>
  <button class="mock-button" style="padding:10px 20px;cursor:pointer" onclick="fetch('/api/confirm', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'review_confirm',approved:false,timestamp:Date.now()}})}})">↻ 自动修复后重新审查</button>
</div>
</div>"""
