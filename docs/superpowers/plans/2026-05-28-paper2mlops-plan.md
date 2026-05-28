# Paper2MLOps 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 paper2mlops Skill，通过 Agent 自动分析论文/代码并生成符合 MLOps 框架规范的 `tasks/<name>/` 代码。

**Architecture:** 平台无关的 Harness 工具层（`core/paper2mlops/`）提供所有分析、生成、验证能力，薄 SKILL.md 做 Claude Code 调度入口。纯 GitHub 输入时等价于 add-mlops-project 的进化版；有论文时增加 Paper Agent 并行分析。

**Tech Stack:** Python 3（arxiv, pdfplumber, ast, http.server）+ Claude Code SKILL.md（Agent/Bash/Write/Edit 工具）

---

## 文件结构

```
core/paper2mlops/           # Harness 工具层（平台无关 Python 包）
├── __init__.py             #   导出公共接口
├── harvester.py            #   输入获取（arxiv/git/web_fetch）
├── parser.py               #   论文解析（PDF 文本提取）
├── analyzer.py             #   代码分析（AST/依赖图/forward 追踪）
├── reports.py              #   报告生成（JSON Schema + HTML 模板）
├── server.py               #   本地 HTTP 服务器（报告交付 + 事件回传）
├── generator.py            #   代码生成（model.py/dataset.py/config.yaml）
└── validator.py            #   验证（导入链/前向传播/train --fast）

.claude/skills/paper2mlops/
└── SKILL.md                # Claude Code 入口（薄调度层）
```

---

### Task 1: 创建 `core/paper2mlops/` 包骨架

**Files:**
- Create: `core/paper2mlops/__init__.py`

- [ ] **Step 1: 写 `__init__.py`**

```python
"""Paper2MLOps Harness 工具层 — 平台无关的论文/代码分析工具包。

提供：
- harvester: 输入获取（arXiv 下载、git clone、网页抓取）
- parser: 论文解析（PDF 文本提取、架构段落定位）
- analyzer: 代码分析（AST 类签名提取、依赖图、forward 追踪）
- reports: 报告生成（JSON Schema + HTML 渲染）
- server: 本地 HTTP 服务器（报告交付 + 用户确认事件回传）
- generator: MLOps 任务代码生成（model.py/dataset.py/config.yaml）
- validator: 验证（导入链/前向传播/train --fast）
"""
```

- [ ] **Step 2: 验证导入**

```bash
python -c "import core.paper2mlops; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add core/paper2mlops/__init__.py
git commit -m "feat(paper2mlops): 创建 core/paper2mlops/ 包骨架"
```

---

### Task 2: `harvester.py` — 输入获取

**Files:**
- Create: `core/paper2mlops/harvester.py`

- [ ] **Step 1: 写 `harvester.py`**

```python
"""输入获取工具 — 从 arXiv/GitHub/网页/本地文件获取论文和代码。"""

import subprocess
import tempfile
import os
import re
from pathlib import Path

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False


def fetch_paper(source: str, output_dir: str) -> dict:
    """
    根据 source 类型下载论文，返回 {"type": "arxiv"|"pdf"|"web", "pdf_path": str, "text": str|None}
    """
    os.makedirs(output_dir, exist_ok=True)

    # arXiv 链接
    arxiv_id = _extract_arxiv_id(source)
    if arxiv_id:
        return _fetch_arxiv(arxiv_id, output_dir)

    # 本地 PDF 文件
    if source.endswith(".pdf") and os.path.isfile(source):
        return {"type": "pdf", "pdf_path": source, "text": None, "source_url": source}

    # HTTP(s) 链接（博客、论文页面等）
    if source.startswith("http"):
        return {"type": "web", "pdf_path": None, "text": None, "source_url": source}

    raise ValueError(f"无法识别的论文来源: {source}")


def fetch_code(source: str, output_dir: str) -> dict:
    """
    根据 source 获取代码仓库，返回 {"type": "github"|"local", "repo_path": str}
    """
    os.makedirs(output_dir, exist_ok=True)

    if source.startswith("http") and "github.com" in source:
        return _clone_repo(source, output_dir)

    if os.path.isdir(source):
        return {"type": "local", "repo_path": source, "source_url": source}

    raise ValueError(f"无法识别的代码来源: {source}")


def detect_inputs(user_args: list[str]) -> list[dict]:
    """
    从用户输入参数中识别所有资源。
    返回 [{"type": "paper"|"code", "value": str}, ...]
    """
    result = []
    for arg in user_args:
        if _looks_like_paper(arg):
            result.append({"type": "paper", "value": arg})
        elif _looks_like_code(arg):
            result.append({"type": "code", "value": arg})
        else:
            raise ValueError(f"无法分类的输入: {arg}")
    return result


def _extract_arxiv_id(url_or_id: str) -> str | None:
    """从 arXiv URL 或裸 ID 中提取 arXiv ID。"""
    patterns = [
        r"arxiv\.org/abs/([\w.-]+)",
        r"arxiv\.org/pdf/([\w.-]+)",
        r"ar5iv\.org/abs/([\w.-]+)",
    ]
    for p in patterns:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    if re.match(r"^[\w.-]+$", url_or_id) and "." in url_or_id:
        return url_or_id
    return None


def _fetch_arxiv(arxiv_id: str, output_dir: str) -> dict:
    """下载 arXiv 论文 PDF 和源码包。"""
    if not HAS_ARXIV:
        raise ImportError("需要安装 arxiv 包: pip install arxiv")

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    pdf_path = os.path.join(output_dir, f"{arxiv_id}.pdf")
    paper.download_pdf(dirpath=output_dir, filename=f"{arxiv_id}.pdf")

    source_tar = os.path.join(output_dir, f"{arxiv_id}.tar.gz")
    try:
        paper.download_source(dirpath=output_dir, filename=f"{arxiv_id}.tar.gz")
    except Exception:
        source_tar = None

    return {
        "type": "arxiv",
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "pdf_path": pdf_path,
        "source_tar": source_tar,
        "text": paper.summary,
        "source_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def _clone_repo(url: str, output_dir: str) -> dict:
    """浅克隆 GitHub 仓库到 output_dir。"""
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    target = os.path.join(output_dir, name)
    if os.path.exists(target):
        return {"type": "github", "repo_path": target, "source_url": url}
    subprocess.run(
        ["git", "clone", "--depth", "1", url, target],
        check=True, capture_output=True,
    )
    return {"type": "github", "repo_path": target, "source_url": url}


def _looks_like_paper(value: str) -> bool:
    """判断一个输入是否像论文资源。"""
    if value.endswith(".pdf"):
        return True
    if _extract_arxiv_id(value):
        return True
    if value.startswith("http") and "github.com" not in value:
        return True
    return False


def _looks_like_code(value: str) -> bool:
    """判断一个输入是否像代码资源。"""
    if "github.com" in value:
        return True
    if os.path.isdir(value):
        return True
    return False
```

- [ ] **Step 2: 测试 arXiv ID 提取**

```bash
python -c "
from core.paper2mlops.harvester import _extract_arxiv_id
assert _extract_arxiv_id('https://arxiv.org/abs/1512.03385') == '1512.03385'
assert _extract_arxiv_id('1512.03385') == '1512.03385'
assert _extract_arxiv_id('https://github.com/pytorch/vision') is None
print('PASS')
"
```

- [ ] **Step 3: 测试 detect_inputs**

```bash
python -c "
from core.paper2mlops.harvester import detect_inputs
r = detect_inputs(['https://arxiv.org/abs/1512.03385', 'https://github.com/pytorch/vision'])
assert len(r) == 2
assert r[0]['type'] == 'paper'
assert r[1]['type'] == 'code'
print('PASS')
"
```

- [ ] **Step 4: Commit**

```bash
git add core/paper2mlops/harvester.py
git commit -m "feat(paper2mlops): 添加 harvester.py — 输入获取和分类"
```

---

### Task 3: `parser.py` — 论文解析

**Files:**
- Create: `core/paper2mlops/parser.py`

- [ ] **Step 1: 写 `parser.py`**

```python
"""论文解析工具 — PDF 文本提取、架构段落定位、GitHub 链接发现。"""

import re
from pathlib import Path


def extract_text(pdf_path: str) -> str:
    """从 PDF 提取完整文本。使用 pdfplumber，fallback 到 PyPDF2。"""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        raise ImportError("需要安装 pdfplumber 或 PyPDF2: pip install pdfplumber")


def find_architecture_section(text: str) -> str:
    """
    定位论文中描述模型架构的段落。
    搜索 'architecture'、'network'、'model' 等关键词附近的章节。
    """
    sections = _split_into_sections(text)
    keywords = [
        "architecture", "network architecture", "model architecture",
        "proposed method", "method", "approach", "framework",
        "network design", "backbone", "encoder", "decoder",
    ]
    scored = []
    for heading, body in sections:
        score = sum(1 for kw in keywords if kw.lower() in (heading + body).lower())
        # 标题中匹配权重大于正文
        score += sum(3 for kw in keywords if kw.lower() in heading.lower())
        # 惩罚过短的节
        if len(body) < 100:
            score = 0
        scored.append((score, heading, body))
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][2]
    # fallback：返回前半部分（摘要+引言之后的第一个大段落）
    return "\n\n".join([body for _, body in sections[:3]])


def find_github_links(text: str) -> list[str]:
    """从文本中提取所有 GitHub 链接。"""
    pattern = r"https?://github\.com/[\w.-]+/[\w.-]+"
    urls = set(re.findall(pattern, text, re.IGNORECASE))
    return sorted(urls)


def find_loss_function(text: str) -> dict | None:
    """
    从论文文本中识别损失函数定义。
    返回 {"type": "CrossEntropyLoss", "context": "..."} 或 None。
    """
    loss_patterns = [
        (r"cross[-\s]?entropy\s*(?:loss)?", "CrossEntropyLoss"),
        (r"binary\s*cross[-\s]?entropy\s*(?:loss)?", "BCEWithLogitsLoss"),
        (r"dice\s*(?:loss|coefficient)", "DiceLoss"),
        (r"focal\s*(?:loss)", "FocalLoss"),
        (r"mean\s*squared\s*error|MSE", "MSELoss"),
        (r"L1\s*(?:loss|norm)|MAE", "L1Loss"),
        (r"CTC\s*(?:loss)?", "CTCLoss"),
        (r"KL\s*divergence", "KLDivLoss"),
    ]
    for pattern, name in loss_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 100)
            end = min(len(text), m.end() + 200)
            return {"type": name, "context": text[start:end].strip()}
    return None


def find_input_shape(text: str) -> dict | None:
    """
    从论文文本中推测输入形状。
    返回 {"shape": [1, 3, 224, 224], "source": "explicit"|"inferred"} 或 None。
    """
    # 显式声明："input size 224×224" / "224×224 RGB images"
    explicit = re.search(
        r"(\d+)\s*[×x]\s*(\d+)\s*(?:pixel|RGB|images?|input|resolution)",
        text, re.IGNORECASE
    )
    if explicit:
        h, w = int(explicit.group(1)), int(explicit.group(2))
        return {"shape": [1, 3, h, w], "source": "explicit"}

    # ImageNet 常见尺寸
    if "imagenet" in text.lower():
        if "224" in text:
            return {"shape": [1, 3, 224, 224], "source": "inferred"}
        if "299" in text:
            return {"shape": [1, 3, 299, 299], "source": "inferred"}

    return None


def find_preprocess_params(text: str) -> dict | None:
    """从论文文本中提取预处理参数（mean/std）。"""
    result = {}
    imagenet_mean = re.search(r"mean\s*[=:]\s*\[?\s*\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", text)
    if imagenet_mean:
        result["mean"] = [float(imagenet_mean.group(i)) for i in range(1, 4)]

    imagenet_std = re.search(r"std\s*[=:]\s*\[?\s*\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", text)
    if imagenet_std:
        result["std"] = [float(imagenet_std.group(i)) for i in range(1, 4)]

    return result if result else None


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """将论文文本按章节标题分割为 (标题, 正文) 列表。"""
    lines = text.split("\n")
    sections = []
    current_heading = "abstract"
    current_body = []

    section_pattern = re.compile(
        r"^(\d+\.?\d*\.?\s+[A-Z][^\n]{2,60})$|"
        r"^([IVX]+\.\s+[A-Z][^\n]{2,60})$|"
        r"^([A-Z][A-Z\s]{3,40})$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = section_pattern.match(line)
        if m and len(line) < 80:
            if current_body:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = line
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, "\n".join(current_body)))
    return sections
```

- [ ] **Step 2: 测试 GitHub 链接提取**

```bash
python -c "
from core.paper2mlops.parser import find_github_links
t = 'Code at https://github.com/author/repo and https://github.com/author/repo'
assert find_github_links(t) == ['https://github.com/author/repo']
print('PASS')
"
```

- [ ] **Step 3: 测试 loss 识别**

```bash
python -c "
from core.paper2mlops.parser import find_loss_function
r = find_loss_function('We use cross-entropy loss with label smoothing')
assert r['type'] == 'CrossEntropyLoss'
r2 = find_loss_function('optimized with Dice loss and binary cross-entropy')
assert r2['type'] == 'DiceLoss'  # Dice 先匹配
print('PASS')
"
```

- [ ] **Step 4: Commit**

```bash
git add core/paper2mlops/parser.py
git commit -m "feat(paper2mlops): 添加 parser.py — 论文 PDF 解析"
```

---

### Task 4: `analyzer.py` — 代码分析

**Files:**
- Create: `core/paper2mlops/analyzer.py`

- [ ] **Step 1: 写 `analyzer.py`**

```python
"""代码分析工具 — AST 类签名提取、依赖图、forward 追踪、硬编码检测。"""

import ast
import os
import re
from pathlib import Path


def find_model_files(repo_path: str) -> list[str]:
    """
    在仓库中定位可能的模型文件。
    返回绝对路径列表，按可能性排序（文件名含 model/net/network 的优先）。
    """
    candidates = []
    model_keywords = ["model", "net", "network", "backbone", "encoder", "decoder"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "checkpoints", "outputs", "data"}]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            lower = f.lower()
            score = sum(2 for kw in model_keywords if kw in lower)
            candidates.append((score, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def extract_init_params(file_path: str, class_name: str = None) -> dict:
    """
    从 Python 文件中提取指定类（或第一个 nn.Module 子类）的 __init__ 参数。
    返回 {"class_name": str, "params": {"param_name": default_value, ...}}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # 如果指定了类名，找匹配的；否则找第一个 nn.Module 子类
            if class_name and node.name == class_name:
                target_class = node
                break
            if not class_name and _is_module_subclass(node):
                target_class = node
                break

    if target_class is None:
        # fallback: 找第一个类
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                target_class = node
                break

    if target_class is None:
        return {"class_name": None, "params": {}}

    init_method = None
    for item in target_class.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_method = item
            break

    if init_method is None:
        return {"class_name": target_class.name, "params": {}}

    params = {}
    args = init_method.args
    # 跳过 self
    all_args = args.args[1:] if args.args and args.args[0].arg == "self" else args.args
    defaults = args.defaults
    # 默认值从后往前对齐
    num_no_default = len(all_args) - len(defaults)
    for i, arg in enumerate(all_args):
        if i >= num_no_default:
            d = defaults[i - num_no_default]
            try:
                val = ast.literal_eval(d)
            except (ValueError, SyntaxError):
                val = ast.unparse(d) if hasattr(ast, "unparse") else str(d)
            params[arg.arg] = val
        else:
            params[arg.arg] = None  # 无默认值的必填参数

    return {"class_name": target_class.name, "params": params}


def find_dataset_files(repo_path: str) -> list[str]:
    """在仓库中定位可能的数据集文件。"""
    candidates = []
    dataset_keywords = ["dataset", "data_loader", "dataloader", "data"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for f in files:
            if not f.endswith(".py"):
                continue
            full = os.path.join(root, f)
            lower = f.lower()
            score = sum(2 for kw in dataset_keywords if kw in lower)
            candidates.append((score, full))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in candidates]


def detect_dependencies(repo_path: str) -> dict:
    """
    分析仓库依赖。返回 {"pip": [...], "local": [...]}
    pip: requirements.txt 或 setup.py 中的包名
    local: 仓库内非 pip 的 .py 模块
    """
    result = {"pip": [], "local": []}

    # 尝试读取 requirements.txt
    req_path = os.path.join(repo_path, "requirements.txt")
    if os.path.isfile(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            result["pip"] = [
                line.strip().split("==")[0].split(">=")[0].strip()
                for line in f if line.strip() and not line.startswith("#")
            ]

    # 尝试从 setup.py 提取
    setup_path = os.path.join(repo_path, "setup.py")
    if os.path.isfile(setup_path):
        with open(setup_path, "r", encoding="utf-8") as f:
            content = f.read()
            found = re.findall(r"['\"]([\w-]+)[\"']\s*[,\)]", content)
            result["pip"].extend(found)

    # 收集所有本地 .py 文件作为潜在依赖
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "tests", "test"}]
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                rel = os.path.relpath(os.path.join(root, f), repo_path)
                result["local"].append(rel)

    result["pip"] = sorted(set(result["pip"]))
    return result


def detect_hardcoded_values(file_path: str) -> list[dict]:
    """
    检测文件中的潜在硬编码值（数字和字符串字面量）。
    返回 [{"value": ..., "line": int, "context": str}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    hardcoded = []
    suspicious_patterns = [
        (r"(?<!\w)\d{2,}(?!\w)", "数字常量"),  # 两位数以上
        (r"(?<=path\s*=\s*)[\"'][^\"']+[\"']", "路径字符串"),
        (r"(?<=size\s*=\s*)\d+", "尺寸参数"),
    ]

    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        if line_stripped.startswith("#") or line_stripped.startswith('"""'):
            continue
        for pattern, desc in suspicious_patterns:
            m = re.search(pattern, line_stripped)
            if m:
                hardcoded.append({
                    "value": m.group(),
                    "line": i,
                    "context": line_stripped[:80],
                    "type": desc,
                })

    return hardcoded


def _is_module_subclass(node: ast.ClassDef) -> bool:
    """判断 AST 类节点是否继承自 nn.Module。"""
    for base in node.bases:
        if isinstance(base, ast.Attribute):
            if base.attr == "Module" and _get_name(base.value) in ("nn", "torch.nn"):
                return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
        if isinstance(base, ast.Call):
            return False
    return False


def _get_name(node: ast.AST) -> str:
    """获取 AST 节点的字符串表示。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_get_name(node.value)}.{node.attr}"
    return ""
```

- [ ] **Step 2: 测试 extract_init_params**

```bash
python -c "
import tempfile, os
code = '''
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(512, num_classes)
'''
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp = f.name
from core.paper2mlops.analyzer import extract_init_params
r = extract_init_params(tmp)
os.unlink(tmp)
assert r['class_name'] == 'MyModel'
assert r['params'] == {'num_classes': 10, 'dropout': 0.5}
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add core/paper2mlops/analyzer.py
git commit -m "feat(paper2mlops): 添加 analyzer.py — 代码 AST 分析"
```

---

### Task 5: `reports.py` — 报告生成（JSON Schema + HTML 渲染）

**Files:**
- Create: `core/paper2mlops/reports.py`

- [ ] **Step 1: 写 `reports.py`**

```python
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
```

- [ ] **Step 2: 测试 build_report_json**

```bash
python -c "
from core.paper2mlops.reports import build_report_json, render_report_html
r = build_report_json(
    'test_model',
    {'source_url': 'https://arxiv.org/abs/1234.5678'},
    paper_analysis={'loss': {'type': 'DiceLoss'}, 'input_shape': {'shape': [1,3,256,256], 'source': 'explicit'}},
    code_analysis={'model': {'class_name': 'UNet', 'source_file': 'model.py', 'init_params': {'n_channels': 3}}, 'dependencies': {'pip': ['torch'], 'local_files': ['utils.py']}},
)
assert r['task_type'] == 'segmentation'
assert r['model']['class_name'] == 'UNet'
html = render_report_html(r)
assert 'UNet' in html
assert 'DiceLoss' in html
assert '/api/confirm' in html
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add core/paper2mlops/reports.py
git commit -m "feat(paper2mlops): 添加 reports.py — JSON 报告生成 + HTML 渲染"
```

---

### Task 6: `server.py` — 本地 HTTP 服务器

**Files:**
- Create: `core/paper2mlops/server.py`

- [ ] **Step 1: 写 `server.py`**

```python
"""本地 HTTP 服务器 — 交付 HTML 报告、接收用户确认事件。"""

import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ReportHandler(SimpleHTTPRequestHandler):
    """处理报告页面请求和确认事件回传。"""

    report_dir: str = ""
    events_path: str = ""

    def do_POST(self):
        if self.path == "/api/confirm":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            event = json.loads(body)
            self._write_event(event)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        # 默认返回最新报告
        if self.path == "/" or self.path == "/index.html":
            self._serve_latest_html()
        else:
            super().do_GET()

    def _serve_latest_html(self):
        """找到 report_dir 中最新的 HTML 文件并返回。"""
        html_files = sorted(
            Path(self.report_dir).glob("*.html"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if html_files:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_files[0].read_bytes())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>No report yet</h2>")

    def _write_event(self, event: dict):
        """将事件追加写入 events.jsonl。"""
        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_message(self, format, *args):
        pass  # 抑制日志，保持安静


def start_server(report_dir: str, state_dir: str, port: int = 65049) -> HTTPServer:
    """
    启动本地 HTTP 服务器。

    参数:
        report_dir: HTML 报告文件存放目录
        state_dir: 事件记录目录（events.jsonl）
        port: 监听端口，默认 65049

    返回:
        HTTPServer 实例
    """
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    ReportHandler.report_dir = report_dir
    ReportHandler.events_path = os.path.join(state_dir, "events.jsonl")

    server = HTTPServer(("127.0.0.1", port), ReportHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server


def check_confirmation(state_dir: str, timeout_seconds: int = 600) -> dict | None:
    """
    轮询检查用户是否已确认。
    读取 events.jsonl 中最新的确认事件。

    参数:
        state_dir: events.jsonl 所在目录
        timeout_seconds: 超时秒数，默认 10 分钟

    返回:
        确认事件 dict，或超时返回 None
    """
    import time
    events_path = os.path.join(state_dir, "events.jsonl")
    start = time.time()

    while time.time() - start < timeout_seconds:
        if os.path.isfile(events_path):
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event.get("action") == "confirm":
                        return event
                    if event.get("action") == "review_confirm":
                        return event
        time.sleep(2)

    return None


def clear_events(state_dir: str):
    """清空事件文件（进入下一阶段前调用）。"""
    events_path = os.path.join(state_dir, "events.jsonl")
    if os.path.isfile(events_path):
        os.remove(events_path)
```

- [ ] **Step 2: 测试服务器启动和事件回传**

```bash
python -c "
import tempfile, os, json, urllib.request
from core.paper2mlops.server import start_server, clear_events, check_confirmation

rd = tempfile.mkdtemp()
sd = tempfile.mkdtemp()

# 启动
server = start_server(rd, sd, port=16505)
print('Server started on 127.0.0.1:16505')

# 写入一个测试 HTML
with open(os.path.join(rd, 'test.html'), 'w') as f:
    f.write('<h1>Test Report</h1>')

# 验证页面可访问
import urllib.request
resp = urllib.request.urlopen('http://127.0.0.1:16505/')
assert b'Test Report' in resp.read()

# 模拟 POST 确认
data = json.dumps({'action': 'confirm', 'task_name': 'test'}).encode()
req = urllib.request.Request('http://127.0.0.1:16505/api/confirm', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
assert json.loads(resp.read())['status'] == 'ok'

# 检查事件
events_path = os.path.join(sd, 'events.jsonl')
assert os.path.isfile(events_path)
with open(events_path) as f:
    event = json.loads(f.readline())
    assert event['action'] == 'confirm'

server.shutdown()
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add core/paper2mlops/server.py
git commit -m "feat(paper2mlops): 添加 server.py — 本地 HTTP 服务器 + 事件回传"
```

---

### Task 7: `generator.py` — MLOps 任务代码生成

**Files:**
- Create: `core/paper2mlops/generator.py`

- [ ] **Step 1: 写 `generator.py`**

```python
"""代码生成器 — 根据分析报告生成 MLOps tasks/<name>/ 全部文件。"""

import os
import shutil


def generate_task(task_name: str, report: dict, output_base: str) -> list[str]:
    """
    根据分析报告生成全部任务文件。

    参数:
        task_name: 任务目录名
        report: 分析报告 JSON
        output_base: MLOps 项目根目录

    返回:
        生成的文件路径列表
    """
    task_dir = os.path.join(output_base, "tasks", task_name)
    os.makedirs(task_dir, exist_ok=True)

    files = []

    # __init__.py
    init_path = os.path.join(task_dir, "__init__.py")
    with open(init_path, "w", encoding="utf-8") as f:
        f.write(_render_init(task_name))
    files.append(init_path)

    # model.py
    model_path = os.path.join(task_dir, "model.py")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(_render_model(task_name, report))
    files.append(model_path)

    # dataset.py
    dataset_path = os.path.join(task_dir, "dataset.py")
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(_render_dataset(task_name, report))
    files.append(dataset_path)

    # config.yaml
    config_path = os.path.join(task_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(_render_config(task_name, report))
    files.append(config_path)

    return files


def copy_dependencies(report: dict, source_repo: str, task_name: str, output_base: str) -> list[str]:
    """将外部仓库的本地依赖文件复制到任务目录。"""
    task_dir = os.path.join(output_base, "tasks", task_name)
    copied = []
    for rel_path in report.get("dependencies", {}).get("local_files", []):
        src = os.path.join(source_repo, rel_path)
        dst = os.path.join(task_dir, os.path.basename(rel_path))
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def _render_init(task_name: str) -> str:
    return f'''"""任务 {task_name} — 由 Paper2MLOps 自动生成。"""

DEFAULT_DATASET = "{task_name}"
DEFAULT_MODEL = "{task_name}"
'''


def _render_model(task_name: str, report: dict) -> str:
    model = report.get("model", {})
    class_name = model.get("class_name", "UnknownModel")
    loss = report.get("loss", {})
    loss_type = loss.get("type", "CrossEntropyLoss")
    task_type = report.get("task_type", "classification")
    input_shape = model.get("input_shape", [1, 3, 224, 224])
    init_params = model.get("init_params", {})
    source_file = model.get("source_file", "")

    params_str = ", ".join(f"{k}={repr(v)}" for k, v in init_params.items())

    # 判断是否需要自定义 loss
    needs_custom_loss = task_type == "segmentation" or loss_type not in ("CrossEntropyLoss",)

    custom_loss_block = ""
    if needs_custom_loss:
        custom_loss_block = f"""
    def get_loss_fn(self):
        # 从原项目复制自定义 loss 实现
        # TODO: 如有本地 loss 文件，在此导入
        import torch.nn as nn
        return nn.{loss_type}()"""

    return f'''"""任务 {task_name} 的模型包装器 — 由 Paper2MLOps 自动生成。"""

import torch.nn as nn
from core.base_model import BaseModel
from core.registry import register_model


@register_model("{task_name}")
class {class_name}Wrapper(BaseModel):
    """包装自 {source_file} 的 {class_name} 模型。"""
    task_type = "{task_type}"

    def __init__(self, {params_str}):
        super().__init__()
        # 如原模型在本地文件中，取消下行注释并调整导入路径
        # import sys, os
        # _here = os.path.dirname(os.path.abspath(__file__))
        # if _here not in sys.path:
        #     sys.path.insert(0, _here)
        # from {os.path.splitext(os.path.basename(source_file))[0] if source_file else "model_file"} import {class_name}
        # self.net = {class_name}({", ".join(f"{k}={k}" for k in init_params)})
        self.net = nn.Identity()  # 占位，替换为实际模型

    def forward(self, x):
        return self.net(x)

    def get_example_input(self):
        import torch
        return torch.randn({input_shape}){custom_loss_block}

    @classmethod
    def from_config(cls, config):
        model_params = config.get("model_params", {{}})
        return cls(**model_params)
'''


def _render_dataset(task_name: str, report: dict) -> str:
    dataset = report.get("dataset", {})
    num_classes = dataset.get("num_classes", 10)
    class_names = dataset.get("class_names", [str(i) for i in range(num_classes)])
    preprocess = report.get("preprocess", {})

    return f'''"""任务 {task_name} 的数据集 — 由 Paper2MLOps 自动生成。"""

import os
from core.base_dataset import BaseDataset
from core.registry import register_dataset


@register_dataset("{task_name}")
class {task_name.capitalize()}Dataset(BaseDataset):
    """数据集包装器。"""

    CLASS_NAMES = {class_names}

    def __init__(self, data_dir, train=True, data_fraction=1.0):
        self.data_dir = data_dir
        self.train = train
        # TODO: 根据分析报告补充数据加载逻辑
        # 数据格式: {dataset.get('format', '未知')}
        self.samples = []

        if data_fraction < 1.0:
            n = max(1, int(len(self.samples) * data_fraction))
            self.samples = self.samples[:n]

    @property
    def num_classes(self):
        return {num_classes}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # TODO: 实现具体的数据加载逻辑
        raise NotImplementedError("需要根据实际数据格式补充 __getitem__ 实现")

    def get_preprocess_config(self):
        return {{
            "mean": {preprocess.get('mean', [0.5, 0.5, 0.5])},
            "std": {preprocess.get('std', [0.5, 0.5, 0.5])},
            "size": {preprocess.get('size', [224, 224])},
            "channels": {preprocess.get('channels', 3)},
            "classes": self.CLASS_NAMES,
        }}

    @classmethod
    def from_config(cls, config, split):
        return cls(
            data_dir=config["paths"]["data_dir"],
            train=(split == "train"),
            data_fraction=config.get("_data_fraction", 1.0),
        )
'''


def _render_config(task_name: str, report: dict) -> str:
    model = report.get("model", {})
    init_params = model.get("init_params", {})

    params_yaml = ""
    if init_params:
        params_yaml = "\nmodel_params:\n"
        for k, v in init_params.items():
            if isinstance(v, str):
                params_yaml += f'  {k}: "{v}"\n'
            else:
                params_yaml += f"  {k}: {v}\n"

    return f'''# 任务 {task_name} 配置 — 由 Paper2MLOps 自动生成
# 任务类型: {report.get("task_type", "classification")}

fast:
  model_name: "{task_name}"
  dataset_name: "{task_name}"
  epochs: 3
  data_fraction: 0.1

full:
  model_name: "{task_name}"
  dataset_name: "{task_name}"
  epochs: 50
{params_yaml}
'''
```

- [ ] **Step 2: 测试生成代码**

```bash
python -c "
import tempfile, os
from core.paper2mlops.generator import generate_task

report = {
    'task_type': 'classification',
    'model': {'class_name': 'ResNet', 'source_file': 'model.py', 'init_params': {'layers': [3,4,6,3], 'num_classes': 10}, 'input_shape': [1, 3, 224, 224]},
    'dataset': {'num_classes': 10},
    'loss': {'type': 'CrossEntropyLoss'},
    'preprocess': {'mean': [0.5,0.5,0.5], 'std': [0.5,0.5,0.5], 'size': [224,224], 'channels': 3},
    'dependencies': {'pip': [], 'local_files': []},
}
out = tempfile.mkdtemp()
files = generate_task('test_model', report, out)
for f in files:
    assert os.path.isfile(f), f'Missing: {f}'
    print(f'Generated: {f}')
print('PASS')
"
```

- [ ] **Step 3: Commit**

```bash
git add core/paper2mlops/generator.py
git commit -m "feat(paper2mlops): 添加 generator.py — MLOps 任务代码生成"
```

---

### Task 8: `validator.py` — 验证

**Files:**
- Create: `core/paper2mlops/validator.py`

- [ ] **Step 1: 写 `validator.py`**

```python
"""验证工具 — 导入链、前向传播、快速训练验证。"""

import subprocess
import sys
import os
from pathlib import Path


def validate_imports(project_root: str) -> dict:
    """
    验证导入链：检查 task 注册到 MODELS/DATASETS。
    返回 {"pass": bool, "model_name": str, "dataset_name": str, "error": str|None}
    """
    task_name = os.path.basename(os.getcwd())
    code = f'''
import sys
sys.path.insert(0, r"{project_root}")
from core.registry import MODELS, DATASETS
# 重新加载模块触发注册
import importlib
for mod in list(sys.modules.keys()):
    if "tasks." in mod:
        del sys.modules[mod]
try:
    importlib.import_module("tasks.{task_name}.model")
    importlib.import_module("tasks.{task_name}.dataset")
except Exception as e:
    print(f"IMPORT_ERROR: {{e}}")
    raise SystemExit(1)
print(f"MODELS: {{list(MODELS.keys())}}")
print(f"DATASETS: {{list(DATASETS.keys())}}")
'''
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=project_root,
    )
    if r.returncode != 0:
        return {"pass": False, "model_name": None, "dataset_name": None, "error": r.stderr or r.stdout}
    return {"pass": True, "model_name": None, "dataset_name": None, "error": None}


def validate_forward_pass(project_root: str, model_name: str) -> dict:
    """
    验证前向传播：创建模型 → get_example_input() → forward() → 检查输出 shape。
    返回 {"pass": bool, "input_shape": list, "output_shape": list, "error": str|None}
    """
    code = f'''
import sys
sys.path.insert(0, r"{project_root}")
import importlib
importlib.import_module("tasks.{model_name}.model")
from core.registry import MODELS
import torch

m = MODELS["{model_name}"]()
x = m.get_example_input()
print(f"INPUT: {{list(x.shape)}}")
with torch.no_grad():
    y = m(x)
print(f"OUTPUT: {{list(y.shape)}}")
print(f"TASK_TYPE: {{m.task_type}}")
'''
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=project_root,
    )
    if r.returncode != 0:
        return {"pass": False, "input_shape": None, "output_shape": None, "error": r.stderr}
    return {"pass": True, "input_shape": [], "output_shape": [], "error": None}


def validate_train_fast(project_root: str, task_name: str) -> dict:
    """
    验证快速训练：python train.py --task {task_name} --fast。
    返回 {"pass": bool, "stdout": str, "error": str|None}
    """
    r = subprocess.run(
        [sys.executable, "train.py", "--task", task_name, "--fast"],
        capture_output=True, text=True, cwd=project_root, timeout=300,
    )
    if r.returncode != 0:
        return {"pass": False, "stdout": r.stdout, "error": r.stderr}
    return {"pass": True, "stdout": r.stdout, "error": None}


def run_all_validations(project_root: str, task_name: str, model_name: str) -> dict:
    """依次运行全部验证步骤。"""
    results = {}

    results["import"] = validate_imports(project_root)
    if not results["import"]["pass"]:
        results["status"] = "failed"
        return results

    results["forward"] = validate_forward_pass(project_root, model_name)
    if not results["forward"]["pass"]:
        results["status"] = "failed"
        return results

    results["train_fast"] = validate_train_fast(project_root, task_name)
    results["status"] = "passed" if results["train_fast"]["pass"] else "partial"

    return results
```

- [ ] **Step 2: Commit**

```bash
git add core/paper2mlops/validator.py
git commit -m "feat(paper2mlops): 添加 validator.py — 三层验证"
```

---

### Task 9: `SKILL.md` — Claude Code 调度入口

**Files:**
- Create: `.claude/skills/paper2mlops/SKILL.md`

- [ ] **Step 1: 写 `SKILL.md`**

```markdown
---
name: paper2mlops
description: |
  把论文（arXiv）、GitHub 仓库、博客文章等任意 ML 资料，
  通过 Agent 自动分析并生成符合 MLOps 框架规范的 tasks/<name>/ 代码。
  当用户说 /paper2mlops、或提到把论文/仓库接入 MLOps 时触发。
---

# Paper2MLOps — 论文/代码自动接入 MLOps

## 核心原则

- **Agent = Model + Harness**：Model 通过 Claude API 调用，Harness 是 `core/paper2mlops/` 提供的工具层
- **全部本地化**：下载、分析、代码生成全在本地，HTTP 服务器仅绑定 127.0.0.1
- **半自动**：分析报告需用户确认后才生成代码；审查报告需用户确认后才完成
- **测试先行 + 审查**：代码生成前先写测试；生成后跑 Agent-to-Agent 审查
- **每次失败 → 工程化修复**：Agent 出错不是手动修，而是改进 Harness 约束规则

## 工具清单

以下 Python 工具通过 Bash 调用，均在 `core/paper2mlops/` 下：

| 工具 | 用途 | 调用示例 |
|------|------|---------|
| harvester.py | 输入获取和分类 | `python -c "from core.paper2mlops.harvester import ..."` |
| parser.py | 论文 PDF 解析 | `python -c "from core.paper2mlops.parser import ..."` |
| analyzer.py | 代码 AST 分析 | `python -c "from core.paper2mlops.analyzer import ..."` |
| reports.py | 报告生成和渲染 | `python -c "from core.paper2mlops.reports import ..."` |
| server.py | 本地 HTTP 服务器 | `python -c "from core.paper2mlops.server import ..."` |
| generator.py | 任务代码生成 | `python -c "from core.paper2mlops.generator import ..."` |
| validator.py | 验证测试 | `python -c "from core.paper2mlops.validator import ..."` |

## 四阶段工作流

### 阶段一：输入获取 & 分类

用户运行 `/paper2mlops <arg1> <arg2> ...`，Dispatcher 执行：

```bash
python -c "
from core.paper2mlops.harvester import detect_inputs
import json
result = detect_inputs(<用户的所有参数>)
print(json.dumps(result, ensure_ascii=False))
"
```

根据 `detect_inputs` 的结果判断：
- 有 paper → 需要 Paper Agent
- 有 code → 需要 Code Agent
- 两者都有 → 并行分发

**获取资料（按需）：**

论文下载：
```bash
python -c "
from core.paper2mlops.harvester import fetch_paper
import json
r = fetch_paper('<source>', '.superpowers/paper2mlops/<task_id>/raw')
print(json.dumps(r, ensure_ascii=False))
"
```

代码克隆：
```bash
python -c "
from core.paper2mlops.harvester import fetch_code
import json
r = fetch_code('<source>', '.superpowers/paper2mlops/<task_id>/raw')
print(json.dumps(r, ensure_ascii=False))
"
```

---

### 阶段二：多 Agent 并行分析

**如输入包含论文 → 启动 Paper Agent（Subagent）：**

```
Agent(
  subagent_type="general-purpose",
  description="分析论文",
  prompt="你是论文分析 Agent。分析以下论文，提取结构化信息。

论文 PDF 路径：{pdf_path}

请用 core/paper2mlops/parser.py 的工具函数提取：
1. 论文文本（extract_text）
2. 模型架构描述段落（find_architecture_section）
3. GitHub 链接（find_github_links）
4. Loss 函数定义（find_loss_function）
5. 输入形状（find_input_shape）
6. 预处理参数（find_preprocess_params）

输出一个 JSON 对象，包含以上全部字段。如有 GitHub 链接，标注出来供后续 Code Agent 使用。"
)
```

**如输入包含代码 → 启动 Code Agent（Subagent）：**

```
Agent(
  subagent_type="general-purpose",
  description="分析代码仓库",
  prompt="你是代码分析 Agent。分析以下仓库，提取结构化信息。

仓库路径：{repo_path}

请用 core/paper2mlops/analyzer.py 的工具函数分析：
1. find_model_files() — 定位模型文件
2. extract_init_params() — 提取构造函数参数
3. find_dataset_files() — 定位数据集文件
4. detect_dependencies() — 识别依赖
5. detect_hardcoded_values() — 检测硬编码

输出一个 JSON 对象，包含 model、dataset、dependencies 字段。"
)
```

**如果两个 Agent 都可用，并行启动它们（同一条消息中两个 Agent 工具调用）。**

---

### 阶段三：分析报告 + 人工确认

收集子 Agent 输出，合并生成报告：

```bash
python -c "
from core.paper2mlops.reports import build_report_json, render_report_html
import json

paper = json.loads('''<Paper Agent 的 JSON 输出>''') if '<有论文>' else None
code = json.loads('''<Code Agent 的 JSON 输出>''') if '<有代码>' else None

report = build_report_json(
    task_name='<推断的任务名>',
    source={'paper_url': '...', 'github_url': '...', 'input_type': 'paper+code'},
    paper_analysis=paper,
    code_analysis=code,
)
# 保存 JSON
with open('.superpowers/paper2mlops/<task_id>/analysis/report.json', 'w') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
# 渲染 HTML
html = render_report_html(report)
with open('.superpowers/paper2mlops/<task_id>/analysis/report.html', 'w') as f:
    f.write(html)
"
```

**启动 HTTP 服务器（如未运行）：**

```bash
python -c "
from core.paper2mlops.server import start_server
start_server(
    report_dir='.superpowers/paper2mlops/<task_id>/analysis',
    state_dir='.superpowers/paper2mlops/<task_id>/state',
)
print('报告已生成: http://localhost:65049')
"
```

**等待用户确认：**

```bash
python -c "
from core.paper2mlops.server import check_confirmation
import json
event = check_confirmation('.superpowers/paper2mlops/<task_id>/state', timeout_seconds=600)
print(json.dumps(event) if event else 'TIMEOUT')
"
```

如果用户修改了字段，更新 report.json 中的对应值后再继续。

---

### 阶段四：测试先行 → 代码生成 → 代码审查

**Step 1: 生成代码**

```bash
python -c "
from core.paper2mlops.generator import generate_task, copy_dependencies
import json

with open('.superpowers/paper2mlops/<task_id>/analysis/report.json') as f:
    report = json.load(f)

files = generate_task(report['task_name'], report, '.')
print('Generated:', files)
"
```

如有本地依赖文件，复制它们：
```bash
python -c "
from core.paper2mlops.generator import copy_dependencies
import json
with open('.superpowers/paper2mlops/<task_id>/analysis/report.json') as f:
    report = json.load(f)
copied = copy_dependencies(report, '<repo_path>', report['task_name'], '.')
print('Copied:', copied)
"
```

**Step 2: 运行验证**

```bash
python -c "
from core.paper2mlops.validator import run_all_validations
import json

result = run_all_validations('.', '<task_name>', '<model_name>')
print(json.dumps(result, ensure_ascii=False, indent=2))
"
```

如果验证失败：分析错误信息 → 修正代码 → 重新验证。最多循环 3 次。

**Step 3: 代码审查（确认点 2）**

启动 Reviewer Agent（Subagent），独立审查：

```
Agent(
  subagent_type="feature-dev:code-reviewer",
  description="审查新生成的 MLOps 任务",
  prompt="审查 tasks/<task_name>/ 下所有文件是否合规。按以下清单逐项检查，报告通过/不通过：

1. 低耦合：core/ 不包含具体任务名
2. 硬编码：参数来自 config 而非写死
3. 接口合规：继承 BaseModel/BaseDataset，实现必要方法
4. 安全隐患：无密钥泄露，gitignore 完整
5. 代码质量：简体中文注释，语义清晰命名
6. 测试验证：三层测试通过"
)
```

审查结果渲染为 HTML 审查报告：
```bash
python -c "
from core.paper2mlops.reports import render_review_html
review = <审查结果 JSON>
html = render_review_html(review)
with open('.superpowers/paper2mlops/<task_id>/analysis/review.html', 'w') as f:
    f.write(html)
"
```

等待用户确认审查结果。如有 ❌ 项，自动修复后重新审查。

**Step 4: 快速训练验证（如有数据）**

```bash
python train.py --task <task_name> --fast
```

**Step 5: 最终结果确认（确认点 3）**

展示生成的文件清单和验证结果。用户确认后完成。

---

## 错误处理

- 下载失败 → 提示用户检查链接，或手动提供文件
- Agent 信心度 low → HTML 报告中红色高亮，要求用户手动填写
- 测试不通过 → 分析错误信息 → 自动修正 → 最多 3 次循环 → 停止并报告
- 审查不通过 → 自动修复 → 最多 3 次循环 → 停止并展示问题清单

## 持久化路径

所有中间产物写入 `.superpowers/paper2mlops/{task_id}/`：
- `raw/` — 下载的论文 PDF、克隆的仓库
- `analysis/` — 分析报告 JSON + HTML
- `state/` — 用户确认事件
- `review.md` — 代码审查报告
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/paper2mlops/SKILL.md
git commit -m "feat(paper2mlops): 添加 SKILL.md — Claude Code 调度入口"
```

---

### Task 10: 端到端集成验证

**Files:**
- Create: `tests/test_paper2mlops.py`

- [ ] **Step 1: 写端到端测试**

```python
"""Paper2MLOps 端到端集成测试。"""

import os
import sys
import tempfile
import json


def test_harvester_detect_inputs():
    """测试输入分类。"""
    from core.paper2mlops.harvester import detect_inputs

    r = detect_inputs([
        "https://arxiv.org/abs/1512.03385",
        "https://github.com/pytorch/vision",
    ])
    assert len(r) == 2
    assert r[0]["type"] == "paper"
    assert r[1]["type"] == "code"


def test_parser_loss_detection():
    """测试论文 loss 函数识别。"""
    from core.paper2mlops.parser import find_loss_function

    assert find_loss_function("We use cross-entropy loss")["type"] == "CrossEntropyLoss"
    assert find_loss_function("with Dice loss and focal loss")["type"] == "DiceLoss"


def test_analyzer_init_params():
    """测试 AST 参数提取。"""
    from core.paper2mlops.analyzer import extract_init_params
    import tempfile

    code = """
import torch.nn as nn
class TestModel(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.2, use_bn=True):
        super().__init__()
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        r = extract_init_params(tmp)
        assert r["class_name"] == "TestModel"
        assert r["params"]["num_classes"] == 1000
        assert r["params"]["dropout"] == 0.2
        assert r["params"]["use_bn"] is True
    finally:
        os.unlink(tmp)


def test_reports_build_json():
    """测试报告 JSON 生成。"""
    from core.paper2mlops.reports import build_report_json

    r = build_report_json(
        "test",
        {"source_url": "http://example.com"},
        paper_analysis={"loss": {"type": "CrossEntropyLoss"}},
        code_analysis={"model": {"class_name": "CNN", "init_params": {}}},
    )
    assert r["task_name"] == "test"
    assert r["loss"]["type"] == "CrossEntropyLoss"


def test_reports_render_html():
    """测试 HTML 渲染。"""
    from core.paper2mlops.reports import build_report_json, render_report_html

    r = build_report_json("test", {"source_url": "http://example.com"}, None, None)
    html = render_report_html(r)
    assert "/api/confirm" in html
    assert "test" in html


def test_server_start_and_event():
    """测试 HTTP 服务器启动和事件回传。"""
    from core.paper2mlops.server import start_server, clear_events
    import urllib.request

    rd = tempfile.mkdtemp()
    sd = tempfile.mkdtemp()

    server = start_server(rd, sd, port=16506)

    # 写入测试页面
    with open(os.path.join(rd, "test.html"), "w") as f:
        f.write("<h1>Test</h1>")

    resp = urllib.request.urlopen("http://127.0.0.1:16506/")
    assert b"Test" in resp.read()

    # 模拟确认
    data = json.dumps({"action": "confirm", "task_name": "x"}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:16506/api/confirm",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

    events_path = os.path.join(sd, "events.jsonl")
    assert os.path.isfile(events_path)

    server.shutdown()


def test_generator_create_files():
    """测试代码生成创建所有文件。"""
    from core.paper2mlops.generator import generate_task

    report = {
        "task_type": "classification",
        "model": {"class_name": "Test", "init_params": {}, "input_shape": [1, 3, 32, 32]},
        "dataset": {"num_classes": 10},
        "loss": {"type": "CrossEntropyLoss"},
        "preprocess": {"mean": [0.5], "std": [0.5], "size": [32, 32], "channels": 3},
        "dependencies": {"pip": [], "local_files": []},
    }
    out = tempfile.mkdtemp()
    files = generate_task("test_gen", report, out)

    assert len(files) == 4  # __init__.py, model.py, dataset.py, config.yaml
    for f in files:
        assert os.path.isfile(f), f"Missing: {f}"

    # 验证 model.py 内容
    model_path = os.path.join(out, "tasks", "test_gen", "model.py")
    with open(model_path) as f:
        content = f.read()
    assert "@register_model" in content
    assert "BaseModel" in content
    assert "get_example_input" in content
    print("All generator assertions passed")


def test_full_pipeline_no_paper():
    """端到端：纯代码输入（无论文）。"""
    from core.paper2mlops.harvester import detect_inputs
    from core.paper2mlops.reports import build_report_json, render_report_html

    # 模拟纯 GitHub 输入
    inputs = detect_inputs(["https://github.com/pytorch/vision"])
    assert len(inputs) == 1
    assert inputs[0]["type"] == "code"

    # 模拟代码分析结果
    code_analysis = {
        "model": {"class_name": "ResNet", "source_file": "resnet.py", "init_params": {"num_classes": 1000}},
        "dataset": {"num_classes": 1000},
        "dependencies": {"pip": ["torchvision"], "local_files": ["resnet.py"]},
    }

    report = build_report_json("resnet_test", {"github_url": "..."}, None, code_analysis)
    assert report["task_type"] == "classification"
    assert report["model"]["class_name"] == "ResNet"

    html = render_report_html(report)
    assert "ResNet" in html
    assert "/api/confirm" in html
    print("Full pipeline (no paper) test passed")
```

- [ ] **Step 2: 运行全部测试**

```bash
python -m pytest tests/test_paper2mlops.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_paper2mlops.py
git commit -m "test(paper2mlops): 添加端到端集成测试"
```

---

### Task 11: 归档 `add-mlops-project` + 更新 CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`
- 归档: `.claude/skills/add-mlops-project/SKILL.md` → `.claude/skills/add-mlops-project/SKILL.md.archived`

- [ ] **Step 1: 归档旧 Skill**

```bash
mv .claude/skills/add-mlops-project/SKILL.md .claude/skills/add-mlops-project/SKILL.md.archived
```

- [ ] **Step 2: 更新 CLAUDE.md**

在 CLAUDE.md 的 "命令" 段末尾添加：

```markdown
- `/paper2mlops <url1> <url2> ...` — 论文/代码自动接入 MLOps（替代旧的 add-mlops-project）
```

同时把 CLAUDE.md 中所有 `add-mlops-project` 的引用替换为 `paper2mlops`。

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/add-mlops-project/SKILL.md.archived CLAUDE.md
git commit -m "feat(paper2mlops): 归档 add-mlops-project，更新 CLAUDE.md 指向 paper2mlops"
```

---

## 依赖安装

```bash
pip install arxiv pdfplumber
```
