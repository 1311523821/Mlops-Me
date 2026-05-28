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
    # 收集所有匹配，返回文本中出现最早的
    matches = []
    for pattern, name in loss_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            matches.append((m.start(), name, m))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    _, name, m = matches[0]
    start = max(0, m.start() - 100)
    end = min(len(text), m.end() + 200)
    return {"type": name, "context": text[start:end].strip()}


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
