"""输入获取工具 — 从 arXiv/GitHub/网页/本地文件获取论文和代码。"""

import subprocess
import os
import re

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
    try:
        paper = next(client.results(search))
    except StopIteration:
        raise ValueError(f"arXiv 上未找到 ID: {arxiv_id}")
    except Exception as e:
        raise ConnectionError(f"arXiv API 请求失败 ({arxiv_id}): {e}")

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
        check=True, capture_output=True, timeout=120,
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
