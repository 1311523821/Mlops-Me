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
