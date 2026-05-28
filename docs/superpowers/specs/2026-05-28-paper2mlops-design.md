# Paper2MLOps — 论文/代码自动接入 MLOps 工作流

## 概述

把论文（arXiv）、GitHub 仓库、博客文章等任意形式的 ML 资料，通过 Agent 自动分析并生成符合 MLOps 框架规范的 `tasks/<name>/` 代码。

**核心理念**：`Agent = Model + Harness`。Model 通过 API 调用（Claude Code / Codex / OpenClaw 等），Harness 提供让它可靠运行的完整环境。

**关键约束**：全部本地化。Harness 所有组件（PDF 解析、代码分析、HTTP 服务器、中间文件存储）均在本地运行。Model API 密钥和论文/代码内容不外泄。

### 与 `add-mlops-project` 的关系

`paper2mlops` 是 `add-mlops-project` 的**替代和进化**。原 Skill 的核心能力（代码分析、wrapper 模式、依赖处理）全部吸收进 `paper2mlops` 的 Code Agent。区别：

| | add-mlops-project（旧） | paper2mlops（新） |
|---|---|---|
| 输入 | GitHub / 本地代码 | GitHub / 论文 / 博客 / 任意组合 |
| 分析方式 | 单 Agent 顺序分析 | 多 Agent 并行分析 |
| 报告 | Markdown | HTML + JSON（在线可修改） |
| 代码生成 | 直接生成 | 测试先行 → 生成 → 审查 |
| 审查 | 手动 | Agent-to-Agent 自动审查 |

纯 GitHub 输入、无论文时，`paper2mlops` 跳过 Paper Agent，流程等价于原 `add-mlops-project`，但享有更好的报告界面和自动化测试/审查。

`add-mlops-project` Skill 在 `paper2mlops` 上线后归档，不再维护。

---

## 总体架构：四阶段流水线

```
输入获取&分类 → 多Agent并行分析 → 分析报告+人工确认 → 测试→生成→审查
```

### 阶段一：输入获取 & 分类

Harness 层统一处理多种输入形式：

| 输入类型 | 获取方式 |
|---------|---------|
| arXiv 论文 | `arxiv` 库下载 PDF + 源码包 |
| GitHub 仓库 | `git clone --depth 1` |
| 博客/网页 | `web_fetch` + browser-harness |
| 本地 PDF | 直接读取（pdfplumber） |

分类器判断输入组合，决定调度策略：
- 纯论文 → 仅 Paper Agent
- 纯代码 → 仅 Code Agent
- 论文+代码 → 两个 Agent 并行

### 阶段二：多 Agent 并行分析

Dispatcher（主控 Agent）只调度，不执行具体分析：

- **Paper Agent**：提取模型架构描述、输入/输出形状、loss 函数、预处理参数、数据格式、GitHub 链接
- **Code Agent**：定位模型文件、提取 `__init__` 参数、追踪 `forward()` 流程、定位 Dataset、识别依赖、检测硬编码

两个 Agent 上下文隔离，可并行运行。

### 阶段三：分析报告 + 人工确认（⏸️ 确认点 1）

合并分析结果，生成双格式报告：

- **JSON**（Agent 间通信）：`analysis/report.json`，结构化数据，Generator Agent 直接读取
- **HTML**（用户确认界面）：`analysis/report.html`，表格 + 信心度色标（🟢高 🟡中 🔴低），用户可在线修改字段

用户确认后进入代码生成。

### 阶段四：测试先行 → 代码生成 → 代码审查

**三步闸门，缺一不可：**

1. **先写测试**：基于分析报告生成三层测试（导入链 → 前向传播 → 完整流程）
2. **生成代码**：按 `add-mlops-project` 规范生成 `tasks/<name>/` 全部文件
3. **Agent-to-Agent 审查**（⏸️ 确认点 2）：Subagent 独立检查六项清单，有 ❌ 项自动修复后重新审查

快速训练验证 → 最终结果确认（⏸️ 确认点 3）。

---

## Harness 层设计

### 四大护栏

| 护栏 | 目标 | 本项目实现 |
|------|------|-----------|
| 上下文工程 | Agent 知道该看什么 | AGENTS.md 活文档 + 按需检索 + 失败案例写入 |
| 架构约束 | Agent 不破坏规范 | BaseModel/BaseDataset 强制接口 + 硬编码检测 |
| 反馈循环 | Agent 知道自己错了 | Agent-to-Agent 审查 + 测试先行 + 失败循环回生成 Agent |
| 熵管理 | 技术债务不积累 | 临时文件自动清理 + MLflow 版本追踪 |

### 工具矩阵（Model 可调用的能力）

| 类别 | 工具 | 实现 |
|------|------|------|
| 输入获取 | arXiv 下载、Git clone、Web fetch、PDF 解析 | `arxiv` / `git` / `web_fetch` / `pdfplumber` |
| 代码分析 | AST 解析、依赖图、forward 追踪、硬编码检测 | `ast` / `grep` / `LSP` |
| 论文解析 | 文本提取、架构段落定位、GitHub 链接发现 | `pdfplumber` + LLM vision + regex |
| 代码生成 | model.py / dataset.py / config.yaml + 依赖复制 | `Write` / `Edit` / `cp` |
| 验证 | 导入链、前向传播、train --fast | `Bash` python |
| 审查 | Subagent 六项清单独立检查 | `Agent(code-reviewer)` |

## 产品形态 & 演进路线

### MVP：Claude Code Skill（第一阶段）

以 `/paper2mlops` 命令触发，全流程在 Claude Code 内执行，浏览器仅用于 HTML 报告审阅。

### 之后：独立 Web 应用（第二阶段）

本项目的 `deploy/` 下新增 Web 界面。由于脱离 Claude Code 后没有现成的 Agent 运行时，需要在 Web 后端实现自己的 Agent 调度器（调用 LLM API），但**复用同一套 Harness 工具层**。

Web 应用形态：
- 前端：粘贴链接 → 查看报告 → 在线修改 → 确认
- 后端：Agent 调度（Model API 调用 + Harness 工具执行）
- 属于本项目的一部分，不是独立项目

### 跨平台兼容设计

Skill 不应该只能在 Claude Code 使用。核心策略：**Harness 工具层平台无关，Skill 只是薄调用入口**。

```
┌──────────────────────────────────────────────┐
│  平台入口层（平台相关，薄）                      │
│  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │Claude Code│  │  Codex   │  │  Web App    │ │
│  │SKILL.md  │  │ skill    │  │ (第二阶段)   │ │
│  └────┬─────┘  └────┬─────┘  └──────┬──────┘ │
├───────┼──────────────┼───────────────┼────────┤
│  调度层（通用）                                  │
│  ┌──────────────────────────────────────────┐ │
│  │         Dispatcher Agent                 │ │
│  │  (相同的调度逻辑，无论哪个平台触发)         │ │
│  └────────────────────┬─────────────────────┘ │
├───────────────────────┼────────────────────────┤
│  Harness 工具层（平台无关 Python 包）             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │
│  │PDF解析 │ │代码分析│ │ HTTP   │ │代码生成│ │
│  │arxiv   │ │AST/grep│ │ Server │ │模板渲染│ │
│  └────────┘ └────────┘ └────────┘ └────────┘ │
└──────────────────────────────────────────────┘
```

**平台差异仅在于**：
- Claude Code：用 `Skill` 工具触发，Agent 工具对应 Claude Code 的工具（Bash/Write/Edit/Agent）
- Codex：用 Codex skill 触发，工具映射到 Codex 等价能力
- Web App：用 HTTP endpoint 触发，工具映射到 Python 函数调用

**Harness 核心不变**：`core/paper2mlops/` 作为项目内的 Python 包，提供所有分析、生成、验证逻辑，各平台共用。

### 关键设计决策

- **Model 可替换**：Harness 不绑定特定模型，通过 API 调用。用户选模型，Harness 提供运行环境
- **平台可替换**：Harness 不绑定 Claude Code。同一套 Python 工具，Claude Code / Codex / Web App 三种入口
- **上下文按需注入**：不把整篇论文+全部代码塞进 prompt。第一步只给摘要和文件清单，需要时再深入
- **每次失败 → 工程化修复**（Mitchell Hashimoto 原则）：Agent 生成的代码有问题 → 改进 Harness 约束规则，不让同类错误再发生
- **全部本地化**：Harness 所有组件本地运行，HTTP 服务器仅绑定 `127.0.0.1`，中间产物落盘不外传

---

## Agent 协作架构

### 角色定义

```
Dispatcher（主控）
  ├── Paper Agent     → 分析论文 → 输出 analysis_paper.json
  ├── Code Agent      → 分析代码 → 输出 analysis_code.json
  ├── Generator Agent → 生成 tasks/<name>/ 全部文件
  └── Reviewer Agent  → 独立审查，输出 review.md
```

### 通信协议

- **统一输入格式**：`{ task_id, input_path, input_type, context }`
- **统一输出格式**：结构化 JSON，包含 `task_type`, `model`, `dataset`, `loss`, `preprocess`, `dependencies`
- **持久化路径**：`.superpowers/paper2mlops/{task_id}/`

### 并行 / 串行自适应

- 论文 + 代码 → Paper Agent 和 Code Agent 并行
- 纯论文 / 纯代码 → 串行
- Generator → Reviewer 始终串行（依赖前一步输出）

---

## 分析报告 Schema

### JSON 格式（Agent 间通信）

```json
{
  "task_name": "自动推断，用户可改",
  "source": { "paper_url": "...", "github_url": "...", "input_type": "paper+code" },
  "task_type": "classification | segmentation | detection | ...",
  "model": {
    "class_name": "...", "source_file": "...",
    "init_params": { "...": "..." },
    "input_shape": [1, 3, 224, 224], "output_shape": [1, 1000]
  },
  "dataset": { "format": "folder_per_class", "num_classes": 1000, "class_names": [...] },
  "loss": { "type": "CrossEntropyLoss", "params": {} },
  "preprocess": { "mean": [...], "std": [...], "size": [...], "channels": 3 },
  "dependencies": { "pip": [...], "local_files": [...] },
  "confidence": { "task_type": "high", "input_shape": "medium", ... }
}
```

### HTML 格式（用户确认界面）

- 表格呈现，每行标注信心度色标
- 中/低可信度项高亮提醒
- 用户可在线编辑字段
- 「确认」/「修改后确认」按钮

---

## 报告交付机制

Harness 内置轻量 HTTP 服务器（Python `http.server`），绑定 `127.0.0.1`。

### 工作流程

1. Harness 启动 → 拉起 HTTP 服务器（如 `http://localhost:65049`）
2. 分析完成 → Agent 写入 `report.html` 到服务目录，页面自动刷新
3. 用户查看报告，可在线修改字段
4. 用户点击「确认」→ 服务器记录 `events.jsonl`
5. Agent 读取确认事件 + 修改内容 → 继续下一步
6. 同一标签页贯穿全流程：报告 → 审查 → 最终结果

### 技术实现

| 组件 | 实现 |
|------|------|
| HTTP 服务器 | Python `http.server`，仅绑定 `127.0.0.1` |
| 页面自动刷新 | 前端定时轮询或 SSE 推送 |
| 在线编辑 | HTML `contenteditable`，修改字段标记后提交 |
| 事件回传 | `POST /api/confirm`，写入 `events.jsonl` |
| Agent 轮询 | Harness 监控 events 文件变更 |
| 安全性 | `127.0.0.1` 仅本机，API key 不经过 Harness |

---

## 用户交互时间线

```
① 用户提供输入 → Dispatcher 获取资料（自动）
② Paper Agent + Code Agent 并行分析（自动）
⏸️ 确认点 1：分析报告（HTML，用户可修改任意字段后确认）
③ 测试生成 → 代码生成 → 运行测试（自动）
⏸️ 确认点 2：代码审查报告（HTML，全 ✅ 后用户确认）
④ 快速训练验证（自动）
⏸️ 确认点 3：最终结果（文件清单 + 验证通过）
🏁 任务就绪，可开始训练
```

---

## 触发方式

### MVP（第一阶段）：Claude Code Skill

用户通过 `/paper2mlops` 命令触发：

```bash
# 论文
/paper2mlops https://arxiv.org/abs/1512.03385

# GitHub
/paper2mlops https://github.com/pytorch/vision

# 论文 + 代码（空格分隔）
/paper2mlops https://arxiv.org/abs/1512.03385 https://github.com/pytorch/vision

# 博客 / 本地 PDF 等
/paper2mlops https://example.com/blog /path/to/paper.pdf
```

Skill 文件 (`SKILL.md`) 定义工作流，Dispatcher 启动后按阶段调度各子 Agent。

### 跨平台兼容

同一套 Harness（`core/paper2mlops/`）可被不同平台调用：

| 平台 | 触发方式 | 工具映射 |
|------|---------|---------|
| Claude Code | `/paper2mlops` Skill | Bash / Write / Agent 等 CC 工具 |
| Codex | Codex skill | Codex 等价工具 |
| Web App（第二阶段） | HTTP endpoint | Python 函数直接调用 |

---

## 错误处理 & 多轮修正

### 获取失败

- arXiv 下载失败 → 提示用户检查链接，或手动提供 PDF
- Git clone 失败（私有仓库/网络问题）→ 提示用户手动 clone 到本地目录，Agent 直接读取
- 博客页面无法抓取 → 建议用户粘贴关键段落，Agent 继续

### 分析不确定

- Agent 对某项信心度为 `low` → HTML 报告中红色高亮，要求用户手动填写
- 用户确认时修改了字段 → 以用户修改为准，覆盖 Agent 推断值

### 代码生成失败

- 测试不通过 → Agent 分析错误信息，修正代码后重新测试（最多 3 次循环）
- 3 次仍失败 → 停止并向用户报告具体错误，由用户决策

### 审查不通过

- 有 ❌ 项 → Agent 自动修复 → 重新审查（最多 3 次循环）
- 3 次仍有 ❌ → 停止并展示问题清单，用户决定「手动修复」或「接受风险继续」

### 长久机制

每次失败都是一次 Harness 改进机会 — 错误模式和修复方案写入 SKILL.md 的「已知问题」段，后续同类论文不再犯相同错误。

---

## 测试策略

对应 CLAUDE.md 规定的三层测试：

1. **导入和注册链**：`importlib.import_module` → 检查 `MODELS` / `DATASETS` 字典
2. **前向传播**：`get_example_input()` → `forward()` → 验证输出 shape
3. **完整流程**：`python train.py --task xxx --fast`（如有数据）

测试先于代码生成 — 测试用例在生成 `model.py` 之前就已定义好，代码生成后立即验证。

---

## 代码审查标准

Subagent 独立审查（不共享实现上下文），按 CLAUDE.md 六项清单：

1. 低耦合：core/ 不包含具体任务名
2. 硬编码：参数来自 config 而非写死
3. 接口合规：继承 BaseModel/BaseDataset，实现必要方法
4. 安全隐患：无密钥泄露，gitignore 完整
5. 代码质量：简体中文注释，语义清晰命名
6. 测试验证：三层测试通过

有 ❌ → 自动修复 → 重新审查，循环直到全 ✅ 或用户介入。

---

## 文件结构

```
MLOps/
├── core/
│   ├── paper2mlops/           # Harness 工具层（平台无关 Python 包）
│   │   ├── harvester.py       #   输入获取（arxiv / git clone / web_fetch）
│   │   ├── analyzer.py        #   代码分析（AST / 依赖图 / forward 追踪）
│   │   ├── parser.py          #   论文解析（PDF 文本提取 / 架构定位）
│   │   ├── generator.py       #   代码生成（模板渲染 / MLOps 接口适配）
│   │   ├── server.py          #   本地 HTTP 服务器（报告交付）
│   │   └── validator.py       #   验证（导入链 / 前向传播 / train --fast）
│   ├── registry.py            #   （现有）
│   ├── base_model.py          #   （现有）
│   └── ...                    #   （现有）
├── .claude/skills/
│   └── paper2mlops/SKILL.md   # Claude Code 入口（薄）
├── .codex/skills/
│   └── paper2mlops/           # Codex 入口（后续）
├── .superpowers/paper2mlops/{task_id}/
│   ├── raw/                   # 下载的论文 PDF / clone 的仓库
│   ├── analysis/              # Paper Agent + Code Agent 的分析输出
│   │   ├── report.json        #   Agent 间通信协议
│   │   ├── report.html        #   用户确认界面
│   │   ├── paper.json         #   Paper Agent 原始输出
│   │   └── code.json          #   Code Agent 原始输出
│   ├── tests/                 # 自动生成的测试用例
│   └── review.md              # 代码审查报告
├── tasks/<name>/              # 生成的 MLOps 任务（目标产物）
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── config.yaml
│   └── <依赖>.py
└── docs/superpowers/specs/    # 设计文档
```

---

## 领域支持

不限领域（CV、NLP、语音、RL 等），Agent 自行判断任务类型并适配 MLOps 接口。首批验证以 CV（分类、分割）为起点，因为现有框架已有成熟实现可对标。
