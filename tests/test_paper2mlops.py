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
    with open(model_path, encoding="utf-8") as f:
        content = f.read()
    assert "@register_model" in content
    assert "BaseModel" in content
    assert "get_example_input" in content


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
