# SMELT-ISR 质量门统一入口。
#
# 用法（需先 `pip install -e ".[dev]"` 安装开发依赖）：
#   make check      全部质量门（lint + typecheck + test）
#   make lint       ruff 静态检查
#   make format     ruff 格式化（只影响手写文件，机械拆分文件由
#                   pyproject 的 per-file-ignores 范围约束）
#   make typecheck  mypy（exec 组合的拆分文件已被 exclude，见
#                   pyproject [tool.mypy] 的说明）
#   make test       全量回归（unittest discover，与 CI/循环测试门一致）
#   make self-test  结构自测（含同种子逐位复现检查）
#
# 注意：拆分文件（exp/part_*.py、exp/mainflow/、exp/analysisflow/、
# selfcheck/checks_*.py）由运行时门（py_compile + 全量 unittest +
# self-test）兜底，静态检查对它们关闭，理由见 pyproject 相关注释。

PYTHON ?= python3

.PHONY: help check lint format typecheck test self-test

help:
	@echo "check / lint / format / typecheck / test / self-test"

check: lint typecheck test

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format tests/ experiment.py \
		selfcheck/__init__.py selfcheck/_manifest.py \
		exp/__init__.py exp/_manifest.py exp/_runner.py
	$(PYTHON) -m ruff check --fix .

typecheck:
	$(PYTHON) -m mypy .

test:
	$(PYTHON) -m unittest discover -s tests -v

self-test:
	$(PYTHON) experiment.py --self-test
