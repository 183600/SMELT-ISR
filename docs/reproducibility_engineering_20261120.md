# 可复现性与工程规范改进记录（2026-11-20）

本轮聚焦「让仓库既有质量门真正可用」：可复现性基础设施（环境快照、
种子清单、运行审计、预注册、产物清单）在此前迭代中已经相当完整，
本次不改数值语义，只修静态检查链条中三处「配置存在但从未生效」的
隐性失效，并完成仓库卫生归档。

## 一、修复三处从未生效的质量门（核心发现）

### 1.1 mypy 的 `exclude` 从未匹配过任何文件
- **问题**：`[tool.mypy] exclude` 接收**正则表达式**，而旧配置写的是
  glob 风格（`"exp/part_*.py"`）。作为正则，`part_*.py` 只匹配
  `part_.py` / `part____.py` 之类，实际不匹配 `part_000_config.py`；
  `exp/mainflow/*.py` 同理。叠加根目录残留的 `__init__.py`（含连字符
  的 `SMELT-ISR` 不是合法包名，mypy 直接中止），mypy 在本仓库从未
  成功检查过任何文件。
- **修复**：
  - 排除项改写为正确正则（`^exp/part_.+\.py$` 等），并补上同属
    exec 组合文件的 `selfcheck/checks_*.py`；
  - 删除根目录 `__init__.py`（无任何引用方；`pip install -e .` 的
    `packages = ["exp", "selfcheck"]` 也不包含它）。
- **结果**：`mypy .` 从「0 文件被检查」变为 34 个源文件、0 错误。
  修复过程中它立即抓出 15 个真实问题（tests/ 依赖 star-import 链
  间接获得 `argparse`/`json`/`hashlib` 的 NameError 隐患、
  `experiment.py` 对动态命名空间的属性访问），全部已修复。

### 1.2 ruff 报 30,484 个错误，信号完全不可用
- **问题**：`exp/__init__.py` 在导入期用 `exec` 把 53 个 part 组合进
  同一命名空间（跨 part 解析名字），任何单文件静态分析都无法判定
  `F821`（9,303 个）/`F405`（2,241 个）等是否成立；注释/字符串/文档
  使用中文全角标点触发 `RUF001/002/003`（约 1.5 万个）。lint 输出
  被架构性误报淹没，等于没有 lint。
- **修复**（配置对齐架构，不改任何代码语义）：
  - 关闭 `RUF001/002/003`（中文全角标点是本项目文档语言）、
    `NPY002`（legacy `np.random` 全局 API 是续跑/检查点逐位存档的
    依赖）、`TRY003`（仓库统一「直接 raise 带长中文说明」风格）；
  - `exp/part_*.py`、`exp/mainflow/`、`exp/analysisflow/`、
    `selfcheck/checks_*.py` 的语法与行为由运行时门兜底
    （`py_compile` + 全量 unittest + `--self-test` 逐位复现检查），
    静态检查对这些文件关闭，理由写入 pyproject 注释；
  - `tests/*.py` 豁免 star-import 既有模式（`F401/F403/F405/I001`）
    与测试惯例项（`PLR2004/PLR0913/PLC0415/PERF401`）。
- **结果**：`ruff check .` 从 30,484 → **0** 违规；期间修掉的真实
  问题包括：`tests/_core_support.py` 抛出 `SkipTest` 缺 `from error`、
  测试签名隐式 `Optional`、`open()`/`os.path` 未用 `pathlib`、
  `zip()` 无显式 `strict=`（全部按「保持原行为」选择 `strict=False`
  或 `itertools.pairwise`）、未使用变量/导入等约 190 处。
- **教训（已写入配置注释）**：`tests/_core_support.py` 顶部的
  「未使用」标准库 import 实为全体测试经 `from _core_support import *`
  间接消费的公共依赖，`ruff --fix` 的 F401 自动删除会一次性炸掉
  257 个测试——这就是 `tests/*.py` 必须豁免 F401 的原因。

### 1.3 `.pre-commit-config.yaml` 损坏，pre-commit 从未能运行
- **问题**：
  - `no-add-to-large-files` 是不存在的 hook id（正确 id 为
    `check-added-large-files`），`pre-commit validate-config` 直接
    报错——本文件的任何 hook 从未被执行过；
  - `black` 与 `ruff-format` 两个格式化器同时启用，输出互不兼容、
    会互相覆盖；
  - `mirrors-prettier` 作用于全部 Markdown/JSON/YAML，会重排中文
    文档；
  - `pre-commit-hooks` v4.6.0 使用已废弃的 stage 名。
- **修复**：修正 hook id；移除 black（格式化工具统一为 ruff-format，
  与 pyproject 的 ruff 配置一致）；prettier 收窄到 JSON/YAML；
  升级 pre-commit-hooks 至 v5.0.0；**顶层 `exclude: '^\.github/'`**
  确保任何 hook 都不会改动 `.github/`；ruff/mypy hook 用
  `exclude`/`files` 与 pyproject 的豁免范围对齐（显式传参会绕过
  mypy 的 exclude，必须在 hook 层过滤），并加 `--force-exclude`。
- **验证**：`pre-commit validate-config` 通过；全部手写文件已满足
  `ruff format --check`。

## 二、仓库卫生与工程规范

### 2.1 根目录审计报告归档
- 根目录此前散落 60+ 个评审报告（`CODE_REVIEW_*.md`、
  `DEEP_REVIEW_ADDENDUM_*.md`、`review_20261120/`、
  `review_report_20261121_v2.md`、`ANALYSIS_IMPROVEMENTS.md`），
  干扰对仓库结构的理解，也不符合「评审链集中归档」的惯例。
- 全部 `git mv` 至 `reviews/`（内容逐字节保留，历史可追溯）；
  `ANALYSIS_IMPROVEMENTS.md` 是方法学建议文档，归入 `docs/`；
  `review_20261121.py` 是仅有 docstring、伪装成 `.py` 的报告，转为
  `reviews/20261121_deep_review.md`（去除 docstring 引号外壳，
  内容不变）。审计链完整性不受影响——只移动、未删除。

### 2.2 新增开发工作流入口
- `pyproject.toml` 新增 `[project.optional-dependencies] dev`
  （pytest / ruff / mypy / pre-commit），`pip install -e ".[dev]"`
  一条命令装齐质量门工具；CI 的精确版本 pin 仍由
  `requirements-ci.txt` 负责。
- 新增 `Makefile`：`make check` = `lint + typecheck + test`；
  另有 `format` / `self-test`。`test` 目标与 CI 测试门、循环测试门
  保持同一命令（`python -m unittest discover -s tests`），避免
  「Makefile 绿、CI 红」的口径漂移。

### 2.3 新增 CITATION.cff
- 按 Citation File Format 1.2.0 提供软件引用元数据（版本与
  pyproject 同步为 0.1.0），`preferred-citation` 指向口径基线论文
  arXiv:2609.01343；论文未披露量的推导规则指向
  `docs/paper_reproduction.md`。科研软件的可引用性属于可复现性
  工作的一部分。

## 三、与受保护文件的关系
- `.github/`：未做任何改动（本回合唯一触碰方式是新增 pre-commit
  顶层 exclude，防止 hook 在未来改动它）。
- `README.md` / `README_en.md` / `算力申请.md`：未改动、未移动。
- 实验结果、审计记录：全部保留（仅按上述清单移动归档）。

## 四、验证
```bash
ruff check .          # 0 违规（改动前 30,484）
mypy .                # 34 个源文件，0 错误（改动前无法启动）
ruff format --check   # 全部手写文件合规
pre-commit validate-config .pre-commit-config.yaml  # 通过
python -m unittest discover -s tests   # 607 tests, OK（与改动前一致）
```

## 五、遗留建议
- CI（tests.yml，不可修改）只跑 pytest；建议未来在依赖安装步骤加
  `pip install -e ".[dev]"` 并追加 `ruff check .` 与 `mypy .` 两步，
  使静态检查与本地 `make check` 等效。
- `exp/mainflow/part_*.py` 与 `exp/analysisflow/part_*.py` 同名不同
  目录，是 mypy duplicate-module 的来源之一；长期看拆分文件应逐步
  收敛为可独立导入的显式模块（此项涉及语义重排，需单独评审）。
