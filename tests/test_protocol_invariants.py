"""对照协议关键不变量的回归测试（外部审计 2026-10-26 补充）。

这些不变量在 README「对照口径与可复现性」与
docs/methodology_audit.md 中声明，但此前没有直接的回归测试
锁定前向行为：

- local_base 的第二轮 aux 必须折算回 k2 宽度：均衡损失约随
  路由宽度线性增长（见 SparseMoE.route 注释），不折算会让
  「立即合并」臂在 (1+N)×2 消融中多受均衡压力，把路由宽度
  与均衡强度混杂（methodology_audit 第 4 节）。
- B 的跨分支合并必须使用第一轮 Top-k1 归一化的固定 gates，
  不得用第二轮/最终分支状态重新计算合并权重（README「B 的
  两轮路由与合并权重」）。

两个测试都经过变异验证：人为移除 aux 折算、把 B 合并权重改
为均匀权重时，测试分别失败；实现恢复后全部通过。

Run with: python -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError as error:  # pragma: no cover
    raise unittest.SkipTest(
        f"torch is required for these tests: {error}"
    )


def _load_experiment():
    path = Path(__file__).resolve().parents[1] / "experiment.py"
    spec = importlib.util.spec_from_file_location(
        "smelt_protocol_experiment", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["smelt_protocol_experiment"] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


def _tiny_config(**overrides):
    values = dict(
        seq_len=16,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        n_layers=4,
        n_experts=4,
        expert_hidden=16,
        top_k=2,
        k1=2,
        k2=2,
        mode="smelt",
    )
    values.update(overrides)
    return experiment.Config(**values)


class ProtocolInvariantTests(unittest.TestCase):
    def test_local_base_second_round_aux_folds_to_k2_width(self):
        # n_experts=4, k1=2, k2=2 -> 等计算量宽度 min(4, 4)=4 != 2，
        # 折算分支必须生效。
        cfg = _tiny_config(mode="local_base")
        block = experiment.TransformerBlock(cfg)
        block.attention_step = (
            lambda x, scale=None, attn_mask=None: x
        )
        widths = []

        def moe_step(x, top_k, scale=None):
            widths.append(int(top_k))
            return x, torch.tensor(float(top_k))

        block.moe_step = moe_step
        _, aux1, aux2 = block.local_two_stage(
            torch.zeros(1, 1, 32), 2, 2, delayed=False
        )
        # 第一轮按 k1=2 路由；第二轮按 min(n_experts, k1*k2)=4。
        self.assertEqual(widths, [2, 4])
        # 第一轮 aux 不折算。
        self.assertEqual(float(aux1), 2.0)
        # 第二轮 aux 折算回 A/B 的单分支宽度 k2：4 * (2/4) = 2。
        self.assertAlmostEqual(float(aux2), 2.0)

    def test_local_base_k2_second_round_keeps_k2_width(self):
        # 机制对照：第二轮固定 k2，宽度与 aux 都不折算。
        cfg = _tiny_config(mode="local_base_k2")
        block = experiment.TransformerBlock(cfg)
        block.attention_step = (
            lambda x, scale=None, attn_mask=None: x
        )
        widths = []

        def moe_step(x, top_k, scale=None):
            widths.append(int(top_k))
            return x, torch.tensor(float(top_k))

        block.moe_step = moe_step
        _, aux1, aux2 = block.local_two_stage(
            torch.zeros(1, 1, 32),
            2,
            2,
            delayed=False,
            second_round_width=2,
        )
        self.assertEqual(widths, [2, 2])
        self.assertEqual(float(aux1), 2.0)
        self.assertEqual(float(aux2), 2.0)

    def test_b_merge_uses_fixed_first_round_gates(self):
        # 前后区域 Block、两轮中间执行全部替换为恒等映射，只保留
        # 「分支创建（固定 gates）-> 跨分支合并」路径：若实现改为
        # 用第二轮/最终分支状态重新计算合并权重，得到的系数将与
        # first_gates 不同，断言即失败。
        cfg = _tiny_config(mode="B", k1=2, k2=2)
        model = experiment.LanguageModel(cfg).eval()
        batch, time = 2, 4

        def identity_block(x, top_k, scale, attn_mask=None):
            return x, torch.zeros(())

        model.blocks[0].forward = identity_block
        model.blocks[cfg.n_layers - 1].forward = identity_block

        # 逐位置变化的第一轮归一化 gates（time=4, k1=2）。
        first_gates = torch.tensor(
            [
                [0.70, 0.30],
                [0.60, 0.40],
                [0.80, 0.20],
                [0.55, 0.45],
            ]
        )

        def first_branch_step(x, k1, scale=None, attn_mask=None):
            b, t, dim = x.shape
            branches = torch.zeros(b, k1, t, dim)
            for index in range(k1):
                # 分支 b 的所有 token 恒为 b+1。
                branches[:, index] = float(index + 1)
            gates = (
                first_gates.unsqueeze(0)
                .expand(b, t, k1)
                .clone()
            )
            return (
                branches,
                gates,
                torch.zeros(b, t, k1, dtype=torch.long),
                torch.zeros(()),
            )

        model.middle_blocks[0].first_branch_step = (
            first_branch_step
        )

        def run_independent_blocks(
            block, branches, top_k, scale=None, attn_mask=None
        ):
            return branches, torch.zeros(())

        model.run_independent_blocks = run_independent_blocks

        with torch.no_grad():
            logits = model(
                torch.zeros(batch, time, dtype=torch.long)
            )["logits"]

        # 期望：每个位置按第一轮 gates 加权合并恒等分支
        # 1.0 / 2.0。
        merged = (
            first_gates[:, 0] * 1.0 + first_gates[:, 1] * 2.0
        )
        expected_x = merged.view(1, time, 1).expand(
            batch, time, cfg.d_model
        )
        with torch.no_grad():
            expected = torch.nn.functional.linear(
                model.final_norm(expected_x),
                model.lm_head.weight,
            )
        torch.testing.assert_close(
            logits, expected, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
