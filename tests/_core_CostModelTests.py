from _core_support import *
from _core_support import _load_experiment, _tiny_config

class CostModelTests(unittest.TestCase):
    def test_a_counts_first_stage(self):
        cfg = _tiny_config(mode="A")
        profile = experiment.mode_cost_profile(cfg)
        # top_k=2, k1=2, k2=2, n_layers=4 -> middle=2
        # pre+post = 2 -> 2*2 + 2*(2 + 4) = 16
        self.assertEqual(
            profile["expert_assignments_per_token"], 16
        )

    def test_a_counts_routers_on_each_branch(self):
        # middle=2, pre=post=1, k1=2：A 的第二轮在 k1 个分支上
        # 各自路由，所以每个中间 Block 有 1+k1=3 次 Router/MoE
        # 执行，总计 1+1+2*3=8，而不是 layers+middle=6。
        profile = experiment.mode_cost_profile(
            _tiny_config(mode="A")
        )
        self.assertEqual(
            profile["router_calls_per_token"], 8
        )
        self.assertEqual(
            profile["moe_passes_per_token"], 8
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        self.assertEqual(
            early["analytic_formula"]["attention_executions"],
            "pre + [1 + (middle-1)*k1 + middle*k1] + post",
        )

    def test_b_counts_first_round_branch_routing(self):
        cfg = _tiny_config(mode="B")
        profile = experiment.mode_cost_profile(cfg)
        # middle=2, pre=post=1, top_k=k1=k2=2:
        # first block creates k1 branches (k1 assignments), the
        # remaining middle-1 first-round blocks route k1 experts
        # on each of the k1 branches (k1*k1 each), and the second
        # round routes k2 experts on each branch.
        self.assertEqual(
            profile["expert_assignments_per_token"],
            2 * (1 + 1) + 2 + 1 * 2 * 2 + 2 * 2 * 2,
        )

    def test_b_early_matches_b_execution_counts(self):
        b = experiment.mode_cost_profile(
            _tiny_config(mode="B")
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        for key in (
            "attention_executions_per_token",
            "moe_passes_per_token",
            "expert_assignments_per_token",
            "router_calls_per_token",
            "analytic_flops_per_token",
        ):
            self.assertEqual(b[key], early[key], key)

    def test_kv_cache_bytes_counts_k_and_v_once(self):
        # d_model=32, n_heads=4, n_kv_heads=2 -> head_dim=8,
        # kv_dim=16；每层 fp16 的 K+V = 16*2*2 = 64 bytes。
        bytes_per_layer = 16 * 2 * 2
        cfg = _tiny_config(mode="smelt")
        profile = experiment.mode_cost_profile(cfg)
        # 物理层仍只有 4 份；middle=2 层访问两次，pre/post 各
        # 一次 -> 共 1+2*2+1=6 次 Attention，而不是 4*2。
        self.assertEqual(
            profile["kv_cache_bytes_per_token"],
            4 * bytes_per_layer,
        )
        self.assertEqual(
            profile["kv_cache_bytes_per_token_per_visit"],
            6 * bytes_per_layer,
        )
        self.assertEqual(
            profile["kv_cache_bytes_per_token_per_visit"],
            profile["attention_executions_per_token"]
            * bytes_per_layer,
        )

        # B 有 k1=2 个独立分支：分支只出现在 middle，pre/post
        # 仍各一份。middle=2 -> 物理容量 (1+1+2*2)=6 份。
        branch_profile = experiment.mode_cost_profile(
            _tiny_config(mode="B")
        )
        self.assertEqual(
            branch_profile["kv_cache_bytes_per_token"],
            6 * bytes_per_layer,
        )
        # 访问次数：pre 1 + [首块 1 + 剩余 1 块×2 分支 +
        # 第二轮 2 块×2 分支] + post 1 = 9。
        self.assertEqual(
            branch_profile["kv_cache_bytes_per_token_per_visit"],
            9 * bytes_per_layer,
        )
        self.assertEqual(
            branch_profile[
                "kv_cache_bytes_per_token_per_visit"
            ],
            branch_profile["attention_executions_per_token"]
            * bytes_per_layer,
        )
        early = experiment.mode_cost_profile(
            _tiny_config(mode="B_early")
        )
        self.assertEqual(
            early["kv_cache_bytes_per_token"],
            branch_profile["kv_cache_bytes_per_token"],
        )
        self.assertEqual(
            early["kv_cache_bytes_per_token_per_visit"],
            branch_profile["kv_cache_bytes_per_token_per_visit"],
        )

    def test_dense_profile_counts_dense_ffn_flops(self):
        # dense 默认 H_ffn = top_k * expert_hidden = 32。
        # 每层 SwiGLU FFN = 6*D*H = 6*32*32；共 4 层。
        cfg = _tiny_config(mode="dense")
        profile = experiment.mode_cost_profile(cfg)
        breakdown = profile["analytic_flops_breakdown"]
        expected_ffn = 4 * 6 * 32 * 32
        self.assertEqual(breakdown["dense_ffn"], expected_ffn)
        self.assertEqual(
            profile["analytic_flops_per_token"],
            sum(breakdown.values()),
        )
        # 等计算口径下，稠密 FFN 的激活 FLOPs 应与 baseline
        # 单次前向激活的 top_k 个专家相同；dense 少了路由开销。
        base = experiment.mode_cost_profile(
            _tiny_config(mode="base")
        )
        self.assertEqual(
            breakdown["dense_ffn"],
            base["analytic_flops_breakdown"]["experts"],
        )
        self.assertGreater(
            base["analytic_flops_per_token"],
            profile["analytic_flops_per_token"],
        )

    def test_budget_search_reports_recommendation(self):
        cfg = _tiny_config(mode="A")
        report = experiment.budget_match_report(
            cfg, "smelt", 0.1
        )
        self.assertIn(
            report["search"]["kind"], {"k2", "k1_k2"}
        )
        self.assertIsNotNone(
            report["search"]["recommended_k2"]
        )
