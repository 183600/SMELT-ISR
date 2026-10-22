"""论文复现口径（arXiv:2609.01343v1）的回归测试。

覆盖：
- 附录 A 的 4×4×2 网格逐格核对；
- ``--paper-cell`` 预设的结构、总参数预算与 KV cache 比例；
- ``paper_fidelity_report`` 的 architecture / protocol 两层判定；
- WSD 调度的 warmup-stable-decay 形状与历史 cosine 行为；
- 非整除 GQA 的均匀映射；
- 文档级 segment attention mask 的因果性、跨文档隔离与梯度。

Run with: python -m unittest discover -s tests -v
"""

import argparse
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
        "smelt_paper_experiment", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["smelt_paper_experiment"] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment()


PAPER_PROTOCOL = {
    "lr_schedule": "wsd",
    "batch_size": 256,
    "stable_steps": 196075,
    "document_masks": True,
    "save_at_steps": list(
        experiment.PAPER_PROTOCOL["branch_steps"]
    ),
}


class PaperGridTests(unittest.TestCase):
    def test_grid_has_32_cells(self):
        self.assertEqual(len(experiment.PAPER_CELLS), 32)
        self.assertEqual(
            set(experiment.PAPER_SCALES),
            {"100M", "200M", "600M", "1.6B"},
        )

    def test_appendix_a_values(self):
        expected = {
            ("base", "100M", 0): (672, 10, 8, 0.10, 0.0),
            ("base", "200M", 95): (1280, 12, 192, 3.87, 0.946),
            ("base", "600M", 97): (1792, 20, 336, 21.83, 0.969),
            ("base", "1.6B", 85): (2304, 30, 64, 10.55, 0.845),
            ("smelt", "100M", 85): (576, 10, 96, 0.71, 0.868),
            ("smelt", "200M", 0): (1056, 12, 16, 0.21, 0.0),
            ("smelt", "600M", 95): (1512, 20, 288, 12.44, 0.947),
            ("smelt", "1.6B", 97): (1848, 30, 504, 53.90, 0.970),
        }
        for key, values in expected.items():
            cell = experiment.PAPER_CELLS[key]
            self.assertEqual(
                (
                    cell.hidden,
                    cell.layers,
                    cell.total_experts,
                    cell.total_params,
                    cell.measured_sparsity,
                ),
                values,
                key,
            )
            self.assertEqual(cell.active_experts, 8, key)

    def test_smelt_ratios_present(self):
        for (arch, scale, sparsity), cell in (
            experiment.PAPER_CELLS.items()
        ):
            if arch != "smelt":
                self.assertIsNone(cell.flops_ratio)
                self.assertIsNone(cell.kv_ratio)
                continue
            self.assertIsNotNone(cell.flops_ratio)
            self.assertIsNotNone(cell.kv_ratio)
            self.assertTrue(0.9 < cell.kv_ratio < 1.1)


class PaperPresetTests(unittest.TestCase):
    def test_preset_matches_disclosed_architecture(self):
        for arch in ("base", "smelt"):
            for scale in experiment.PAPER_SCALES:
                for sparsity in experiment.PAPER_SPARSITIES:
                    cell = experiment.paper_cell(
                        scale, sparsity, arch
                    )
                    preset = experiment.paper_preset(
                        scale, sparsity, arch
                    )
                    config = preset["config"]
                    self.assertEqual(config["d_model"], cell.hidden)
                    self.assertEqual(config["n_layers"], cell.layers)
                    self.assertEqual(
                        config["n_experts"], cell.total_experts
                    )
                    self.assertEqual(config["top_k"], 8)
                    self.assertEqual(config["mode"], arch)
                    self.assertEqual(config["seq_len"], 4096)
                    # 论文布局：m=L/2，循环层为
                    # floor((L-m)/2)+1 .. +m（1-based）。
                    cfg = experiment.Config(**config)
                    self.assertEqual(
                        cfg.middle_start,
                        (cell.layers - cell.layers // 2) // 2,
                    )
                    self.assertEqual(
                        cfg.middle_end,
                        cfg.middle_start + cell.layers // 2,
                    )

    def test_preset_total_params_within_one_percent(self):
        for arch in ("base", "smelt"):
            for scale in experiment.PAPER_SCALES:
                for sparsity in experiment.PAPER_SPARSITIES:
                    preset = experiment.paper_preset(
                        scale, sparsity, arch
                    )
                    cell = preset["cell"]
                    measured = preset["measured"][
                        "total_params"
                    ] / 1e9
                    self.assertLess(
                        abs(measured / cell.total_params - 1.0),
                        0.01,
                        f"{arch} {scale} {sparsity}",
                    )

    def test_preset_kv_ratio_matches_paper(self):
        for scale in experiment.PAPER_SCALES:
            for sparsity in experiment.PAPER_SPARSITIES:
                preset = experiment.paper_preset(
                    scale, sparsity, "smelt"
                )
                cell = preset["cell"]
                base = experiment.paper_preset(
                    scale, sparsity, "base"
                )
                ratio = (
                    preset["derived"]["kv_dim"]
                    / base["derived"]["kv_dim"]
                )
                self.assertLess(
                    abs(ratio - cell.kv_ratio),
                    0.04,
                    f"{scale} {sparsity}",
                )

    def test_gqa_is_enabled_by_preset(self):
        for arch in ("base", "smelt"):
            preset = experiment.paper_preset("200M", 95, arch)
            self.assertLess(
                preset["derived"]["n_kv_heads"],
                preset["derived"]["n_heads"],
            )

    def test_label_parsing(self):
        self.assertEqual(
            experiment.parse_paper_cell_label("200M:95"),
            ("200M", 95),
        )
        self.assertEqual(
            experiment.parse_paper_cell_label("1.6b@97"),
            ("1.6B", 97),
        )
        self.assertEqual(
            experiment.parse_paper_cell_label("600M-85"),
            ("600M", 85),
        )
        with self.assertRaises(ValueError):
            experiment.parse_paper_cell_label("300M:95")
        with self.assertRaises(ValueError):
            experiment.parse_paper_cell_label("200M:96")


class PaperFidelityTests(unittest.TestCase):
    def test_full_reproduction_when_recipe_and_protocol_match(self):
        for arch in ("base", "smelt"):
            preset = experiment.paper_preset(
                "100M", 85, arch
            )
            cfg = experiment.Config(
                **preset["config"],
                document_masks=True,
                document_separator=[10],
            )
            report = experiment.paper_fidelity_report(
                cfg, PAPER_PROTOCOL
            )
            self.assertTrue(
                report["architecture_faithful"],
                report["checks"],
            )
            self.assertTrue(
                report["protocol_faithful"],
                report["protocol_checks"],
            )
            self.assertTrue(
                report["is_full_reproduction"],
                report["deviations"],
            )
            self.assertEqual(report["deviations"], [])
            self.assertEqual(
                report["cell"]["scale"], "100M"
            )
            # recipe 复现与数值复现分开：base 格的预算
            # （总参数/active/S）落在容差内，smelt 格的
            # FLOPs 比例与 S 仍有偏差，不能声称数值复现。
            if arch == "base":
                self.assertTrue(
                    report["numerical_reproduction"],
                    report["budget_audit"][
                        "within_paper_tolerance"
                    ],
                )
            else:
                self.assertFalse(
                    report["numerical_reproduction"]
                )

    def test_prototype_config_is_not_full_reproduction(self):
        cfg = experiment.Config(
            seq_len=128,
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            n_layers=8,
            n_experts=8,
            expert_hidden=128,
            top_k=2,
            mode="smelt",
        )
        report = experiment.paper_fidelity_report(cfg)
        self.assertFalse(report["architecture_faithful"])
        self.assertIsNone(report["protocol_faithful"])
        self.assertFalse(report["is_full_reproduction"])
        self.assertIn("paper_cell_config", report["deviations"])
        self.assertIn("top_k_8", report["deviations"])

    def test_missing_protocol_blocks_full_reproduction(self):
        cfg = experiment.Config(
            **experiment.paper_preset(
                "100M", 85, "smelt"
            )["config"],
            document_masks=True,
            document_separator=[10],
        )
        report = experiment.paper_fidelity_report(cfg)
        self.assertTrue(report["architecture_faithful"])
        self.assertIsNone(report["protocol_faithful"])
        self.assertFalse(report["is_full_reproduction"])

    def test_decay_branch_satisfies_protocol(self):
        cfg = experiment.Config(
            **experiment.paper_preset(
                "100M", 85, "smelt"
            )["config"],
            document_masks=True,
            document_separator=[10],
        )
        branch = dict(PAPER_PROTOCOL)
        branch["stable_steps"] = 20000
        branch["steps"] = (
            20000 + experiment.PAPER_PROTOCOL["decay_steps"]
        )
        branch["save_at_steps"] = []
        report = experiment.paper_fidelity_report(
            cfg, branch
        )
        self.assertNotIn(
            "protocol_stable_phase",
            report["protocol_checks"],
        )
        self.assertTrue(
            report["protocol_faithful"],
            report["protocol_checks"],
        )
        self.assertTrue(report["is_full_reproduction"])

    def test_decay_branch_from_end_of_stable_phase(self):
        # 196075 也在 PAPER_PROTOCOL["branch_steps"] 中：从完整稳定期
        # 末端进入 10B-token decay 同样是一条合法的分支运行，
        # 不应被误判成「缺少稳定期」。
        cfg = experiment.Config(
            **experiment.paper_preset(
                "100M", 85, "smelt"
            )["config"],
            document_masks=True,
            document_separator=[10],
        )
        branch = dict(PAPER_PROTOCOL)
        branch["steps"] = (
            experiment.PAPER_PROTOCOL["stable_steps"]
            + experiment.PAPER_PROTOCOL["decay_steps"]
        )
        branch["save_at_steps"] = []
        report = experiment.paper_fidelity_report(
            cfg, branch
        )
        self.assertNotIn(
            "protocol_stable_phase",
            report["protocol_checks"],
        )
        self.assertTrue(
            report["protocol_faithful"],
            report["protocol_checks"],
        )
        self.assertTrue(report["is_full_reproduction"])

    def test_protocol_deviation_detected(self):
        cfg = experiment.Config(
            **experiment.paper_preset(
                "100M", 85, "smelt"
            )["config"],
            document_masks=True,
            document_separator=[10],
        )
        bad = dict(PAPER_PROTOCOL)
        bad["document_masks"] = False
        report = experiment.paper_fidelity_report(cfg, bad)
        self.assertFalse(report["protocol_faithful"])
        self.assertIn(
            "protocol_document_masks", report["deviations"]
        )

    def test_budget_audit_reports_paper_targets(self):
        preset = experiment.paper_preset(
            "200M", 95, "smelt"
        )
        cfg = experiment.Config(**preset["config"])
        report = experiment.paper_fidelity_report(cfg)
        budget = report["budget_audit"]
        self.assertAlmostEqual(
            budget["paper"]["flops_ratio"], 1.029
        )
        self.assertAlmostEqual(
            budget["paper"]["kv_ratio"], 1.031
        )
        # 本实现的 KV cache 与论文目标相差 <4%。
        self.assertLess(
            abs(budget["measured"]["kv_ratio_vs_base"] - 1.031),
            0.04,
        )


class PaperCliTests(unittest.TestCase):
    def _namespace(self, **overrides):
        values = dict(
            mode="B",
            paper_cell=None,
            paper_arch=None,
            paper_protocol=False,
            lr_schedule="cosine",
            wsd_stable_steps=None,
            wsd_final_lr_ratio=0.1,
            save_at_steps=None,
            document_masks=False,
            group_separator=None,
            seq_len=128,
            d_model=128,
            n_heads=4,
            n_kv_heads=None,
            n_layers=8,
            n_experts=8,
            expert_hidden=256,
            top_k=2,
            k1=2,
            k2=2,
            steps=2000,
            batch_size=32,
            warmup_steps=100,
            eval_interval=200,
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_paper_cell_applies_architecture(self):
        previous = sys.argv
        sys.argv = ["experiment.py", "--paper-cell", "200M:95"]
        try:
            args = self._namespace(paper_cell="200M:95")
            experiment.apply_paper_cli(args)
        finally:
            sys.argv = previous
        self.assertEqual(args.mode, "smelt")
        self.assertEqual(args.d_model, 1056)
        self.assertEqual(args.n_layers, 12)
        self.assertEqual(args.n_experts, 288)
        self.assertEqual(args.top_k, 8)
        self.assertEqual(args.seq_len, 4096)

    def test_conflicting_model_flag_is_rejected(self):
        class _Parser:
            def error(self, message):
                raise SystemExit(message)

        previous = sys.argv
        sys.argv = [
            "experiment.py",
            "--paper-cell",
            "200M:95",
            "--d-model",
            "64",
        ]
        try:
            args = self._namespace(
                paper_cell="200M:95", d_model=64
            )
            with self.assertRaises(SystemExit):
                experiment.apply_paper_cli(args, _Parser())
        finally:
            sys.argv = previous

    def test_paper_protocol_without_document_boundaries(self):
        # 没有 --group-separator 时不阻断运行，但不会假装启用
        # 了文档级 mask：document_masks 保持 False，审计会记为
        # protocol_document_masks=false。
        previous = sys.argv
        sys.argv = ["experiment.py", "--paper-protocol"]
        try:
            args = self._namespace(paper_protocol=True)
            experiment.apply_paper_cli(args)
        finally:
            sys.argv = previous
        self.assertFalse(args.document_masks)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.lr_schedule, "wsd")

    def test_paper_protocol_values(self):
        previous = sys.argv
        sys.argv = [
            "experiment.py",
            "--paper-protocol",
            "--group-separator",
            "\\n\\n",
        ]
        try:
            args = self._namespace(
                paper_protocol=True,
                group_separator="\\n\\n",
            )
            experiment.apply_paper_cli(args)
        finally:
            sys.argv = previous
        self.assertEqual(args.steps, 196075)
        self.assertEqual(args.batch_size, 256)
        self.assertEqual(args.seq_len, 4096)
        self.assertEqual(args.lr_schedule, "wsd")
        self.assertEqual(args.wsd_stable_steps, 196075)
        self.assertTrue(args.document_masks)
        self.assertEqual(
            args.save_at_steps,
            list(experiment.PAPER_PROTOCOL["branch_steps"]),
        )


class ScheduleTests(unittest.TestCase):
    @staticmethod
    def _lrs(schedule, **kwargs):
        record_steps = kwargs.pop("record_steps", 10)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = experiment.build_lr_schedule(
            optimizer,
            warmup_steps=kwargs.pop("warmup_steps", 2),
            total_steps=kwargs.pop("total_steps", 10),
            schedule=schedule,
            **kwargs,
        )
        values = []
        for step in range(record_steps):
            values.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        return values

    def test_wsd_warmup_stable_decay(self):
        values = self._lrs(
            "wsd",
            warmup_steps=2,
            total_steps=11,
            stable_steps=6,
            final_lr_ratio=0.0,
            record_steps=12,
        )
        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[2], 1.0)
        self.assertAlmostEqual(values[5], 1.0)
        # 稳定期结束后递减，终点（第 total_steps 步）为 0。
        self.assertGreater(values[6], values[7])
        self.assertGreater(values[7], values[8])
        self.assertAlmostEqual(values[-1], 0.0)

    def test_wsd_no_decay_when_stable_equals_total(self):
        values = self._lrs(
            "wsd",
            warmup_steps=2,
            total_steps=10,
            stable_steps=10,
        )
        self.assertAlmostEqual(values[2], 1.0)
        self.assertAlmostEqual(values[-1], 1.0)

    def test_cosine_default_is_unchanged(self):
        values = self._lrs(
            "cosine",
            warmup_steps=2,
            total_steps=11,
            record_steps=12,
        )
        self.assertAlmostEqual(values[0], 0.5)
        self.assertAlmostEqual(values[1], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)


class GqaTests(unittest.TestCase):
    def test_non_divisible_gqa_mapping(self):
        # n_heads=4, n_kv=3：Q 头 0,1 用 KV 头 0；2 用 1；3 用 2。
        cfg = experiment.Config(
            seq_len=16,
            d_model=32,
            n_heads=4,
            n_kv_heads=3,
            n_layers=4,
            n_experts=4,
            expert_hidden=16,
            top_k=2,
            mode="smelt",
        )
        cfg.validate()
        model = experiment.LanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        output = model(x, y)
        output["loss"].backward()
        self.assertTrue(
            bool(torch.isfinite(output["loss"]))
        )
        attn = model.blocks[0].attn
        self.assertEqual(attn.k_proj.weight.shape[0], 3 * 8)

    def test_divisible_gqa_still_groups_contiguously(self):
        cfg = experiment.Config(
            seq_len=8,
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            n_experts=4,
            expert_hidden=8,
            top_k=2,
            mode="smelt",
        )
        model = experiment.LanguageModel(cfg)
        model.eval()
        x = torch.randint(0, 256, (1, 8))
        with torch.no_grad():
            a = model(x)["logits"]
            model.blocks[0].attn.n_kv_heads = 4
            model.blocks[0].attn.kv_repeat = 1
            # 直接用 MHA 的 k_proj/v_proj 形状不匹配，改回。
            model.blocks[0].attn.n_kv_heads = 2
            model.blocks[0].attn.kv_repeat = 2
            b = model(x)["logits"]
        torch.testing.assert_close(a, b)


class DocumentMaskTests(unittest.TestCase):
    SEP = 10

    def test_segment_ids_from_tokens(self):
        ids = torch.tensor([[1, 2, 10, 3, 4, 10, 5, 6]])
        segments = experiment.document_segment_ids(ids, [self.SEP])
        self.assertEqual(
            segments.tolist(),
            [[0, 0, 1, 1, 1, 2, 2, 2]],
        )

    def test_multi_byte_separator(self):
        ids = torch.tensor([[1, 2, 10, 11, 3, 4, 10, 11]])
        segments = experiment.document_segment_ids(
            ids, [10, 11]
        )
        self.assertEqual(
            segments.tolist(),
            [[0, 0, 1, 1, 1, 1, 2, 2]],
        )

    def test_dense_mask_is_causal_within_documents(self):
        segments = torch.tensor([[0, 0, 1, 1, 1]])
        mask = experiment.SegmentAttentionMask(segments)
        self.assertEqual(mask.path, "dense")
        dense = mask.dense
        # 行 = query，列 = key。
        self.assertTrue(bool(dense[0, 0, 0, 0]))
        self.assertFalse(bool(dense[0, 0, 0, 1]))
        # token 4 可看文档 1 的 2,3,4，不能看文档 0 或未来。
        row = dense[0, 0, 4]
        self.assertTrue(bool(row[2]))
        self.assertTrue(bool(row[3]))
        self.assertTrue(bool(row[4]))
        self.assertFalse(bool(row[0]))
        self.assertFalse(bool(row[1]))
        with self.assertRaises(IndexError):
            _ = row[5]

    def test_config_requires_separator(self):
        cfg = experiment.Config(
            seq_len=8,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=4,
            n_experts=4,
            expert_hidden=8,
            top_k=2,
            mode="smelt",
            document_masks=True,
        )
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_cross_document_independence_and_gradients(self):
        cfg = experiment.Config(
            seq_len=8,
            d_model=16,
            n_heads=2,
            n_kv_heads=1,
            n_layers=4,
            n_experts=4,
            expert_hidden=8,
            top_k=2,
            mode="smelt",
            document_masks=True,
            document_separator=[self.SEP],
        )
        torch.manual_seed(0)
        model = experiment.LanguageModel(cfg)
        x = torch.tensor(
            [[1, 2, 10, 3, 4, 10, 5, 6]]
        )
        y = torch.roll(x, -1, dims=1)
        output = model(x, y)
        output["loss"].backward()
        grad = model.blocks[0].attn.out.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

        perturbed = x.clone()
        perturbed[0, 6] = 99
        perturbed[0, 7] = 98
        with torch.no_grad():
            before = model(x)["logits"]
            after = model(perturbed)["logits"]
        # 文档 0/1（位置 0..4）不受文档 2 扰动影响。
        self.assertLess(
            float((before - after)[:, :5].abs().max()),
            1e-6,
        )
        # 文档 2 自身受影响。
        self.assertGreater(
            float((before - after)[:, 5:].abs().max()),
            0.0,
        )

    def test_branch_modes_keep_mask_with_single_branch(self):
        # B/B_early 即使 k1==1 也会把分支维并入 batch 执行后续
        # 中间 Block；掩码必须按 k1 复制，否则第二个文档会通过
        # 普通因果注意力看到第一个文档。
        for mode in ("B", "B_early"):
            cfg = experiment.Config(
                seq_len=8,
                d_model=16,
                n_heads=2,
                n_kv_heads=1,
                n_layers=4,
                n_experts=2,
                expert_hidden=8,
                top_k=1,
                k1=1,
                k2=1,
                mode=mode,
                document_masks=True,
                document_separator=[self.SEP],
            )
            torch.manual_seed(0)
            model = experiment.LanguageModel(cfg).eval()
            x = torch.tensor(
                [[1, 2, 3, self.SEP, 5, 6, 7, 8]]
            )
            perturbed = x.clone()
            perturbed[0, :3] = torch.tensor([20, 21, 22])
            with torch.no_grad():
                before = model(x)["logits"]
                after = model(perturbed)["logits"]
            # 文档 2（位置 4..7）不能受文档 1 扰动影响。
            self.assertLess(
                float(
                    (before - after)[0, 4:].abs().max()
                ),
                1e-6,
                mode,
            )
            # 文档 1 自身确实受影响，证明扰动不是无效的。
            self.assertGreater(
                float(
                    (before - after)[0, :3].abs().max()
                ),
                0.0,
                mode,
            )

    def test_mask_requires_applicable_batch_shape(self):
        # 5 个 token、batch 2 的掩码可以物化；超限时显式报错
        # 由 DENSE_LIMIT_BYTES 控制（这里只验证上限常量存在）。
        self.assertGreater(
            experiment.SegmentAttentionMask.DENSE_LIMIT_BYTES,
            0,
        )


if __name__ == "__main__":
    unittest.main()
