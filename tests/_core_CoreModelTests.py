from _core_support import *
from _core_support import _load_experiment, _tiny_config

class CoreModelTests(unittest.TestCase):
    def test_all_modes_forward_backward(self):
        for mode in (
            "base",
            "smelt",
            "A",
            "local_base",
            "local_base_k2",
            "B",
            "B_early",
            "dense",
        ):
            cfg = _tiny_config(mode=mode)
            model = experiment.LanguageModel(cfg)
            x = torch.randint(0, 256, (2, 16))
            y = torch.randint(0, 256, (2, 16))
            output = model(x, y)
            self.assertTrue(
                math.isfinite(float(output["loss"])),
                mode,
            )
            output["loss"].backward()
            if cfg.mode != "dense":
                grad = model.blocks[0].moe.router.weight.grad
                self.assertIsNotNone(grad, mode)
                self.assertTrue(
                    bool(torch.isfinite(grad).all()), mode
                )

    def test_top1_router_receives_lm_gradient(self):
        cfg = _tiny_config(
            top_k=1,
            k1=1,
            k2=1,
            mode="base",
            balance_coef=0.0,
        )
        model = experiment.LanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        output = model(x, y)
        output["loss"].backward()
        grad = model.blocks[0].moe.router.weight.grad
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_k1_one_equivalences(self):
        common = dict(
            seq_len=16,
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            n_layers=4,
            n_experts=4,
            expert_hidden=16,
        )
        x = torch.randint(0, 256, (2, 16))
        for left, right, cfg in (
            ("smelt", "B", dict(top_k=1, k1=1, k2=1)),
            ("local_base", "A", dict(k1=1, k2=2, top_k=2)),
        ):
            first = experiment.LanguageModel(
                experiment.Config(**common, **cfg, mode=left)
            ).eval()
            second = experiment.LanguageModel(
                experiment.Config(**common, **cfg, mode=right)
            ).eval()
            second.load_state_dict(first.state_dict())
            with torch.no_grad():
                a = first(x)["logits"]
                b = second(x)["logits"]
            torch.testing.assert_close(
                a, b, rtol=1e-5, atol=1e-6
            )

    def test_gqa_forward_backward(self):
        cfg = _tiny_config(n_kv_heads=1)
        model = experiment.LanguageModel(cfg)
        x = torch.randint(0, 256, (2, 16))
        y = torch.randint(0, 256, (2, 16))
        output = model(x, y)
        output["loss"].backward()
        self.assertTrue(
            math.isfinite(float(output["loss"]))
        )
        self.assertEqual(
            model.blocks[0].attn.k_proj.weight.shape[0],
            1 * (32 // 4),
        )

    def test_delayed_merge_gate_is_applied_once(self):
        # A 的第一轮分支若先乘 gate，最终跨分支收缩又乘同一组
        # gate，会让专家贡献变成 p_i²·expert。这里用桩替换 MoE，
        # 只保留门控/合并路径：正确结果应为 x + Σ p_i·expert_i。
        cfg = _tiny_config(mode="A", k1=2, k2=2)
        block = experiment.TransformerBlock(cfg)
        block.attention_step = (
            lambda x, scale=None, attn_mask=None: x
        )
        block.norm_moe = torch.nn.Identity()

        def unmerged(x, top_k):
            if x.dim() == 3:
                experts = torch.tensor(
                    [[[[2.0, 0.0], [0.0, 4.0]]]]
                )
                gates = torch.tensor([[[0.75, 0.25]]])
                return (
                    experts,
                    gates,
                    None,
                    torch.zeros(()),
                )
            experts = torch.zeros(
                *x.shape[:-1], 2, x.size(-1)
            )
            gates = torch.full((*x.shape[:-1], 2), 0.5)
            return experts, gates, None, torch.zeros(())

        def merge(
            expert_outputs,
            top_weights,
            out_shape,
            residual=None,
            residual_scale=1.0,
        ):
            return residual

        block.moe.unmerged = unmerged
        block.moe.merge = merge
        output, _, _ = block.local_two_stage(
            torch.zeros(1, 1, 2),
            2,
            2,
            delayed=True,
        )
        torch.testing.assert_close(
            output[0, 0],
            torch.tensor([1.5, 1.0]),
        )

    def test_loop_residual_scaling(self):
        cfg = _tiny_config(residual_scaling="paper")
        self.assertAlmostEqual(
            cfg.loop_residual_scale, 0.5
        )
        global_cfg = _tiny_config(
            residual_scaling="global"
        )
        self.assertAlmostEqual(
            global_cfg.loop_residual_scale, 1.0
        )
