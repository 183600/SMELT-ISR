from _core_support import *
from _core_support import _load_experiment, _tiny_config

class RoutingTests(unittest.TestCase):
    def test_chance_levels_are_reported(self):
        a = torch.tensor([[0, 1], [2, 3], [0, 2]])
        b = torch.tensor([[1, 2], [2, 0], [1, 3]])
        metrics = experiment._routing_set_metrics(
            a, b, 4
        )
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(
            metrics["jaccard_chance_uniform"], 0.0
        )
        self.assertGreaterEqual(
            metrics["jaccard_chance_marginal"], 0.0
        )
        self.assertIsNotNone(
            metrics["jaccard_lift_over_marginal"]
        )

    def test_routing_set_bool_does_not_mutate_or_fail_on_views(self):
        indices = torch.tensor([[-1, 99]], dtype=torch.int64)
        experiment._routing_set_bool(indices, 5)
        self.assertEqual(indices.tolist(), [[-1, 99]])
        # stride-0 的 expand 视图不能做原地 clamp_；分支创建时
        # 的 a_expanded 正是这种布局。
        base = torch.tensor([[0, 1]], dtype=torch.int64)
        expanded = base.expand(2, 2)
        flags = experiment._routing_set_bool(expanded, 4)
        self.assertEqual(flags.shape, (2, 4))
        self.assertEqual(base.tolist(), [[0, 1]])

    def test_branch_creation_overlap_aligns_each_branch_slot(self):
        cfg = experiment.Config(
            mode="B",
            n_experts=4,
            d_model=4,
            n_heads=1,
            n_layers=4,
            seq_len=1,
            expert_hidden=2,
            top_k=2,
            k1=2,
            k2=2,
        )
        probe = experiment.RoutingProbe(cfg, batch_size=2)
        first = {
            "indices": torch.tensor([[[0, 1]], [[2, 3]]]),
            "input_shape": [2, 1, 4],
        }
        second = {
            "indices": torch.tensor(
                [[0, 2], [1, 3], [2, 0], [3, 1]]
            ),
            "input_shape": [4, 1, 4],
        }
        overlap = probe._pair_overlap(
            first, second, cfg.middle_start
        )
        # 每个分支只与它自己携带的第一轮专家比较：
        # {0}->{0,2}, {1}->{1,3}, {2}->{2,0}, {3}->{3,1}。
        self.assertAlmostEqual(
            overlap["branch_level"][
                "second_round_jaccard_mean"
            ],
            0.5,
        )
        self.assertAlmostEqual(
            overlap["branch_level"]["first_round_recall_mean"],
            1.0,
        )
        self.assertEqual(overlap["branch_self_reuse_share"], 1.0)

    def test_routing_indices_keep_large_expert_ids(self):
        class _FakeMoE:
            n_experts = 40000

        cfg = experiment.Config(
            mode="base",
            n_experts=40000,
            d_model=2,
            n_heads=1,
            n_layers=4,
            seq_len=1,
            expert_hidden=1,
            top_k=1,
            k1=1,
            k2=1,
        )
        probe = experiment.RoutingProbe(cfg, batch_size=1)
        module = _FakeMoE()
        probe._block_index[id(module)] = 0
        logits = torch.zeros(1, 40000)
        logits[0, 39999] = 10.0
        probe.record(
            module,
            torch.zeros(1, 1, 2),
            torch.tensor([[0, 39999]]),
            torch.tensor([[0.5, 0.5]]),
            logits,
        )
        probe._materialize_events()
        self.assertEqual(
            probe.events[0]["indices"].tolist(),
            [[0, 39999]],
        )

    def test_routing_shuffle_seed_uses_stable_module_index(self):
        first = experiment._routing_shuffle_seed(7, 0)
        self.assertEqual(
            first, experiment._routing_shuffle_seed(7, 0)
        )
        self.assertEqual(
            experiment._routing_shuffle_seed(7, 1),
            experiment._routing_shuffle_seed(7, 1),
        )
        self.assertNotEqual(
            first, experiment._routing_shuffle_seed(8, 0)
        )
        self.assertNotEqual(
            first, experiment._routing_shuffle_seed(7, 1)
        )

    def test_routing_shuffle_is_distinct_and_advances(self):
        generator = torch.Generator().manual_seed(0)
        first = experiment._routing_shuffle_indices(
            4, torch.Size([64, 2]), generator
        )
        second = experiment._routing_shuffle_indices(
            4, torch.Size([64, 2]), generator
        )
        # 每个 token 的 top-k 必须是无放回的不同专家。
        self.assertTrue(
            bool(
                (first[:, 0] != first[:, 1]).all()
            )
        )
        self.assertTrue(
            bool(
                (second[:, 0] != second[:, 1]).all()
            )
        )
        # 第二次调用不能复用第一次的随机 assignment。
        self.assertFalse(torch.equal(first, second))

    def test_routing_shuffle_reproducible_across_fresh_models(self):
        class _FixedEvalDataset:
            def __init__(self, seq_len):
                self.seq_len = seq_len

            def fixed_eval_batch(
                self,
                split,
                batch_index,
                batches,
                batch_size,
                device,
            ):
                generator = torch.Generator().manual_seed(0)
                x = torch.randint(
                    0,
                    256,
                    (batch_size, self.seq_len),
                    generator=generator,
                )
                y = torch.randint(
                    0,
                    256,
                    (batch_size, self.seq_len),
                    generator=generator,
                )
                return x, y

        cfg = _tiny_config(mode="A", seq_len=8, n_layers=4)
        experiment.set_seed(123)
        first_model = experiment.LanguageModel(cfg)
        experiment.set_seed(123)
        second_model = experiment.LanguageModel(cfg)
        dataset = _FixedEvalDataset(cfg.seq_len)

        first = experiment.analyze_routing_interventions(
            first_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=7,
        )
        second = experiment.analyze_routing_interventions(
            second_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=7,
        )
        self.assertEqual(
            first["interventions"]["routing_shuffle"]["loss"],
            second["interventions"]["routing_shuffle"]["loss"],
        )
        self.assertEqual(
            first["interventions"]["routing_shuffle"][
                "token_accuracy"
            ],
            second["interventions"]["routing_shuffle"][
                "token_accuracy"
            ],
        )

        other_seed = experiment.analyze_routing_interventions(
            first_model,
            dataset,
            batch_size=2,
            batches=1,
            device=torch.device("cpu"),
            amp_dtype=None,
            seed=8,
        )
        self.assertNotEqual(
            first["interventions"]["routing_shuffle"]["loss"],
            other_seed["interventions"]["routing_shuffle"]["loss"],
        )
