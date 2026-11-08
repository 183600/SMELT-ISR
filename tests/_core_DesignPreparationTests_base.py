from _core_support import *
from _core_support import _load_experiment, _tiny_config

class _DesignPreparationTestsBase(unittest.TestCase):
    """先验功效、随机化区组顺序与设计计划核对。"""
    def _payload(self):
        return {
            "study_id": "smelt-isr-design-test",
            "frozen_commit": "a" * 40,
            "primary_metric": "val_loss",
            "common_step": 100,
            "min_effect": 0.01,
            "direction": "decrease",
            "baseline_mode": "smelt",
            "comparisons": [
                ["smelt", "A"],
                ["smelt", "B"],
            ],
            "seeds": [42, 43, 44],
            "budget_match": "smelt",
            "steps": 200,
            "data": {
                "sha256": "b" * 64,
                "split_policy": "contiguous",
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            },
        }
    def _args(self, **overrides):
        values = dict(
            primary_metric=None,
            hypothesis_direction=None,
            hypothesis_min_effect=None,
            hypothesis_baseline=None,
            common_step=None,
            budget_match=None,
            equivalence_margin=0.01,
            steps=200,
            seeds=None,
            seed=42,
            mode="A",
            statistical_baseline=False,
            resume=None,
            evaluate_test=False,
            run_order=None,
            run_order_seed=None,
        )
        values.update(overrides)
        return argparse.Namespace(**values)
    def _write_pilot_run(
        self,
        root: Path,
        mode: str,
        seed: int,
        value: float,
        *,
        failed: bool = False,
        balance_coef: float = 0.01,
    ) -> Path:
        run_dir = root / f"{mode}_seed{seed}"
        run_dir.mkdir()
        (run_dir / "config.json").write_text(
            json.dumps({
                "config": {
                    "mode": mode,
                    "balance_coef": balance_coef,
                },
                "args": {"seed": seed},
            }),
            encoding="utf-8",
        )
        final = {
            "step": "final",
            "val_loss": value,
            "numerical_integrity_ok": True,
        }
        if failed:
            final["numerical_stop"] = True
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in (
                    {"step": 100, "val_loss": value},
                    final,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return run_dir
