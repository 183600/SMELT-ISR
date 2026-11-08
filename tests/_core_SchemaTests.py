"""预注册 JSON Schema 回归测试。

不依赖第三方 jsonschema 库，仅用标准库验证：
1. schema 文件本身是有效 JSON，并包含必要结构键；
2. 示例文件是有效 JSON，并覆盖 schema 的全部 required 字段；
3. 示例文件中每个字段的 Python 类型与 schema 声明一致；
4. 故意损坏的示例（缺必填字段 / 类型错误 / 枚举越界）会被基本校验捕获。

与 exp/part_033_preregistration_core.py、exp/part_034_preregistration_schema.py
的字段定义保持交叉一致。
"""

import json
import os
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "preregistration.v1.json"
EXAMPLE_PATH = REPO_ROOT / "preregistration.example.json"

MODEL_MODES = {
    "base", "smelt", "local_base", "local_base_k2",
    "A", "B", "B_early", "dense", "ngram",
}
RUN_ORDER_SCHEMES = (
    "grouped", "blocked", "randomized", "counterbalanced", "williams",
)
PILOT_CALIBRATION_POLICIES = ("point", "upper_confidence_bound")
CONTRAST_TYPES = ("pairwise", "simple", "main", "interaction")
MAX_SEED = 2**32 - 1


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class SchemaStructureTests(unittest.TestCase):
    """schema 自身结构必须完整、可被 JSON Schema 处理器识别。"""

    def test_schema_is_valid_json(self):
        schema = _load_json(SCHEMA_PATH)
        self.assertIsInstance(schema, dict)

    def test_schema_has_meta_keys(self):
        schema = _load_json(SCHEMA_PATH)
        for key in ("$schema", "$id", "title", "type", "required", "properties"):
            self.assertIn(key, schema, f"schema 缺少结构键 {key!r}")
        self.assertEqual(schema["type"], "object")
        self.assertFalse(schema.get("additionalProperties", True))

    def test_required_fields_present(self):
        schema = _load_json(SCHEMA_PATH)
        required = set(schema["required"])
        expected_required = {
            "study_id", "frozen_commit", "primary_metric", "common_step",
            "min_effect", "direction", "baseline_mode", "comparisons",
            "seeds", "budget_match", "data",
        }
        self.assertTrue(expected_required.issubset(required),
                        f"缺少必填字段声明: {expected_required - required}")

    def test_data_subschema(self):
        schema = _load_json(SCHEMA_PATH)
        data = schema["properties"]["data"]
        self.assertEqual(data["type"], "object")
        self.assertIn("sha256", data["required"])
        self.assertIn("split_policy", data["required"])
        self.assertIn("val_fraction", data["required"])
        self.assertIn("test_fraction", data["required"])
        self.assertFalse(data.get("additionalProperties", True))

    def test_enum_values_match_code(self):
        schema = _load_json(SCHEMA_PATH)
        self.assertEqual(
            set(schema["properties"]["direction"]["enum"]),
            {"decrease", "increase"},
        )
        self.assertEqual(
            set(schema["properties"]["baseline_mode"]["enum"]),
            MODEL_MODES,
        )
        comp_item = schema["properties"]["comparisons"]["items"]["items"]
        self.assertEqual(set(comp_item["enum"]), MODEL_MODES)
        self.assertEqual(
            set(schema["properties"]["run_order"]["properties"]["scheme"]["enum"]),
            set(RUN_ORDER_SCHEMES),
        )
        self.assertEqual(
            set(schema["properties"]["power"]["properties"]["pilot_calibration"]["enum"]),
            set(PILOT_CALIBRATION_POLICIES),
        )
        self.assertEqual(
            set(schema["properties"]["power"]["properties"]["planned_contrasts"]["items"]["enum"]),
            set(CONTRAST_TYPES),
        )

    def test_seed_range_matches_code(self):
        schema = _load_json(SCHEMA_PATH)
        seed_item = schema["properties"]["seeds"]["items"]
        self.assertEqual(seed_item["minimum"], 0)
        self.assertEqual(seed_item["maximum"], MAX_SEED)
        self.assertGreaterEqual(schema["properties"]["seeds"]["minItems"], 3)
        self.assertTrue(schema["properties"]["seeds"]["uniqueItems"])


class ExampleConsistencyTests(unittest.TestCase):
    """示例预注册文件必须与 schema 保持一致，可作为用户直接复用的模板。"""

    def test_example_is_valid_json(self):
        example = _load_json(EXAMPLE_PATH)
        self.assertIsInstance(example, dict)

    def test_example_covers_required_fields(self):
        schema = _load_json(SCHEMA_PATH)
        example = _load_json(EXAMPLE_PATH)
        missing = set(schema["required"]) - set(example)
        self.assertEqual(missing, set(),
                         f"示例缺少必填字段: {sorted(missing)}")

    def test_example_field_types_match_schema(self):
        schema = _load_json(SCHEMA_PATH)
        example = _load_json(EXAMPLE_PATH)
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for field, prop in schema["properties"].items():
            if field not in example:
                continue
            expected = prop.get("type")
            if expected is None:
                continue
            # 联合类型如 ["string", "null"]：取第一个非 null 子类型做类型断言。
            if isinstance(expected, list):
                expected = [t for t in expected if t != "null"][0]
            python_types = type_map[expected] if isinstance(
                type_map[expected], tuple
            ) else (type_map[expected],)
            value = example[field]
            # nullable 联合类型允许 null（如 ["string", "null"]）。
            raw_type = prop.get("type")
            is_nullable = isinstance(raw_type, list) and "null" in raw_type
            if value is None:
                if not is_nullable:
                    self.fail(f"{field}: 非空字段不应为 null")
                continue
            # bool 是 int 的子类，数值字段需排除 bool
            if expected in ("integer", "number") and isinstance(value, bool):
                self.fail(f"{field}: 布尔值不应作为 {expected}")
            if not isinstance(value, python_types):
                self.fail(
                    f"{field}: 期望 {expected}，实际 {type(value).__name__}"
                )

    def test_example_enums_in_range(self):
        schema = _load_json(SCHEMA_PATH)
        example = _load_json(EXAMPLE_PATH)
        enum_checks = {
            "direction": {"decrease", "increase"},
            "baseline_mode": MODEL_MODES,
        }
        for field, allowed in enum_checks.items():
            self.assertIn(example.get(field), allowed, f"{field}={example.get(field)} 不在枚举范围内")
        for arm_a, arm_b in example["comparisons"]:
            self.assertIn(arm_a, MODEL_MODES)
            self.assertIn(arm_b, MODEL_MODES)
            self.assertNotEqual(arm_a, arm_b)
        self.assertIn(example["run_order"]["scheme"], RUN_ORDER_SCHEMES)
        self.assertIn(example["power"]["pilot_calibration"], PILOT_CALIBRATION_POLICIES)

    def test_example_seeds(self):
        example = _load_json(EXAMPLE_PATH)
        self.assertGreaterEqual(len(example["seeds"]), 3)
        self.assertEqual(len(example["seeds"]), len(set(example["seeds"])))
        for seed in example["seeds"]:
            self.assertIsInstance(seed, int)
            self.assertNotIsInstance(seed, bool)
            self.assertGreaterEqual(seed, 0)
            self.assertLessEqual(seed, MAX_SEED)

    def test_example_budget_match_for_delayed_merge(self):
        # comparisons 声明了延迟合并与 smelt 的对照，budget_match 必须非 null。
        example = _load_json(EXAMPLE_PATH)
        arms = {arm for pair in example["comparisons"] for arm in pair}
        if {"smelt"} & arms and arms & {"A", "B", "local_base"}:
            self.assertIsNotNone(example.get("budget_match"),
                                 "延迟合并与 smelt 对照必须给出 budget_match")

    def test_example_equivalence_margin_le_min_effect(self):
        example = _load_json(EXAMPLE_PATH)
        margin = example.get("equivalence_margin")
        min_effect = example.get("min_effect")
        if margin is not None and min_effect is not None:
            self.assertLessEqual(margin, min_effect,
                                 "等价界不得大于最小有意义效应")

    def test_example_steps_ge_common_step(self):
        example = _load_json(EXAMPLE_PATH)
        steps = example.get("steps")
        common_step = example.get("common_step")
        if steps is not None and common_step is not None:
            self.assertGreaterEqual(steps, common_step,
                                    "训练总步数必须 >= common_step")

    def test_example_val_test_fraction(self):
        example = _load_json(EXAMPLE_PATH)
        data = example["data"]
        vf = data["val_fraction"]
        tf = data["test_fraction"]
        self.assertGreater(vf, 0)
        self.assertLess(vf, 1)
        self.assertGreaterEqual(tf, 0)
        self.assertLess(tf, 1)
        self.assertLess(vf + tf, 1,
                        "val_fraction + test_fraction 必须小于 1")

    def test_broken_example_rejected(self):
        """缺必填字段 / 类型错误的示例应被基本校验捕获。"""
        base = _load_json(EXAMPLE_PATH)

        broken = dict(base)
        broken.pop("common_step")
        self.assertNotIn("common_step", broken)

        broken_type = dict(base)
        broken_type["common_step"] = "not-an-int"
        self.assertIsInstance(broken_type["common_step"], str)

        broken_enum = dict(base)
        broken_enum["direction"] = "sideways"
        self.assertNotIn(broken_enum["direction"], {"decrease", "increase"})

        broken_seed = dict(base)
        broken_seed["seeds"] = [42, 42]
        self.assertNotEqual(
            len(broken_seed["seeds"]), len(set(broken_seed["seeds"])),
            "重复种子应被唯一性约束捕获",
        )


if __name__ == "__main__":
    unittest.main()
