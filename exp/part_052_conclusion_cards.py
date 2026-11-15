from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CARD_SCHEMA = "smelt-isr/conclusion-card/v1"
_INDEX_NAME = "index.json"
_MAX_SLUG_LENGTH = 96


def _card_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    slug = slug.strip("_") or "contrast"
    return slug[:_MAX_SLUG_LENGTH]


def _card_rq_map(analysis: Dict[str, object]) -> Dict[str, List[str]]:
    """mode → 研究问题标签列表（来自各运行登记的 --rq）。"""
    registry = analysis.get("hypothesis_registry")
    mapping: Dict[str, List[str]] = {}
    if not isinstance(registry, dict):
        return mapping
    cells = registry.get("cells")
    if not isinstance(cells, list):
        return mapping
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        mode = cell.get("mode")
        questions = cell.get("research_questions")
        if mode is None or not isinstance(questions, list):
            continue
        labels = [str(question) for question in questions if str(question).strip()]
        if labels:
            mapping[str(mode)] = labels
    return mapping


def _card_verdict(
    contrast: Dict[str, object],
) -> Tuple[str, str, List[str]]:
    """
    由既有判定字段合成结论卡片的机器可读裁决。

    输入全部来自 ``--compare`` 已经算好的 Holm 显著性、TOST
    等价性、脆弱性、效率与符号翻转字段；这里不做任何新的统计
    计算，只按固定决策树组合，保证卡片可以被机器复核。
    """
    caveats: List[str] = []
    significant = bool(contrast.get("significant"))
    equivalent = bool(contrast.get("equivalent"))
    fragility = contrast.get("fragility")
    fragility_verdict = (
        str(fragility.get("verdict"))
        if isinstance(fragility, dict) and fragility.get("verdict")
        else None
    )
    cost_verdict = contrast.get("cost_verdict")
    if contrast.get("differential_missingness"):
        caveats.append(
            "两臂缺失运行数不同：完整案例分析建立在不同运行子集上"
        )
    power = contrast.get("power_at_effect")
    if isinstance(power, (int, float)) and float(power) < 0.5:
        caveats.append(
            f"观测效应量自身的功效仅 {_format_finite(power, '.0%')}，"
            "结论对种子波动敏感"
        )
    if significant:
        if fragility_verdict == "fragile":
            caveats.append("删除个别种子后显著性消失（fragility=fragile）")
            return (
                "significant_but_fragile",
                "统计显著但结论脆弱：显著性依赖个别随机种子，"
                "增加独立种子或调查异常运行前不得作为确证结论",
                caveats,
            )
        robust_possible = contrast.get(
            "significant_robust_possible", True
        )
        randomization_significant = contrast.get(
            "randomization_significant"
        )
        if (
            robust_possible is not False
            and randomization_significant is False
        ):
            caveats.append(
                "精确符号翻转检验未达显著：显著性依赖配对差值"
                "对称性假设"
            )
            return (
                "significant_assumption_dependent",
                "统计显著但依赖分布假设：t 检验显著而符号翻转"
                "不显著，结论应与逐种子差值分布一起解读",
                caveats,
            )
        code = "supported"
        text = "显著且未见脆弱性/假设冲突标记：可作为支持证据（仍受种子数与预注册口径约束）"
        if cost_verdict == "gain_with_cost":
            code = "supported_with_cost_penalty"
            text = "统计显著但效率更低：收益需与计算/时延成本一起权衡"
            caveats.append("变体效率低于基线（cost_verdict=gain_with_cost）")
        elif cost_verdict == "dominated_regression":
            code = "dominated"
            text = "效果更差且效率更低：被基线帕累托支配"
        return code, text, caveats
    if equivalent:
        return (
            "equivalent",
            "TOST 等价性通过（Holm 校正后）：支持「无实质差异」"
            "而非「未发现差异」；该主张仍以预注册等价界为条件",
            caveats,
        )
    blocker = contrast.get("equivalence_blocker")
    if blocker == "effect_exceeds_margin":
        return (
            "inconclusive_effect_exceeds_margin",
            "不显著且点估计已超出等价界：这不是种子不够，"
            "增加种子只会让「不等价」更确定；应报告效应方向并复核设计",
            caveats,
        )
    mde = contrast.get("mde_80")
    mde_text = (
        f"80% 功效 MDE={_format_finite(mde)}"
        if isinstance(mde, (int, float))
        else "MDE 不可估"
    )
    return (
        "inconclusive_low_power",
        f"不显著且不等价：当前种子数不足以判定（{mde_text}）；"
        "「不显著」不能读成「等价」或「无差异」",
        caveats,
    )


def conclusion_card_payload(
    analysis: Dict[str, object],
    contrast: Dict[str, object],
    family: str,
    rq_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, object]:
    """把单个对比与汇总分析组装成机器可读的结论卡片。"""
    rq_map = rq_map or {}
    baseline = str(contrast.get("baseline"))
    variant = str(contrast.get("variant"))
    variant_mode = variant.split("=", 1)[-1] if "=" in variant else variant
    baseline_mode = (
        baseline.split("=", 1)[-1] if "=" in baseline else baseline
    )
    rq_labels: List[str] = []
    for mode in (variant_mode, baseline_mode):
        rq_labels.extend(rq_map.get(mode, []))
    verdict_code, verdict_text, caveats = _card_verdict(contrast)
    interval = {
        key: contrast.get(key)
        for key in (
            "ci95_low",
            "ci95_high",
            "ci_family_adjusted_low",
            "ci_family_adjusted_high",
        )
        if contrast.get(key) is not None
    }
    fragility = contrast.get("fragility")
    card: Dict[str, object] = {
        "schema": CARD_SCHEMA,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "study": {
            "metric": analysis.get("metric"),
            "equivalence_margin": analysis.get("equivalence_margin"),
            "n_runs": analysis.get("n_runs"),
        },
        "rq": sorted(set(rq_labels)) or None,
        "contrast": {
            "family": family,
            "baseline": baseline,
            "variant": variant,
            "factor": contrast.get("factor"),
            "baseline_value": contrast.get("baseline_value"),
            "variant_value": contrast.get("variant_value"),
        },
        "effect": {
            "delta": contrast.get("delta"),
            "hedges_g": contrast.get("hedges_g"),
            "hedges_gz": contrast.get("hedges_gz"),
            "cohens_dz": contrast.get("cohens_dz"),
            "direction": contrast.get("verdict"),
            "interval": interval,
        },
        "significance": {
            "test": contrast.get("test"),
            "t": contrast.get("t"),
            "df": contrast.get("df"),
            "p_value": contrast.get("p_value"),
            "p_value_holm": contrast.get("p_value_holm"),
            "holm_alpha_threshold": contrast.get(
                "holm_alpha_threshold"
            ),
            "significant": bool(contrast.get("significant")),
            "significant_robust": contrast.get(
                "significant_robust"
            ),
            "randomization_p_value_holm": contrast.get(
                "randomization_p_value_holm"
            ),
        },
        "equivalence": {
            "margin_relative": contrast.get("equivalence_margin"),
            "margin_abs": contrast.get("equivalence_margin_abs"),
            "tost_p_value_holm": contrast.get("tost_p_value_holm"),
            "equivalent": bool(contrast.get("equivalent")),
            "blocker": contrast.get("equivalence_blocker"),
            "margin_needed_relative": contrast.get(
                "equivalence_margin_needed_relative"
            ),
        },
        "power": {
            "n_pairs": contrast.get("n_pairs"),
            "n_baseline": contrast.get("n_baseline"),
            "n_variant": contrast.get("n_variant"),
            "mde_80": contrast.get("mde_80"),
            "mde_80_relative": contrast.get("mde_80_relative"),
            "power_at_effect": contrast.get("power_at_effect"),
            "seeds_to_detect": contrast.get("seeds_to_detect"),
        },
        "fragility": {
            "verdict": (
                fragility.get("verdict")
                if isinstance(fragility, dict)
                else None
            ),
            "max_leave_one_out_p_value": (
                fragility.get("max_leave_one_out_p_value")
                if isinstance(fragility, dict)
                else None
            ),
            "min_seed_deletions_to_lose_significance": (
                fragility.get("min_seed_deletions_to_lose_significance")
                if isinstance(fragility, dict)
                else None
            ),
        },
        "cost": {
            "verdict": contrast.get("cost_verdict"),
            "ratios": (
                contrast.get("cost", {}).get("ratios")
                if isinstance(contrast.get("cost"), dict)
                else None
            ),
        },
        "verdict_code": verdict_code,
        "verdict": verdict_text,
        "caveats": caveats,
    }
    bayesian = analysis.get("bayesian_analysis")
    if isinstance(bayesian, dict):
        for entry in bayesian.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("baseline")) == baseline and str(
                entry.get("variant")
            ) == variant:
                card["bayesian"] = {
                    key: entry.get(key)
                    for key in (
                        "available",
                        "log_bayes_factor_10",
                        "bayes_factor_10",
                        "evidence_category",
                        "posterior_prob_alternative",
                        "posterior_prob_superior",
                        "posterior_mean",
                        "posterior_hdi95_low",
                        "posterior_hdi95_high",
                        "prior",
                    )
                }
                break
        else:
            card["bayesian"] = None
    else:
        card["bayesian"] = None
    return card


def _card_markdown(card: Dict[str, object]) -> str:
    """把结论卡片渲染成简短的 Markdown 报告。"""
    contrast = card.get("contrast") or {}
    effect = card.get("effect") or {}
    significance = card.get("significance") or {}
    equivalence = card.get("equivalence") or {}
    power = card.get("power") or {}
    lines = [
        f"# 结论卡片：{contrast.get('baseline')} → {contrast.get('variant')}",
        "",
        f"- 判定指标：`{card.get('study', {}).get('metric')}`"
        f"（对比家族：{contrast.get('family')}，因子：{contrast.get('factor')}）",
        f"- 效应：Δ={_format_finite(effect.get('delta'))}，"
        f"Hedges g={_format_finite(effect.get('hedges_g'))}",
        f"- 显著性：p={_format_finite(significance.get('p_value'))}，"
        f"Holm p={_format_finite(significance.get('p_value_holm'))}，"
        f"significant={significance.get('significant')}",
        f"- 等价性：TOST Holm p={_format_finite(equivalence.get('tost_p_value_holm'))}，"
        f"equivalent={equivalence.get('equivalent')}"
        f"（等价界 ±{_format_finite(equivalence.get('margin_relative'))}）",
        f"- 功效：n={power.get('n_pairs')}，MDE80={_format_finite(power.get('mde_80'))}，"
        f"power@effect={_format_finite(power.get('power_at_effect'))}，"
        f"检出约需种子={power.get('seeds_to_detect')}",
        f"- 脆弱性：{(card.get('fragility') or {}).get('verdict')}",
    ]
    bayesian = card.get("bayesian")
    if isinstance(bayesian, dict) and bayesian.get("available"):
        lines.append(
            f"- 贝叶斯补充：lnBF10={_format_finite(bayesian.get('log_bayes_factor_10'), '+.2f')}，"
            f"P(δ>0|D)={_format_finite(bayesian.get('posterior_prob_superior'), '.3f')}"
            f"（{bayesian.get('evidence_category')}）"
        )
    lines += [
        "",
        f"**裁决（{card.get('verdict_code')}）**：{card.get('verdict')}",
    ]
    caveats = card.get("caveats") or []
    if caveats:
        lines.append("")
        lines.append("注意事项：")
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def write_conclusion_cards(
    out_dir: Path,
    analysis: Dict[str, object],
) -> Dict[str, object]:
    """
    为 ``--compare`` 分析的每个单因子/基线对比写一张结论卡片。

    落盘 ``<out_dir>/<rq>_<baseline>__to__<variant>.json`` 与同名
    ``.md``，并写 ``index.json`` 汇总裁决分布。混杂对比不生成
    卡片（无法归因），只在 index 中点名。函数只读取分析结果，
    不做新的统计计算；重复调用会覆盖同路径卡片（卡片是派生
    产物，以最后一次分析为准）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rq_map = _card_rq_map(analysis)
    metric = analysis.get("metric")
    written: List[Dict[str, object]] = []
    skipped_confounded: List[Dict[str, object]] = []
    verdict_counts: Dict[str, int] = {}
    families = (
        ("single_factor", analysis.get("contrasts")),
        ("baseline", analysis.get("reference_contrasts")),
    )
    for family, contrasts in families:
        if not isinstance(contrasts, list):
            continue
        for contrast in contrasts:
            if not isinstance(contrast, dict):
                continue
            if contrast.get("confounded"):
                skipped_confounded.append(
                    {
                        "family": family,
                        "baseline": contrast.get("baseline"),
                        "variant": contrast.get("variant"),
                        "differing_factors": contrast.get(
                            "differing_factors"
                        ),
                    }
                )
                continue
            card = conclusion_card_payload(
                analysis, contrast, family, rq_map
            )
            rq_prefix = (
                "_".join(sorted(set(card["rq"] or [])))
                .replace("/", "-")
                if card.get("rq")
                else "rq_unspecified"
            )
            slug = _card_slug(
                f"{card['contrast']['baseline']}__to__{card['contrast']['variant']}"
            )
            if family == "baseline":
                slug = f"{slug}__baseline_family"
            stem = f"{_card_slug(rq_prefix)}_{slug}"
            card_path = out_dir / f"{stem}.json"
            md_path = out_dir / f"{stem}.md"
            _atomic_write_json(card_path, card)
            md_path.write_text(
                _card_markdown(card), encoding="utf-8"
            )
            verdict_code = str(card.get("verdict_code"))
            verdict_counts[verdict_code] = (
                verdict_counts.get(verdict_code, 0) + 1
            )
            written.append(
                {
                    "family": family,
                    "baseline": card["contrast"]["baseline"],
                    "variant": card["contrast"]["variant"],
                    "verdict_code": verdict_code,
                    "json": card_path.name,
                    "markdown": md_path.name,
                }
            )
    index = {
        "schema": "smelt-isr/conclusion-card-index/v1",
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "metric": metric,
        "equivalence_margin": analysis.get("equivalence_margin"),
        "n_cards": len(written),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "cards": written,
        "skipped_confounded": skipped_confounded,
        "note": (
            "结论卡片由 --compare 的既有判定字段自动合成，"
            "verdict 供排序与归档使用，不替代人工审阅；"
            "significant_but_fragile / inconclusive_* 的正确反应是"
            "增加独立种子或复核设计，不是反复换种子直到显著"
        ),
    }
    index_path = out_dir / _INDEX_NAME
    _atomic_write_json(index_path, index)
    return {
        "out_dir": str(out_dir),
        "n_cards": len(written),
        "index_path": str(index_path),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "cards": written,
        "skipped_confounded": skipped_confounded,
        "index_sha256": _sha256_file(index_path),
    }
