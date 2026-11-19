#!/usr/bin/env bash
# 示例：RQ1 对比实验与全因子消融的最佳实践
# 要求：--seeds >=6, --common-step 预注册, --budget-match, --primary-metric 等价性检验
set -euo pipefail

DATA=corpus.txt
OUT=runs/rq1_comparative_ablation
SEEDS=(42 43 44 45 46 47)
COMMON_STEP=2000

mkdir -p "$OUT"

# 1) 生成设计计划（不训练）
python experiment.py --data "$DATA" --out "$OUT" \
  --seeds "${SEEDS[@]}" \
  --suite-ablation --ablation-modes A B \
  --plan-design

# 2) 执行全因子消融：local_base/A/B × balance_coef=0/0.01
# 预注册假设：延迟合并相对 smelt 降低验证损失 >=2%
python experiment.py --data "$DATA" --out "$OUT" \
  --seeds "${SEEDS[@]}" \
  --suite-ablation --ablation-modes A B \
  --common-step "$COMMON_STEP" \
  --budget-match smelt \
  --primary-metric val_loss \
  --hypothesis "延迟合并改善验证损失" \
  --hypothesis-direction decrease \
  --hypothesis-min-effect 0.02 \
  --hypothesis-baseline smelt \
  --rq RQ1 \
  --equivalence-margin 0.01

# 3) 只读汇总：受控变量检查 + Mean±Std + 显著性 + 消融覆盖
python experiment.py --compare "$OUT"/* \
  --compare-baseline smelt \
  --compare-metric val_loss_at_common_step \
  --compare-report "$OUT/comparison_analysis.json"

echo "对比分析已写入 $OUT/comparison_analysis.json"
