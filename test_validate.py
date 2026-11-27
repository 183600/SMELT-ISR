import sys
sys.path.insert(0, '.')
from exp.part_000_config import Config
cfg = Config(vocab_size=256, seq_len=128, d_model=128, n_heads=4, n_layers=8, n_experts=8, expert_hidden=256, top_k=2, k1=2, k2=2, mode="B")
cfg.validate()
print("Validation passed")