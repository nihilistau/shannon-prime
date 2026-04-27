import sys
sys.path.insert(0, r'D:\F\shannon-prime-repos\shannon-prime\tools')
import numpy as np
from sp_regime_analysis import (detect_transition, detect_transition_gl,
                                 load_k_vectors, sqfree_pad_dim)

models = ['dolphin1b', 'phi3mini', 'qwen3_8b_q3', 'qwen3_8b_q8', 'qwen36_27b']
base   = r'D:\F\shannon-prime-repos\shannon-prime\logs\phase12'

print(f"{'Model':22s}  {'head_dim':>8s}  {'analysis':>8s}  {'std':>4s}  {'gl':>4s}  {'lookahead':>9s}")
print("-" * 70)
for name in models:
    kv, *_ = load_k_vectors(rf'{base}\kv_{name}.npz')
    head_dim     = kv.shape[3]
    analysis_dim = sqfree_pad_dim(head_dim)

    std_L, std_info = detect_transition(kv, analysis_dim, use_sqfree=True)
    gl_L,  gl_info  = detect_transition_gl(kv, analysis_dim, use_sqfree=True, alpha=0.25)

    la = gl_info['lookahead_layers']
    print(f"{name:22s}  {head_dim:8d}  {analysis_dim:8d}  {std_L:4d}  {gl_L:4d}  {la:+9d}")
