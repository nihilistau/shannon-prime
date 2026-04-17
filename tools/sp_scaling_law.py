# Shannon-Prime VHT2: Exact Spectral KV Cache Compression
# Copyright (C) 2026 Ray Daniels. All Rights Reserved.
#
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial license available — contact raydaniels@gmail.com
#
# See LICENSE in the project root for full terms.

"""
K-corr → PPL scaling law: design rule for KV cache compression.

Empirical law fit across 9 configurations spanning (Dolphin 1B Q8,
Qwen3-8B Q8, Qwen3-8B Q3), 4 orders of magnitude of PPL ratio,
and 8× parameter range:

    log(PPL / PPL_base) ≈ K · (1 − K_corr)² / (params^β · bits^γ)

    K ≈ 4700,  α = 2 (quadratic),  β ≈ 1.1,  γ ≈ 1.5

Usage as a pre-bench filter:
    from sp_scaling_law import predicted_ppl_ratio, is_pareto_viable

    # Will this config work on my model?
    ratio = predicted_ppl_ratio(k_corr=0.988, params_b=8.0, bits=8)
    print(f"Predicted PPL impact: +{(ratio - 1)*100:.2f}%")

    # Skip configs that blow the budget
    if not is_pareto_viable(k_corr=0.972, params_b=1.0, bits=8, budget=0.03):
        print("Not worth benching — predicted PPL hit > 3%")

Usage as a target calculator:
    from sp_scaling_law import min_k_corr_for_budget

    floor = min_k_corr_for_budget(params_b=70.0, bits=8, budget=0.03)
    print(f"Llama-70B Q8: need K_corr ≥ {floor:.4f} for 3% PPL budget")
"""

import math

# Empirical constants — fit from 9 datapoints, ±16% (see Appendix C of
# "The KV Cache Is a View" v2). Do not tune without new measurements.
K_COEFF = 4700.0
ALPHA   = 2.0    # Exponent on (1 − K_corr): quadratic (K·Q·V bilinearity)
BETA    = 1.1    # Exponent on params: sub-linear (head averaging)
GAMMA   = 1.5    # Exponent on bits: super-linear (W-matrix amplification)


def predicted_ppl_ratio(k_corr: float, params_b: float, bits: int) -> float:
    """
    Predict PPL / PPL_baseline from K reconstruction fidelity.

    Args:
        k_corr:   Mean Pearson correlation between original and reconstructed
                  K vectors (across heads). Range [0, 1].
        params_b: Model parameter count in billions (e.g. 8.0 for Qwen3-8B).
        bits:     Effective weight precision (8 for Q8, 4 for Q4, 16 for bf16).

    Returns:
        Predicted PPL ratio (1.0 = no degradation, 1.03 = +3%).
    """
    if k_corr >= 1.0:
        return 1.0
    err = 1.0 - k_corr
    log_ratio = K_COEFF * (err ** ALPHA) / (params_b ** BETA * bits ** GAMMA)
    return math.exp(log_ratio)


def predicted_ppl_pct(k_corr: float, params_b: float, bits: int) -> float:
    """Convenience: predicted PPL impact as a percentage (e.g. 2.5 for +2.5%)."""
    return (predicted_ppl_ratio(k_corr, params_b, bits) - 1.0) * 100.0


def is_pareto_viable(k_corr: float, params_b: float, bits: int,
                     budget: float = 0.03) -> bool:
    """
    Pre-bench filter: is this config worth running?

    Args:
        budget: Maximum acceptable ΔPPL/PPL (default 0.03 = 3%).

    Returns:
        True if predicted PPL impact is within budget.
    """
    return predicted_ppl_ratio(k_corr, params_b, bits) <= 1.0 + budget


def min_k_corr_for_budget(params_b: float, bits: int,
                          budget: float = 0.03) -> float:
    """
    Compute the minimum K_corr that stays within the PPL budget.

    Args:
        params_b: Model size in billions.
        bits:     Weight precision.
        budget:   Maximum ΔPPL/PPL (default 3%).

    Returns:
        Minimum K_corr (e.g. 0.988 for Dolphin 1B Q8 at 3% budget).
    """
    log_budget = math.log(1.0 + budget)
    denominator = params_b ** BETA * bits ** GAMMA
    err_sq = log_budget * denominator / K_COEFF
    if err_sq <= 0:
        return 1.0
    err = math.sqrt(err_sq)
    return max(0.0, 1.0 - err)


def safe_k_corr_table(budget: float = 0.03) -> dict:
    """
    Reference table of safe K_corr floors for common models.

    Returns dict mapping model description to minimum K_corr.
    """
    models = [
        ("Dolphin 1B Q8",      1.0,   8),
        ("Qwen3-8B Q8",        8.0,   8),
        ("Qwen3-8B Q3",        8.0,   3),
        ("Llama-70B Q8",      70.0,   8),
        ("Llama-70B Q4",      70.0,   4),
        ("Wan 2.2 14B bf16",  14.0,  16),
    ]
    return {
        name: min_k_corr_for_budget(p, b, budget)
        for name, p, b in models
    }


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    """Print the safe K_corr floor table."""
    import sys
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
    table = safe_k_corr_table(budget)
    print(f"Safe K_corr floors for {budget*100:.0f}% PPL budget:")
    print(f"{'Model':<25} {'Min K_corr':>12}")
    print("-" * 40)
    for name, floor in table.items():
        print(f"{name:<25} {floor:>12.4f}")


if __name__ == "__main__":
    main()