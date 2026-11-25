from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed"

rng = np.random.default_rng(2027)
rows = []

studies = [
    (
        "ablation",
        "casr_v2_ablation.csv",
        ["failure_rate"],
        [
            "casr_3",
            "casr_0",
            "v2_fixed_margin",
            "v2_no_recovery",
        ],
    ),
    (
        "scalability",
        "casr_v2_scalability.csv",
        ["configuration", "failure_rate"],
        [
            "casr_3",
            "palr_5",
            "adaptive",
            "lock_aware",
        ],
    ),
]

for study, filename, groups, baselines in studies:

    df = pd.read_csv(ROOT / "data/raw" / filename)

    for key, subset in df.groupby(groups):

        if not isinstance(key, tuple):
            key = (key,)

        metadata = dict(zip(groups, key))

        pivot = subset.pivot(
            index="seed",
            columns="method",
            values="blocked_moves",
        )

        for baseline in baselines:

            proposed = pivot["casr_v2_s2"].to_numpy()
            reference = pivot[baseline].to_numpy()

            diff = proposed - reference

            assert np.isfinite(diff).all()

            wins = int((diff < 0).sum())
            ties = int((diff == 0).sum())
            losses = int((diff > 0).sum())

            n = wins + losses

            p = (
                binomtest(
                    wins, n, 0.5,
                    alternative="two-sided",
                ).pvalue
                if n else 1.0
            )

            indices = rng.integers(
                0, len(diff),
                size=(10000, len(diff)),
            )

            boot = diff[indices].mean(axis=1)

            low, high = np.percentile(
                boot, [2.5, 97.5]
            )

            rows.append({
                "study": study,
                **metadata,
                "baseline": baseline,
                "n_pairs": len(diff),
                "proposed_mean": proposed.mean(),
                "baseline_mean": reference.mean(),
                "mean_difference": diff.mean(),
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "p_value": p,
            })

stats = pd.DataFrame(rows)

# Correct across all 56 exploratory comparisons.
p = stats.p_value.to_numpy()
order = np.argsort(p)

adjusted = np.maximum.accumulate(
    p[order] * (len(p) - np.arange(len(p)))
)

corrected = np.empty_like(p)
corrected[order] = np.minimum(adjusted, 1.0)

stats["p_holm"] = corrected
stats["significant"] = stats.p_holm < 0.05

assert len(stats) == 56

stats.to_csv(
    OUT / "casr_final_blocking_statistics.csv",
    index=False,
)

print("\n===== FINAL BLOCKING STATISTICS =====")
print(stats.round(5).to_string(index=False))

print("\n===== SIGNIFICANT COMPARISONS =====")
print(
    stats[stats.significant]
    .round(5)
    .to_string(index=False)
)

print("\nTotal comparisons:", len(stats))
print("Significant:", int(stats.significant.sum()))
print("\nAnalysis saved.")
