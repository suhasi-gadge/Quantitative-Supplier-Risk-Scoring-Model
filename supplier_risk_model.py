"""
Quantitative Supplier Risk Scoring Model
==========================================
Calculates a composite weighted risk score for each supplier based on:
  - Financial Stability    (25%)
  - Lead Time Variability  (20%)
  - Quality Defect Rate    (20%)
  - Geopolitical Risk      (20%)
  - Past Performance       (15%)

Scores are Min-Max normalised before weighting so all attributes are
compared on a 0-1 scale.  Higher composite score = HIGHER risk.
Suppliers are then bucketed into three tiers:
  Low     (score < 0.35)
  Medium  (0.35 – 0.65)
  High    (score > 0.65)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ── 1. Simulate supplier dataset ─────────────────────────────────────────────

np.random.seed(42)
N = 40  # number of suppliers

supplier_names = [f"Supplier_{i:02d}" for i in range(1, N + 1)]

# Raw attribute ranges (intentionally wide to create variation)
# financial_stability_score : 1 (very weak) → 10 (very strong)
# lead_time_variability_days: 0 (perfectly consistent) → 30 (highly variable)
# quality_defect_rate_pct   : 0 % → 15 %
# geopolitical_risk_score   : 1 (very safe region) → 10 (high-risk region)
# past_performance_rating   : 1 (poor) → 10 (excellent)

raw = {
    "supplier_name":            supplier_names,
    "financial_stability_score":     np.round(np.random.uniform(1, 10, N), 2),
    "lead_time_variability_days":    np.round(np.random.uniform(0, 30, N), 1),
    "quality_defect_rate_pct":       np.round(np.random.uniform(0, 15, N), 2),
    "geopolitical_risk_score":       np.round(np.random.uniform(1, 10, N), 2),
    "past_performance_rating":       np.round(np.random.uniform(1, 10, N), 2),
}

# Manually inject a handful of extreme profiles for illustration
raw["financial_stability_score"][0]  = 9.5   # very stable
raw["geopolitical_risk_score"][0]    = 1.2   # very safe → should be Low risk
raw["past_performance_rating"][0]    = 9.8

raw["financial_stability_score"][1]  = 1.1   # financially weak
raw["quality_defect_rate_pct"][1]    = 14.5  # terrible quality
raw["geopolitical_risk_score"][1]    = 9.3   # dangerous region → should be High risk

df = pd.DataFrame(raw)

print("=" * 65)
print("QUANTITATIVE SUPPLIER RISK SCORING MODEL")
print("=" * 65)
print(f"\nDataset loaded: {len(df)} suppliers, {len(df.columns) - 1} risk attributes\n")

# ── 2. Define risk weights ────────────────────────────────────────────────────

WEIGHTS = {
    "financial_stability_score":  0.25,   # high score = low risk → inverted later
    "lead_time_variability_days": 0.20,   # high variability = high risk
    "quality_defect_rate_pct":    0.20,   # high defects = high risk
    "geopolitical_risk_score":    0.20,   # high score = high risk
    "past_performance_rating":    0.15,   # high score = low risk → inverted later
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Attributes where a HIGHER raw value means LOWER risk (we invert them)
INVERT_ATTRIBUTES = {"financial_stability_score", "past_performance_rating"}

# ── 3. Normalise attributes (Min-Max to [0, 1]) ───────────────────────────────

feature_cols = list(WEIGHTS.keys())
scaler       = MinMaxScaler()
normalised   = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    columns=[f"norm_{c}" for c in feature_cols],
)

# Invert where necessary: risk_direction = 1 - normalised_value
for col in feature_cols:
    norm_col = f"norm_{col}"
    if col in INVERT_ATTRIBUTES:
        normalised[norm_col] = 1.0 - normalised[norm_col]

print("Normalisation complete (Min-Max, inversions applied where needed).")

# ── 4. Compute composite risk score ──────────────────────────────────────────

df["composite_risk_score"] = sum(
    normalised[f"norm_{col}"] * weight
    for col, weight in WEIGHTS.items()
)
df["composite_risk_score"] = df["composite_risk_score"].round(4)

# ── 5. Assign risk tiers ──────────────────────────────────────────────────────

LOW_THRESHOLD    = 0.35
HIGH_THRESHOLD   = 0.65

def assign_tier(score: float) -> str:
    if score < LOW_THRESHOLD:
        return "Low"
    elif score <= HIGH_THRESHOLD:
        return "Medium"
    else:
        return "High"

df["risk_tier"] = df["composite_risk_score"].apply(assign_tier)

# ── 6. Print summary statistics ──────────────────────────────────────────────

print("\n── Risk Score Statistics ──────────────────────────────────────")
print(df["composite_risk_score"].describe().round(4).to_string())

print("\n── Risk Tier Distribution ─────────────────────────────────────")
tier_counts = df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"])
for tier, count in tier_counts.items():
    bar   = "█" * count
    pct   = count / len(df) * 100
    print(f"  {tier:<8} {count:>3} suppliers  ({pct:5.1f}%)  {bar}")

print("\n── Top 5 Highest-Risk Suppliers ───────────────────────────────")
top5 = df.nlargest(5, "composite_risk_score")[
    ["supplier_name", "composite_risk_score", "risk_tier",
     "financial_stability_score", "quality_defect_rate_pct",
     "geopolitical_risk_score"]
]
print(top5.to_string(index=False))

print("\n── Top 5 Lowest-Risk Suppliers ────────────────────────────────")
bot5 = df.nsmallest(5, "composite_risk_score")[
    ["supplier_name", "composite_risk_score", "risk_tier",
     "financial_stability_score", "quality_defect_rate_pct",
     "geopolitical_risk_score"]
]
print(bot5.to_string(index=False))

# ── 7. Generate CSV report ────────────────────────────────────────────────────

report_cols = [
    "supplier_name",
    "financial_stability_score",
    "lead_time_variability_days",
    "quality_defect_rate_pct",
    "geopolitical_risk_score",
    "past_performance_rating",
    "composite_risk_score",
    "risk_tier",
]

report_df = df[report_cols].sort_values("composite_risk_score", ascending=False).reset_index(drop=True)
report_df.index += 1  # 1-based rank

OUTPUT_PATH = "supplier_risk_report.csv"
report_df.to_csv(OUTPUT_PATH, index_label="rank")

print(f"\n✓ Report written → {OUTPUT_PATH}  ({len(report_df)} rows)\n")
print("=" * 65)
