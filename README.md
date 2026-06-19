# Quantitative-Supplier-Risk-Scoring-Model# Quantitative Supplier Risk Scoring Model

A Python-based weighted scoring model that objectively quantifies and tiers supplier risk across financial, operational, quality, and geopolitical dimensions.

## Files

| File | Description |
|---|---|
| `supplier_risk_model.py` | Main Python script — simulates data, runs the model, writes the report |
| `supplier_risk_report.csv` | Generated output — one row per supplier with scores and risk tier |

## How It Works

### 1. Dataset
40 suppliers are simulated with five risk attributes, each with intentional variation and a few manually injected edge cases (e.g. a financially strong supplier in a safe region vs. a weak supplier in a high-risk zone) to validate the model's discrimination power.

### 2. Risk Factors & Weights

| Attribute | Weight | Risk Direction |
|---|---|---|
| Financial Stability Score (1–10) | 25% | Inverted — higher score = safer |
| Lead Time Variability (days) | 20% | Direct — higher = riskier |
| Quality Defect Rate (%) | 20% | Direct — higher = riskier |
| Geopolitical Risk Score (1–10) | 20% | Direct — higher = riskier |
| Past Performance Rating (1–10) | 15% | Inverted — higher score = safer |

### 3. Normalisation
All attributes are scaled to [0, 1] using Min-Max normalisation. Inverted attributes are flipped so that `1.0` always represents maximum risk contribution, ensuring fair cross-attribute comparison.

### 4. Composite Risk Score
A single weighted sum in [0, 1] is computed per supplier. Higher score = higher risk.

### 5. Risk Tiers

| Tier | Score Range |
|---|---|
| 🟢 Low | < 0.35 |
| 🟡 Medium | 0.35 – 0.65 |
| 🔴 High | > 0.65 |

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install pandas numpy scikit-learn
```

## Usage

```bash
python supplier_risk_model.py
```

The script prints a summary to the console and writes `supplier_risk_report.csv` to the working directory.

## Output

The CSV report is ranked by composite risk score (highest first) and includes:

- `rank` — position by risk score
- `supplier_name`
- All five raw attribute values
- `composite_risk_score` — rounded to 4 decimal places
- `risk_tier` — Low / Medium / High

## Sample Results

```
── Risk Tier Distribution ─────────────────────────────────────
  Low        6 suppliers  ( 15.0%)
  Medium    28 suppliers  ( 70.0%)
  High       6 suppliers  ( 15.0%)

── Top 3 Highest-Risk Suppliers ───────────────────────────────
Supplier_02   0.7910   High   (fin: 1.1, defects: 14.5%, geo: 9.3)
Supplier_15   0.7274   High   (fin: 2.6, defects: 11.6%, geo: 9.5)
Supplier_07   0.7182   High   (fin: 1.5, defects: 10.9%, geo: 8.4)
```
