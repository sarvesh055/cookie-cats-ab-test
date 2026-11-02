# ============================================================
# Day 2 — Statistical Testing (Chi-Square, CI, Bootstrap)
# ============================================================
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
df = pd.read_csv("cookie_cats.csv")

# ------------------------------------------------------------
# 1. CHI-SQUARE TEST (Retention 1 and 7)
# ------------------------------------------------------------
print("=== CHI-SQUARE TESTS ===")
for metric in ["retention_1", "retention_7"]:
    # Create a 2x2 contingency table: [ [gate30_retained, gate30_not], [gate40_retained, gate40_not] ]
    ct = pd.crosstab(df["version"], df[metric])
    
    chi2, p, dof, expected = chi2_contingency(ct)
    
    print(f"\nMetric: {metric}")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    if p < 0.05:
        print("Result: Statistically significant difference (reject H0)")
    else:
        print("Result: No statistically significant difference (fail to reject H0)")

# LOOK: Is the p-value for retention_7 < 0.05? (Spoiler: It should be ~0.0016).

# ------------------------------------------------------------
# 2. EFFECT SIZE & CONFIDENCE INTERVALS (Retention 7)
# ------------------------------------------------------------
print("\n=== EFFECT SIZE & 95% CI (Retention 7) ===")
# Counts of successes (True) and totals
gate30 = df[df["version"] == "gate_30"]["retention_7"]
gate40 = df[df["version"] == "gate_40"]["retention_7"]

count = np.array([gate30.sum(), gate40.sum()])
nobs = np.array([len(gate30), len(gate40)])

# Difference in proportions (gate_40 - gate_30)
diff = (count[1]/nobs[1]) - (count[0]/nobs[0])
ci_low, ci_high = confint_proportions_2indep(count[1], nobs[1], count[0], nobs[0], method='wald')

print(f"Observed difference (gate_40 - gate_30): {diff*100:.2f}%")
print(f"95% Confidence Interval: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

# LOOK: Does the 95% CI cross zero? If both numbers are negative, 
#       it means the drop is statistically significant and we are confident.

# ------------------------------------------------------------
# 3. BOOTSTRAP ROBUSTNESS CHECK
# ------------------------------------------------------------
# A/B tests are sometimes skewed. Let's bootstrap the difference
# to double-check the math without relying on formulas.
print("\n=== BOOTSTRAP CHECK (1000 iterations) ===")
np.random.seed(42)
diffs = []
for _ in range(1000):
    # Resample with replacement
    boot30 = np.random.choice(gate30, size=len(gate30), replace=True)
    boot40 = np.random.choice(gate40, size=len(gate40), replace=True)
    diffs.append(boot40.mean() - boot30.mean())

diffs = np.array(diffs)
print(f"Bootstrap Mean Diff: {np.mean(diffs)*100:.2f}%")
print(f"Bootstrap 95% CI: [{np.percentile(diffs, 2.5)*100:.2f}%, {np.percentile(diffs, 97.5)*100:.2f}%]")
print(f"% of bootstrap samples where gate_40 was worse: {(diffs < 0).mean()*100:.1f}%")

# LOOK: The % where gate_40 is worse should be ~99%. This is proof
#       that gate_40 is definitively the loser.