# ============================================================
# Day 1 — Cookie Cats A/B Test: Load, EDA, Sample Ratio Mismatch
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------
df = pd.read_csv("cookie_cats.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nDtypes:\n", df.dtypes)
print("\nHead:\n", df.head())

# LOOK: 5 columns expected — userid, version, sum_gamerounds,
#       retention_1, retention_7. ~90k rows.

# ------------------------------------------------------------
# 2. DATA QUALITY
# ------------------------------------------------------------
print("\nNulls per column:\n", df.isnull().sum())
print("\nDuplicate userids:", df["userid"].duplicated().sum())
print("\nUnique versions:", df["version"].unique())

# LOOK: any nulls? duplicates? Exactly two versions (gate_30, gate_40)?
#       If duplicates exist, that's a data integrity flag to mention.

# ------------------------------------------------------------
# 3. GROUP SIZES + SAMPLE RATIO MISMATCH (SRM) CHECK
# ------------------------------------------------------------
# SRM = did users split in the ratio we expected? If not, something
# in randomization/assignment is broken and the test is untrustworthy.
counts = df["version"].value_counts().sort_index()
print("\nGroup counts:\n", counts)
print("\nProportions:\n", (counts / len(df)).round(4))

observed = counts.values.reshape(1, -1)          # [[n_30, n_40]]
chi2, p_srm, dof, expected = chi2_contingency(observed)
print(f"\nSRM chi-square p-value: {p_srm:.4f}")
print("Expected counts under 50/50:", expected.round(0))

# LOOK: p > 0.05  →  no SRM detected, groups are balanced. GOOD.
#       p < 0.05  →  SRM present, flag it. (Cookie Cats is balanced,
#       so you should see p well above 0.05 — nice clean result.)

# ------------------------------------------------------------
# 4. OUTCOME VARIABLES — retention rates by group
# ------------------------------------------------------------
summary = df.groupby("version")[["retention_1", "retention_7"]].mean().T
summary["abs_diff"] = summary["gate_40"] - summary["gate_30"]
summary["rel_diff_%"] = (summary["abs_diff"] / summary["gate_30"] * 100).round(2)
print("\nRetention by group:\n", summary.round(4))

# LOOK: retention_1 ~44%, retention_7 ~18% overall. Compare gate_30
#       vs gate_40 — the direction of the diff is your headline finding.
#       (Spoiler: gate_30 wins on retention_7 — but pretend you don't know.)

# ------------------------------------------------------------
# 5. GAME ROUNDS DISTRIBUTION (watch for the outlier!)
# ------------------------------------------------------------
print("\nsum_gamerounds describe:\n", df["sum_gamerounds"].describe(
    percentiles=[.25, .5, .75, .90, .99, .999]))

# LOOK: max is ~49,854 — a single whale. This will wreck mean-based
#       stats. Note it now; Day 2 you'll decide whether to trim.
#       Median is ~16 rounds. That skew is the story.

# Trim extreme outlier for plotting only (do NOT delete from df yet)
p99 = df["sum_gamerounds"].quantile(0.99)
plot_df = df[df["sum_gamerounds"] <= p99]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) distribution of rounds, log-ish view via 99th pct clip
sns.histplot(data=plot_df, x="sum_gamerounds", hue="version",
             bins=50, ax=axes[0], alpha=0.6)
axes[0].set_title(f"sum_gamerounds (clipped at 99th pct = {p99:.0f})")
axes[0].set_xlabel("Rounds played in first 14 days")

# (b) retention rates bar chart
ret = df.groupby("version")[["retention_1", "retention_7"]].mean()
ret.plot(kind="bar", ax=axes[1], rot=0, color=["#4C72B0", "#DD8452"])
axes[1].set_title("Retention rates by group")
axes[1].set_ylabel("Retention rate")
axes[1].set_xlabel("")
axes[1].legend(["1-day", "7-day"], title="")
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.3f", padding=2)

plt.tight_layout()
plt.savefig("day1_eda.png", dpi=120, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 6. QUICK SANITY — means of rounds (expect noise due to whale)
# ------------------------------------------------------------
print("\nMean rounds by group (contaminated by whale):\n",
      df.groupby("version")["sum_gamerounds"].mean().round(2))
print("\nMedian rounds by group (robust):\n",
      df.groupby("version")["sum_gamerounds"].median())

# LOOK: mean gate_30 looks higher than gate_40, but that's mostly the
#       whale. Medians will be nearly identical. This is exactly the
#       "mean vs median + outlier" lesson to mention in your writeup.