# Cookie Cats A/B Test Analysis

## 📌 Project Overview
Cookie Cats is a mobile puzzle game. The product team ran an A/B test to determine whether placing the first "gate" (a forced wait or in-app purchase prompt) at level 30 vs. level 40 would improve player retention. 

## 📊 The Data
- **Dataset:** 90,189 users randomly assigned to two groups (`gate_30` and `gate_40`).
- **Metrics:** 1-day retention and 7-day retention.
- **Data Quality:** Confirmed no missing values, no duplicates, and a successful Sample Ratio Mismatch (SRM) check (chi-square p = 1.0).

## 🔬 Methodology
1. **Exploratory Data Analysis:** Handled extreme outliers (one player with 49,854 rounds) by utilizing median rather than mean for game rounds.
2. **Statistical Testing:** Ran Chi-Square tests for both retention metrics.
3. **Confidence Intervals:** Calculated a 95% CI for the difference in 7-day retention.
4. **Bootstrap Validation:** Ran a 1,000-iteration bootstrap simulation to confirm the CI without relying on formulaic assumptions.

## 📈 Results
- **1-Day Retention:** No statistically significant difference (p = 0.075).
- **7-Day Retention:** Statistically significant drop for `gate_40` (p = 0.0016). 
- **Effect Size:** Moving the gate to level 40 decreased 7-day retention by 0.82% (95% CI: [-1.33%, -0.31%]).
- **Bootstrap Check:** 99.8% of bootstrap samples confirmed `gate_40` performed worse.

## 💡 Business Recommendation
**Do not move the gate to level 40.** 
While 1-day retention was unaffected, 7-day retention is a critical metric for long-term player lifetime value (LTV). The statistically significant drop indicates that Gate 30 provides a better long-term player experience. Further testing should explore gate placements *earlier* than level 30.

## 🛠️ Tech Stack
Python (pandas, scipy, matplotlib, numpy)