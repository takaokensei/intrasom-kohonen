# Scope Decision: Route A (Comprehensive Focused Study)

**Date:** 2026-07-30  
**Repository:** `intrasom-kohonen`

---

## 1. Scope Selection Rationale

For the Round 4 manuscript expansion, **Route A** is selected.

### Why Route A over Route B?
- **Route A** expands the study into a publication-grade full journal article by introducing:
  1. **30 Seeds per Configuration**: Statistically robust sample size ($N = 30$ seeds, totalizing $360$ runs per dataset $\times 3 \text{ datasets} = 1,080$ trained SOM models).
  2. **Paired Statistical Hypothesis Testing**: Non-parametric Wilcoxon signed-rank tests comparing downstream clustering performance ($\text{ARI}_{\text{ground\_a}}$ vs $\text{ARI}_{\text{ground\_b}}$) with False Discovery Rate (FDR / Benjamini-Hochberg) multiple-comparison corrections.
  3. **Bootstrap Confidence Intervals**: 95% percentile bootstrap CIs (1,000 resamples) for mean divergence and ARI metrics.
  4. **Multi-Dataset Generalization**: Evaluation across 3 benchmark datasets with distinct dimensionalities and class separabilities (*Synthetic Control*, *Wine*, and *Digits*).
- **Route B** (cross-tool execution of R `kohonen` and Python `MiniSom`) is documented in Section VI-E as an expanded, concrete Future Work objective, keeping the current manuscript tightly focused, methodologically sound, and zero-flaw.
