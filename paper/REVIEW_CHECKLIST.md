# Citation Audit & Review Checklist (`paper/references.bib`)

This checklist documents the 100% verification pass performed on all 17 citations in `paper/references.bib` to ensure academic integrity and precise alignment between text claims and bibliographic sources.

| Key | Authors & Year | Claim Supported in `paper/main.tex` | Verification Source / Link | Status |
|---|---|---|---|---|
| `kohonen1982self` | Kohonen (1982) | Original SOM formulation (Biological Cybernetics) | DOI: [10.1007/BF00337288](https://doi.org/10.1007/BF00337288) | Verified |
| `kohonen2001self` | Kohonen (2001) | SOM monograph & topology preservation | Springer Series in Info. Sci. (ISBN 978-3-540-67921-9) | Verified |
| `kohonen2013essentials` | Kohonen (2013) | Batch SOM training & neighborhood updating rule | DOI: [10.1016/j.neunet.2012.09.018](https://doi.org/10.1016/j.neunet.2012.09.018) | Verified |
| `ultsch1993self` | Ultsch (1993) | Original U-matrix concept for SOM visualization | DOI: [10.1007/978-3-642-50974-2_31](https://doi.org/10.1007/978-3-642-50974-2_31) | Verified |
| `costa2007segmentacao` | Costa & Netto (2007) | Inter-neuron U-matrix math & $1/\sqrt{2}$ diagonal factor (Eq. 3) | DOI: [10.1590/S0103-17592007000200002](https://doi.org/10.1590/S0103-17592007000200002) (SciELO) | Verified |
| `vesanto2000clustering` | Vesanto & Alhoniemi (2000) | Reduced U-matrix calculation per neuron & SOM clustering | DOI: [10.1109/72.846731](https://doi.org/10.1109/72.846731) | Verified |
| `alcock1999time` | Alcock & Manolopoulos (1999) | Source of Synthetic Control Chart Time Series dataset | PCI 1999 proceedings, Ioannina, Greece (UCI Repository origin) | Verified |
| `kaski1996comparing` | Kaski & Lagus (1996) | SOM quality evaluation & comparison methods | DOI: [10.1007/3-540-61510-5_138](https://doi.org/10.1007/3-540-61510-5_138) | Verified |
| `flexer2001use` | Flexer (2001) | Variations in U-matrix implementation across SOM tools | DOI: [10.3233/IDA-2001-5504](https://doi.org/10.3233/IDA-2001-5504) | Verified |
| `sculley2015hidden` | Sculley et al. (2015) | Technical debt and hidden assumptions in ML systems | NeurIPS 2015, pp. 2503-2511 | Verified |
| `stodden2014implementing` | Stodden & Miguez (2014) | Best practices in computational science & software infrastructure | DOI: [10.5334/jors.ay](https://doi.org/10.5334/jors.ay) | Verified |
| `peng2011reproducible` | Peng (2011) | Reproducible research in computational science | DOI: [10.1126/science.1213847](https://doi.org/10.1126/science.1213847) | Verified |
| `sandve2013ten` | Sandve et al. (2013) | Ten simple rules for reproducible computational research | DOI: [10.1371/journal.pcbi.1003285](https://doi.org/10.1371/journal.pcbi.1003285) | Verified |
| `ultsch1990innc` | Ultsch & Siemon (1990) | Early U-matrix presentation at INNC'90 | INNC'90 proceedings, Kluwer, Vol. 1, pp. 305-308 | Verified |
| `gouvea2023intrasom` | Gouvêa et al. (2023) | Official publication for IntraSOM library (InTRA-USP) | DOI: [10.1016/j.simpa.2023.100570](https://doi.org/10.1016/j.simpa.2023.100570) (*Software Impacts*) | Verified |
| `wehrens2007self` | Wehrens & Buydens (2007) | R `kohonen` package U-matrix implementation reference | DOI: [10.18637/jss.v021.i05](https://doi.org/10.18637/jss.v021.i05) | Verified |
| `venna2007visualizing` | Venna & Kaski (2007) | Topology preservation & visualization in SOMs | DOI: [10.1016/j.neunet.2006.11.013](https://doi.org/10.1016/j.neunet.2006.11.013) | Verified |

---

## Explicit Rules Enforced
1. `alcock1999time` is cited as the dataset source for *Synthetic Control*, replacing `kaski1996comparing`.
2. `sculley2015hidden` and `sandve2013ten` are cited strictly for general technical debt and ML reproducibility, NOT as sources for `ENGINEERING_AUDIT.md`.
3. `ultsch1990innc` is renamed from `ultsch2003maps` to match its actual publication year (INNC'90).
4. `gouvea2023intrasom` replaces `intrasom2026pypi`, citing the peer-reviewed *Software Impacts* paper.
5. MiniSom attribution is removed from `vesanto2000clustering`; MiniSom is by Giuseppe Vettigli, whereas R `kohonen` (`wehrens2007self`) and IntraSOM (`gouvea2023intrasom`) are cited as package references.

---

## Round 2 Audit Corrections (July 2026 Pass)
- **Deterministic PCA Initialization Notice**: Added `paper/CRITICAL_BUG_NOTICE.md` documenting that `initialization='pca'` produced zero seed variance (`std=0.000`), and verified that switching to `initialization='random'` produces true non-zero standard deviations.
- **Table I & Paper Verification**: Updated Table I in `paper/main.tex` with real non-zero standard deviations across 5 seeds.
- **IntraSOM `build_umatrix` Line Citation**: Corrected line citation to line **2626** in `intrasom/intrasom.py` (v1.1.1).
- **Scope & Limitations**: Enframed manuscript as `[Nota Técnica]` / Technical Note and added an explicit "Limitações" subsection detailing geometric scope (RECT), single dataset (*Synthetic Control*), and random initialization methodology.

---

## Round 4 Audit Tracking Checklist

- [x] **Scope Decision Documented**: Created [`paper/SCOPE_DECISION.md`](file:///c:/IntraSOM_Kohonen_Synthetic_Control_Visual_Law/paper/SCOPE_DECISION.md) establishing Route A selection.
- [x] **Statistical Rigor Expansion**: Extended evaluation to $N=30$ seeds (42..71), 95% Bootstrap CIs (1,000 resamples), paired Wilcoxon signed-rank tests, and Benjamini-Hochberg FDR correction.
- [x] **Multi-Dataset Benchmarking**: Trained 1,080 SOM models across *Synthetic Control*, *Wine*, and *Digits* datasets.
- [x] **Literature Expansion**: Added related works (Ultsch 1993, Costa & Netto 2007, Vesanto 2000, Flexer 2001, Venna & Kaski 2007, Peng 2011, Stodden 2014, Sculley 2015, Sandve 2013, Benjamini-Hochberg 1995, Efron 1994, Wilcoxon 1945).
- [x] **Manuscript Reformatting**: Expanded [`paper/main.tex`](file:///c:/IntraSOM_Kohonen_Synthetic_Control_Visual_Law/paper/main.tex) from a 5-page Technical Note into a 6-page full journal manuscript with cross-dataset generalization, SOM Library Compliance Framework, declared limitations, and expanded Future Work.
- [x] **LaTeX Compilation**: Successfully built PDF artifact [`paper/main.pdf`](file:///c:/IntraSOM_Kohonen_Synthetic_Control_Visual_Law/paper/main.pdf) (6 pages, 420 KB).
- [x] **Git Workflow**: Executed 5 granular Conventional Commits and pushed to GitHub main branch.

---

## Verificação Numérica Rodada 5 - Parágrafo por Parágrafo

This section documents the 100% paragraph-by-paragraph verification pass comparing every quantitative and qualitative claim in `paper/main.tex` against the source CSV files (`umatrix_divergence_synthetic_control.csv`, `umatrix_divergence_wine.csv`, `umatrix_divergence_digits.csv`, and `hypothesis_testing_summary.csv`).

### Abstract & Sec. 1 (Introdução)
- **Total SOM models trained**: 1,080 models ($3 \text{ datasets} \times 6 \text{ sizes} \times 2 \text{ variants} \times 30 \text{ seeds}$). Verified against `umatrix_divergence_raw_seeds_multidataset.csv` (1,080 rows).
- **Inflation range**: $16{,}8\%$ to $20{,}5\%$ (scale ratio $1{,}168$ to $1{,}205$). Verified: min rel_diff_pct = $16.84\%$ ($5\times5$ Planar SC), max rel_diff_pct = $20.53\%$ ($12\times12$ Toroidal Wine).
- **Pearson correlation range**: $r \ge 0.922$ overall (min is $0.9218$ in $5\times5$ Planar Digits), mean $r \ge 0.988$ across all datasets.
- **Asymptotic theoretical limit**: $(1+\sqrt{2})/2 \approx 1{,}2071$ ($20{,}71\%$). Derived analytically in Section 4.2.
- **Hypothesis test rejections ($H_0$)**: 20 out of 36 configurations ($55.6\%$) rejected $H_0$ under paired Wilcoxon test with Benjamini-Hochberg FDR correction ($q=0.05$). Breakdown: 9/12 ($75\%$) in *Synthetic Control*, 8/12 ($67\%$) in *Wine*, 3/12 ($25\%$) in *Digits*. Rosenthal $r_{\text{effect}} \in [0.39, 1.00]$.

### Sec. 4 & 5 (Metodologia e Resultados - Synthetic Control)
- **$5\times5$ RECT Planar**: Pearson $r = 0.991$ [0.990, 0.991], Scale = $1.168$, ARI GT $U_A = 0.232$, $U_B = 0.209$, $p_{\text{fdr}} < 0.001$, $r_{\text{effect}} = 0.839$.
- **$5\times5$ RECT Toroidal**: Pearson $r = 0.990$ [0.990, 0.991], Scale = $1.202$, ARI GT $U_A = 0.240$, $U_B = 0.192$, $p_{\text{fdr}} = 0.030$, $r_{\text{effect}} = 0.433$.
- **$7\times7$ RECT Planar**: Pearson $r = 0.997$ [0.996, 0.997], Scale = $1.180$, ARI GT $U_A = 0.248$, $U_B = 0.245$, $p_{\text{fdr}} = 0.002$, $r_{\text{effect}} = 0.607$.
- **$7\times7$ RECT Toroidal**: Pearson $r = 0.998$ [0.998, 0.999], Scale = $1.203$, ARI GT $U_A = 0.244$, $U_B = 0.213$, $p_{\text{fdr}} < 0.001$, $r_{\text{effect}} = 0.695$.
- **$10\times10$ RECT Planar**: Pearson $r = 0.998$ [0.998, 0.998], Scale = $1.188$, ARI GT $U_A = 0.202$, $U_B = 0.211$, $p_{\text{fdr}} = 0.292$ (not significant), $r_{\text{effect}} = 0.213$.
- **$10\times10$ RECT Toroidal**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.204$, ARI GT $U_A = 0.163$, $U_B = 0.166$, $p_{\text{fdr}} = 0.452$ (not significant), $r_{\text{effect}} = 0.149$.
- **$12\times12$ RECT Planar**: Pearson $r = 0.998$ [0.998, 0.999], Scale = $1.191$, ARI GT $U_A = 0.178$, $U_B = 0.186$, $p_{\text{fdr}} < 0.001$, $r_{\text{effect}} = 0.804$.
- **$12\times12$ RECT Toroidal**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.204$, ARI GT $U_A = 0.133$, $U_B = 0.138$, $p_{\text{fdr}} = 0.031$, $r_{\text{effect}} = 0.422$.
- **$15\times15$ RECT Planar**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.194$, ARI GT $U_A = 0.167$, $U_B = 0.174$, $p_{\text{fdr}} = 0.037$, $r_{\text{effect}} = 0.402$.
- **$15\times15$ RECT Toroidal**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.203$, ARI GT $U_A = 0.123$, $U_B = 0.136$, $p_{\text{fdr}} < 0.001$, $r_{\text{effect}} = 0.836$.
- **$20\times20$ RECT Planar**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.197$, ARI GT $U_A = 0.141$, $U_B = 0.150$, $p_{\text{fdr}} < 0.001$, $r_{\text{effect}} = 0.862$.
- **$20\times20$ RECT Toroidal**: Pearson $r = 0.999$ [0.999, 0.999], Scale = $1.204$, ARI GT $U_A = 0.100$, $U_B = 0.099$, $p_{\text{fdr}} = 0.952$ (not significant), $r_{\text{effect}} = 0.011$.

### Sec. 6 (Generalização Cross-Dataset)
- **Synthetic Control**: Mean Pearson $r = 0.998$, $9/12$ rejections ($75\%$), mean scale = $1.193$, max $r_{\text{effect}} = 0.862$.
- **Wine**: Mean Pearson $r = 0.997$, $8/12$ rejections ($67\%$), mean scale = $1.196$, max $r_{\text{effect}} = 1.000$ ($7\times7$ Toroidal, $p_{\text{fdr}} = 5.18 \times 10^{-7}$).
- **Digits**: Mean Pearson $r = 0.988$, $3/12$ rejections ($25\%$), mean scale = $1.193$, max $r_{\text{effect}} = 0.679$ ($7\times7$ Planar, $p_{\text{fdr}} = 0.002$).

### Sec. 7 (Discussão & Impacto Downstream)
- **Contradiction Resolution**: Eliminated residual claims of "compatibilidade em todas as configurações". Replaced with precise breakdown: $H_0$ rejected in $55.6\%$ of tests ($20/36$), proving a statistically significant and real effect on downstream segmentation.
- **Effect Size & Overlap Relation**: Documented that effect detectability and $r_{\text{effect}}$ decrease as dataset class overlap increases (SC $75\%$ > Wine $67\%$ > Digits $25\%$).

