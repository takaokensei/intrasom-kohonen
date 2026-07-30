# Active Figures Directory (`paper/figures/`)

This directory contains the 4 publication-quality figures consumed by `paper/main.tex`.
All figures are generated automatically by `paper/scripts/generate_figures.py` from the
experimental output files in `outputs/metrics/` and `outputs/umatrices/`.

## Active Figures Manifest

| File | Format | Referenced in `main.tex` | Description |
|---|---|---|---|
| `fig1_umatrix_comparison` | `.pdf` / `.png` | Fig. 1 | U-matrix heatmaps: (a) IntraSOM 1.1.1, (b) Classical Costa & Netto (2007), (c) Absolute difference |
| `fig2_divergence_distribution` | `.pdf` / `.png` | Fig. 2 | Relative divergence distribution per neuron (a) and mean relative divergence bar chart across models (b) |
| `fig3_systematic_divergence` | `.pdf` / `.png` | Fig. 3 | Systematic divergence vs. grid size $N$: (a) relative difference %, (b) scale ratio $\bar{U}_A/\bar{U}_B$ |
| `fig4_segmentation_ari` | `.pdf` / `.png` | Fig. 4 | Adjusted Rand Index (ARI) between U-matrix segmentations ($U_A$ vs $U_B$) as a function of grid size |

## Regeneration Command

To regenerate all figures from the latest experimental metrics:

```bash
python paper/scripts/generate_figures.py
```
