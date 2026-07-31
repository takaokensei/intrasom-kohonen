"""
generate_figures.py
===================
Publication-grade scientific figure generation for IEEEtran manuscript:
"Efeito da Normalização Diagonal na U-Matrix de Mapas Auto-Organizáveis Retangulares"

Figures produced:
  fig1_umatrix_comparison.pdf     — Heatmaps comparing U_A (IntraSOM 1.1.1) vs U_B (Costa & Netto 2007)
  fig2_systematic_divergence.pdf  — Systematic inflation ratio & rel diff vs map dimension N (with theoretical limit 1.207)
  fig3_segmentation_ari.pdf       — Cluster segmentation ARI & effect size across map sizes & topologies
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── IEEE Publication Style Settings ───────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         8,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'axes.titleweight':  'bold',
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'figure.dpi':        300,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'mathtext.fontset':  'cm',
})

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UMAT_DIR  = os.path.join(WORKSPACE, "outputs", "umatrices")
MET_DIR   = os.path.join(WORKSPACE, "outputs", "metrics")
FIG_DIR   = os.path.join(WORKSPACE, "paper",   "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Curated IEEE color palette
COLOR_PLANAR  = "#1f77b4"  # IEEE Blue
COLOR_TOROID  = "#d62728"  # IEEE Red
COLOR_REF     = "#555555"  # Dark Grey for theoretical line
CMAP_HEAT     = "viridis"
CMAP_DIFF     = "rocket" if "rocket" in plt.colormaps() else "inferno"

# Load divergence table
div_csv = os.path.join(MET_DIR, "umatrix_divergence_synthetic_control.csv")
df_div  = pd.read_csv(div_csv)

def _dim(size_str: str) -> int:
    return int(size_str.split("x")[0])

sizes_ordered = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]
dims = [_dim(s) for s in sizes_ordered]

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Heatmaps comparison for SOM_10x10_RECT_toroid
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "SOM_10x10_RECT_toroid"

umat_a  = np.load(os.path.join(UMAT_DIR, f"{MODEL}_intrasom.npy"))
umat_b  = np.load(os.path.join(UMAT_DIR, f"{MODEL}_classical.npy"))
umat_df = np.load(os.path.join(UMAT_DIR, f"{MODEL}_diff.npy"))

vmin = min(umat_a.min(), umat_b.min())
vmax = max(umat_a.max(), umat_b.max())

fig1, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))

im0 = axes[0].imshow(umat_a, cmap=CMAP_HEAT, vmin=vmin, vmax=vmax, origin="upper")
axes[0].set_title("(a) IntraSOM v1.1.1 ($U_A$)\n(normas brutas)")
axes[0].set_xlabel("Coluna")
axes[0].set_ylabel("Linha")
cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.ax.tick_params(labelsize=7)

im1 = axes[1].imshow(umat_b, cmap=CMAP_HEAT, vmin=vmin, vmax=vmax, origin="upper")
axes[1].set_title("(b) Costa & Netto 2007 ($U_B$)\n(fator $1/\\sqrt{2}$)")
axes[1].set_xlabel("Coluna")
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=7)

im2 = axes[2].imshow(umat_df, cmap=CMAP_DIFF, origin="upper")
axes[2].set_title("(c) Diferença Absoluta\n$|U_A - U_B|$")
axes[2].set_xlabel("Coluna")
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.ax.tick_params(labelsize=7)

plt.tight_layout()
path1 = os.path.join(FIG_DIR, "fig1_umatrix_comparison.pdf")
plt.savefig(path1, bbox_inches="tight")
plt.savefig(path1.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 1 saved -> {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Systematic inflation & scale ratio vs grid dimension N
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6))

sub_p = df_div[df_div["variant"] == "RECT_planar"].set_index("size")
sub_t = df_div[df_div["variant"] == "RECT_toroid"].set_index("size")

rd_p = [sub_p.loc[s, "rel_diff_pct_mean"] for s in sizes_ordered]
rd_t = [sub_t.loc[s, "rel_diff_pct_mean"] for s in sizes_ordered]

sr_p = [sub_p.loc[s, "scale_ratio_mean"] for s in sizes_ordered]
sr_t = [sub_t.loc[s, "scale_ratio_mean"] for s in sizes_ordered]

ax1.plot(dims, rd_p, marker="o", color=COLOR_PLANAR, label="Planar", linewidth=1.5, markersize=4)
ax1.plot(dims, rd_t, marker="s", color=COLOR_TOROID, label="Toroidal", linewidth=1.5, markersize=4)
ax1.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax1.set_ylabel("Diferença Relativa Média (%)")
ax1.set_title("(a) Divergência Relativa (%)")
ax1.set_xticks(dims)
ax1.set_ylim(15, 23)
ax1.legend(frameon=False, loc="lower right")
ax1.grid(True, linestyle=":", alpha=0.5)

ax2.plot(dims, sr_p, marker="o", color=COLOR_PLANAR, label="Planar", linewidth=1.5, markersize=4)
ax2.plot(dims, sr_t, marker="s", color=COLOR_TOROID, label="Toroidal", linewidth=1.5, markersize=4)
ax2.axhline(1.2071, color=COLOR_REF, linestyle="--", linewidth=1.0, alpha=0.8, label="Limite Teórico (1,207)")
ax2.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax2.set_ylabel("Razão de Escala $\\overline{U}_A / \\overline{U}_B$")
ax2.set_title("(b) Convergência ao Teto Teórico")
ax2.set_xticks(dims)
ax2.set_ylim(1.15, 1.22)
ax2.legend(frameon=False, loc="lower right")
ax2.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
path2 = os.path.join(FIG_DIR, "fig2_systematic_divergence.pdf")
plt.savefig(path2, bbox_inches="tight")
plt.savefig(path2.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 2 saved -> {path2}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — ARI between U_A and U_B segmentations vs grid dimension
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(4.5, 2.7))

ari_p = [sub_p.loc[s, "seg_ari_mean"] for s in sizes_ordered]
ari_t = [sub_t.loc[s, "seg_ari_mean"] for s in sizes_ordered]

ax.plot(dims, ari_p, marker="o", color=COLOR_PLANAR, label="Planar", linewidth=1.5, markersize=4)
ax.plot(dims, ari_t, marker="s", color=COLOR_TOROID, label="Toroidal", linewidth=1.5, markersize=4)

ax.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.0, alpha=0.7, label="ARI = 1 (idênticos)")
ax.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax.set_ylabel("ARI ($U_A$ vs. $U_B$)")
ax.set_title("Concordância de Segmentação por Resolução")
ax.set_xticks(dims)
ax.set_ylim(0.4, 1.05)
ax.legend(frameon=False, loc="lower right")
ax.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
path3 = os.path.join(FIG_DIR, "fig3_segmentation_ari.pdf")
plt.savefig(path3, bbox_inches="tight")
plt.savefig(path3.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 3 saved -> {path3}")

print("\n[ALL 3 IEEE FIGURES GENERATED CLEANLY]")
