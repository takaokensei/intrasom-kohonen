"""
generate_figures.py
===================
Gera todas as figuras científicas do artigo reformulado "Efeito da Normalização
Diagonal na U-Matrix de SOMs Retangulares", com base nos dados reais do projeto.

Figuras produzidas:
  Fig 1 — Par de heatmaps da U-matrix (IntraSOM vs. Classical) para o modelo
           SOM_10x10_RECT_toroid (melhor exemplo: alta correlação, alta divergência).
  Fig 2 — Histograma + mapa de calor da diferença relativa entre as duas versões,
           com estatísticas resumo por modelo.
  Fig 3 — Divergência sistemática (rel_diff e scale_ratio) em função do tamanho
           da grade, para RECT_planar e RECT_toroid.
  Fig 4 — ARI da segmentação (comparação entre as duas versões) vs. tamanho da grade,
           evidenciando quando a escolha de implementação muda a partição de clusters.

Requisitos:
  outputs/umatrices/SOM_*_intrasom.npy   (gerado por src/umatrix_comparison.py)
  outputs/umatrices/SOM_*_classical.npy
  outputs/umatrices/SOM_*_diff.npy
  outputs/metrics/umatrix_divergence.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─── Publication style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.labelsize':    10,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'legend.fontsize':   8,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UMAT_DIR  = os.path.join(WORKSPACE, "outputs", "umatrices")
MET_DIR   = os.path.join(WORKSPACE, "outputs", "metrics")
FIG_DIR   = os.path.join(WORKSPACE, "paper",   "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Colormap consistente com o artigo
CMAP      = "magma"
CMAP_DIFF = "YlOrRd"

# ─────────────────────────────────────────────────────────────────────────────
# Load divergence table
# ─────────────────────────────────────────────────────────────────────────────
div_csv = os.path.join(MET_DIR, "umatrix_divergence.csv")
df_div  = pd.read_csv(div_csv)


def _dim(size_str: str) -> int:
    return int(size_str.split("x")[0])


df_div["dim"] = df_div["size"].apply(_dim)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Heatmaps comparativos para SOM_10x10_RECT_toroid
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "SOM_10x10_RECT_toroid"

umat_a  = np.load(os.path.join(UMAT_DIR, f"{MODEL}_intrasom.npy"))
umat_b  = np.load(os.path.join(UMAT_DIR, f"{MODEL}_classical.npy"))
umat_df = np.load(os.path.join(UMAT_DIR, f"{MODEL}_diff.npy"))

# Shared color scale across both heatmaps
vmin = min(umat_a.min(), umat_b.min())
vmax = max(umat_a.max(), umat_b.max())

fig1, axes = plt.subplots(1, 3, figsize=(12, 3.8),
                           gridspec_kw={"width_ratios": [1, 1, 1]})

im0 = axes[0].imshow(umat_a, cmap=CMAP, vmin=vmin, vmax=vmax, origin="upper")
axes[0].set_title("(a) IntraSOM 1.1.1\n(norma bruta, sem $1/\\sqrt{2}$)")
axes[0].set_xlabel("Coluna")
axes[0].set_ylabel("Linha")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="U-matrix")

im1 = axes[1].imshow(umat_b, cmap=CMAP, vmin=vmin, vmax=vmax, origin="upper")
axes[1].set_title("(b) Formulação Clássica\n(Costa \\& Netto 2007, Eq. 3)")
axes[1].set_xlabel("Coluna")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="U-matrix")

# Difference map with its own color scale
im2 = axes[2].imshow(umat_df, cmap=CMAP_DIFF, origin="upper")
axes[2].set_title("(c) Diferença Absoluta\n$|U_{\\mathrm{IntraSOM}} - U_{\\mathrm{Class}}|$")
axes[2].set_xlabel("Coluna")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="$|\\Delta U|$")

# Row from divergence table for annotation
row = df_div[df_div["model"] == MODEL].iloc[0]
fig1.suptitle(
    f"Modelo: {MODEL.replace('_', '\\_')}    |    "
    f"$r$ = {row.pearson_r:.4f}    |    "
    f"Dif. Rel. Média = {row.rel_diff_pct:.1f}\\%    |    "
    f"Razão de Escala = {row.scale_ratio:.4f}",
    fontsize=9, y=1.01
)

plt.tight_layout()
path1 = os.path.join(FIG_DIR, "fig1_umatrix_comparison.pdf")
plt.savefig(path1, bbox_inches="tight")
plt.savefig(path1.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 1 saved -> {path1}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Distribuição da diferença relativa por neurônio + resumo global
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.0))

# Left: histogram for the 10x10 RECT_toroid model
rel_diff_per_neuron = (umat_a - umat_b) / umat_b * 100  # in percent
flat = rel_diff_per_neuron.flatten()
ax_hist.hist(flat, bins=20, color="#3b82f6", edgecolor="white", linewidth=0.6)
ax_hist.axvline(np.mean(flat), color="#dc2626", linestyle="--", linewidth=1.5,
                label=f"Média = {np.mean(flat):.1f}%")
ax_hist.axvline(np.median(flat), color="#16a34a", linestyle=":", linewidth=1.5,
                label=f"Mediana = {np.median(flat):.1f}%")
ax_hist.set_xlabel("Diferença Relativa por Neurônio (%)\n$(U_{\\mathrm{IntraSOM}} - U_{\\mathrm{Class}}) / U_{\\mathrm{Class}} \\times 100$")
ax_hist.set_ylabel("Frequência")
ax_hist.set_title(f"(a) Distribuição: {MODEL.replace('_', ' ')}\n($n = {umat_a.size}$ neurônios)")
ax_hist.legend(frameon=False)

# Right: rel_diff_pct per model (grouped bar)
colors_var = {"RECT_planar": "#2563eb", "RECT_toroid": "#ea580c"}
labels_var = {"RECT_planar": "RECT Plana", "RECT_toroid": "RECT Toroide"}
sizes_ordered = ["5x5", "7x7", "10x10", "12x12", "15x15", "20x20"]
x = np.arange(len(sizes_ordered))
width = 0.38

for i, (var, color) in enumerate(colors_var.items()):
    sub = df_div[df_div["variant"] == var].set_index("size")
    vals = [sub.loc[s, "rel_diff_pct"] if s in sub.index else 0 for s in sizes_ordered]
    ax_bar.bar(x + (i - 0.5) * width, vals, width, label=labels_var[var],
               color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(sizes_ordered)
ax_bar.set_xlabel("Tamanho da Grade")
ax_bar.set_ylabel("Diferença Relativa Média (%)")
ax_bar.set_title("(b) Divergência Média entre as Duas\nImplementações por Modelo")
ax_bar.legend(frameon=False)
ax_bar.set_ylim(0, 25)
# Reference line at theoretical ~14.6% (sqrt(2)-1 for purely diagonal)
ax_bar.axhline(((np.sqrt(2) - 1) / (0.5 * (1 + 1/np.sqrt(2))) * 100) / 2,
               color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7,
               label="Limite teórico ($\\sqrt{2}-1 \\approx 41.4\\%$ das diag.)")

plt.tight_layout()
path2 = os.path.join(FIG_DIR, "fig2_divergence_distribution.pdf")
plt.savefig(path2, bbox_inches="tight")
plt.savefig(path2.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 2 saved -> {path2}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Divergência sistemática em função do tamanho da grade
# ─────────────────────────────────────────────────────────────────────────────
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

dims = [_dim(s) for s in sizes_ordered]

for var, color, label in [
    ("RECT_planar",  "#2563eb", "RECT Plana"),
    ("RECT_toroid",  "#ea580c", "RECT Toroide"),
]:
    sub = df_div[df_div["variant"] == var].set_index("size")
    rd  = [sub.loc[s, "rel_diff_pct"]  if s in sub.index else np.nan for s in sizes_ordered]
    sr  = [sub.loc[s, "scale_ratio"]   if s in sub.index else np.nan for s in sizes_ordered]

    ax1.plot(dims, rd, marker="o", color=color, label=label,
             linewidth=1.8, markersize=6)
    ax2.plot(dims, sr, marker="s", color=color, label=label,
             linewidth=1.8, markersize=6)

ax1.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax1.set_ylabel("Diferença Relativa Média (%)")
ax1.set_title("(a) Divergência Relativa\n$\\overline{|U_A - U_B|} / \\overline{U_B} \\times 100$")
ax1.set_xticks(dims)
ax1.set_ylim(0, 25)
ax1.legend(frameon=False)
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax2.set_ylabel("Razão de Escala $\\overline{U_A} / \\overline{U_B}$")
ax2.set_title("(b) Inflação Sistemática de Valores\nIntraSOM vs. Formulação Clássica")
ax2.set_xticks(dims)
ax2.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7, label="Igualdade")
ax2.set_ylim(0.9, 1.35)
ax2.legend(frameon=False)
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
path3 = os.path.join(FIG_DIR, "fig3_systematic_divergence.pdf")
plt.savefig(path3, bbox_inches="tight")
plt.savefig(path3.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 3 saved -> {path3}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — ARI da segmentação: impacto na partição de clusters
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax = plt.subplots(figsize=(7.5, 4.0))

for var, color, label, marker in [
    ("RECT_planar",  "#2563eb", "RECT Plana",   "o"),
    ("RECT_toroid",  "#ea580c", "RECT Toroide", "s"),
]:
    sub = df_div[df_div["variant"] == var].set_index("size")
    ari = [sub.loc[s, "seg_ari"] if s in sub.index else np.nan for s in sizes_ordered]
    ax.plot(dims, ari, marker=marker, color=color, label=label,
            linewidth=1.8, markersize=6)

ax.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1.0, alpha=0.6,
           label="ARI = 1 (segmentações idênticas)")
ax.axhline(0.8, color="#f59e0b", linestyle=":", linewidth=1.0, alpha=0.6,
           label="ARI = 0.8 (referência)")

ax.fill_between(dims, [0]*6, [0.8]*6, alpha=0.04, color="#dc2626")
ax.set_xlabel("Dimensão $N$ da Grade ($N \\times N$)")
ax.set_ylabel("ARI entre segmentações IntraSOM vs. Clássica")
ax.set_title("Impacto da Escolha de Implementação na Segmentação da U-Matrix\n"
             "(ARI entre cluster labels gerados pelas duas versões)")
ax.set_xticks(dims)
ax.set_ylim(0, 1.05)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Annotate 5x5 models (most affected)
row_55p = df_div[(df_div["size"]=="5x5") & (df_div["variant"]=="RECT_planar")].iloc[0]
row_55t = df_div[(df_div["size"]=="5x5") & (df_div["variant"]=="RECT_toroid")].iloc[0]
ax.annotate(f"ARI={row_55p.seg_ari:.2f}\n(5×5 Plana)",
            xy=(5, row_55p.seg_ari), xytext=(6, row_55p.seg_ari - 0.15),
            arrowprops=dict(arrowstyle="->", color="#2563eb"), fontsize=8, color="#2563eb")
ax.annotate(f"ARI={row_55t.seg_ari:.2f}\n(5×5 Toroide)",
            xy=(5, row_55t.seg_ari), xytext=(6.5, row_55t.seg_ari - 0.25),
            arrowprops=dict(arrowstyle="->", color="#ea580c"), fontsize=8, color="#ea580c")

plt.tight_layout()
path4 = os.path.join(FIG_DIR, "fig4_segmentation_ari.pdf")
plt.savefig(path4, bbox_inches="tight")
plt.savefig(path4.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
plt.close()
print(f"[OK] Fig 4 saved -> {path4}")

print("\n[ALL FIGURES GENERATED SUCCESSFULLY]")
