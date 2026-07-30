import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set publication-quality style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 300

fig_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(fig_dir, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: QE and TE Across Map Sizes & Variants
# -------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "metrics", "model_comparison_results.csv")
df = pd.read_csv(csv_path)

# Filter SOM rows
som_df = df[df['variant'].isin(['HEX_toroid', 'HEX_planar', 'RECT_planar', 'RECT_toroid'])].copy()

# Extract map size as integer dimension
som_df['size_dim'] = som_df['Modelo'].apply(lambda x: int(x.split()[1].split('x')[0]))
som_df = som_df.sort_values('size_dim')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

colors = {
    'HEX_toroid': '#2b5c8f',  # Dark Blue
    'HEX_planar': '#d95f02',  # Orange/Red
    'RECT_planar': '#7570b3', # Purple
    'RECT_toroid': '#1b9e77'  # Teal/Green
}

markers = {
    'HEX_toroid': 'o',
    'HEX_planar': 's',
    'RECT_planar': '^',
    'RECT_toroid': 'D'
}

labels = {
    'HEX_toroid': 'HEX Toroide',
    'HEX_planar': 'HEX Plana',
    'RECT_planar': 'RECT Plana',
    'RECT_toroid': 'RECT Toroide'
}

for var in ['HEX_toroid', 'HEX_planar', 'RECT_planar', 'RECT_toroid']:
    sub = som_df[som_df['variant'] == var]
    ax1.plot(sub['size_dim'], sub['Erro Quantização'], label=labels[var], color=colors[var], marker=markers[var], linewidth=1.8, markersize=6)
    ax2.plot(sub['size_dim'], sub['Erro Topográfico'], label=labels[var], color=colors[var], marker=markers[var], linewidth=1.8, markersize=6)

ax1.set_xlabel('Dimensão da Grade ($N \\times N$)')
ax1.set_ylabel('Erro de Quantização (QE)')
ax1.set_title('(a) Erro de Quantização ($QE \\downarrow$)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xticks([5, 7, 10, 12, 15, 20])
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

ax2.set_xlabel('Dimensão da Grade ($N \\times N$)')
ax2.set_ylabel('Erro Topográfico (TE)')
ax2.set_title('(b) Erro Topográfico ($TE \\downarrow$)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xticks([5, 7, 10, 12, 15, 20])
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
fig1_path = os.path.join(fig_dir, "fig1_qe_te_comparison.pdf")
fig1_png = os.path.join(fig_dir, "fig1_qe_te_comparison.png")
plt.savefig(fig1_path, bbox_inches='tight')
plt.savefig(fig1_png, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved Figure 1 to {fig1_path}")

# -------------------------------------------------------------
# Figure 2: Edge Count & Neighborhood Topology (HEX vs RECT)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))

sizes = ['5x5', '7x7', '10x10', '12x12', '15x15', '20x20']
hex_edges = [3*5*6, 3*7*8, 3*10*10, 3*12*12, 3*15*16, 3*20*20] # 3*N*M for Toroid HEX
rect_edges = [4*5*5, 4*7*7, 4*10*10, 4*12*12, 4*15*15, 4*20*20] # 4*N*M for Toroid RECT 8-conn

x = np.arange(len(sizes))
width = 0.35

rects1 = ax.bar(x - width/2, hex_edges, width, label='HEX Toroide (6 vizinhos, $3NM$)', color='#2b5c8f')
rects2 = ax.bar(x + width/2, rect_edges, width, label='RECT Toroide (8 vizinhos, $4NM$)', color='#d95f02')

ax.set_xlabel('Tamanho da Grade Solicitada')
ax.set_ylabel('Número Total de Arestas Únicas na U-Matrix')
ax.set_title('Contagem de Arestas da U-Matrix Expandida por Geometria')
ax.set_xticks(x)
ax.set_xticklabels(['5×5 (30 vs 25 neurônios)', '7×7 (56 vs 49 neurônios)', '10×10 (100 neurônios)', '12×12 (144 neurônios)', '15×15 (240 vs 225 neurônios)', '20×20 (400 neurônios)'], rotation=15, ha='right', fontsize=8)
ax.legend(frameon=True)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

# Annotate 10x10 values
ax.annotate('300 arestas', xy=(2 - width/2, 300), xytext=(2 - width/2 - 0.2, 420),
            arrowprops=dict(facecolor='#2b5c8f', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8, fontweight='bold', color='#2b5c8f')

ax.annotate('400 arestas (8-conn)', xy=(2 + width/2, 400), xytext=(2 + width/2 + 0.1, 550),
            arrowprops=dict(facecolor='#d95f02', shrink=0.08, width=1.5, headwidth=6),
            fontsize=8, fontweight='bold', color='#d95f02')

plt.tight_layout()
fig2_path = os.path.join(fig_dir, "fig2_edge_counts.pdf")
fig2_png = os.path.join(fig_dir, "fig2_edge_counts.png")
plt.savefig(fig2_path, bbox_inches='tight')
plt.savefig(fig2_png, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved Figure 2 to {fig2_path}")

# -------------------------------------------------------------
# Figure 3: U-Matrix Diagonal Normalization Comparison Diagram
# -------------------------------------------------------------
fig, (ax_ultsch, ax_intrasom) = plt.subplots(1, 2, figsize=(9.5, 4.0))

# Ultsch / Costa & Netto (2007) Diagram
ax_ultsch.set_xlim(-0.5, 2.5)
ax_ultsch.set_ylim(-0.5, 2.5)
ax_ultsch.set_aspect('equal')
ax_ultsch.axis('off')
ax_ultsch.set_title('Formulação Clássica (Ultsch 1993 / Costa & Netto 2007)\nPonderação Isotrópica: $d_{diag} / \\sqrt{2}$', fontsize=10, fontweight='bold')

# Draw grid points
grid_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
grid_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
ax_ultsch.scatter(grid_x, grid_y, color='#24283b', s=120, zorder=5)

# Center neuron (1,1)
ax_ultsch.scatter([1], [1], color='#7aa2f7', s=250, zorder=6)
ax_ultsch.text(1, 1, '$w_{i,j}$', color='white', fontsize=10, fontweight='bold', ha='center', va='center', zorder=7)

# Orthogonal lines
ortho_coords = [(1, 2), (2, 1), (1, 0), (0, 1)]
for cx, cy in ortho_coords:
    ax_ultsch.plot([1, cx], [1, cy], color='#2b5c8f', linewidth=2.5, zorder=3)
    ax_ultsch.text((1+cx)/2, (1+cy)/2, '1.0', color='#2b5c8f', fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2b5c8f', alpha=0.9))

# Diagonal lines
diag_coords = [(2, 2), (2, 0), (0, 0), (0, 2)]
for cx, cy in diag_coords:
    ax_ultsch.plot([1, cx], [1, cy], color='#d95f02', linestyle='--', linewidth=2.0, zorder=3)
    ax_ultsch.text((1+cx)/2 + (0.1 if cx>1 else -0.1), (1+cy)/2 + (0.1 if cy>1 else -0.1), '$1/\\sqrt{2} \\approx 0.707$', color='#d95f02', fontsize=7.5, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='#fff5eb', edgecolor='#d95f02', alpha=0.9))

# IntraSOM 1.1.1 Diagram
ax_intrasom.set_xlim(-0.5, 2.5)
ax_intrasom.set_ylim(-0.5, 2.5)
ax_intrasom.set_aspect('equal')
ax_intrasom.axis('off')
ax_intrasom.set_title('Implementação IntraSOM 1.1.1 (build_umatrix)\nNorma Euclidiana Bruta: sem fator $1/\\sqrt{2}$', fontsize=10, fontweight='bold')

ax_intrasom.scatter(grid_x, grid_y, color='#24283b', s=120, zorder=5)
ax_intrasom.scatter([1], [1], color='#bb9af7', s=250, zorder=6)
ax_intrasom.text(1, 1, '$w_{i,j}$', color='white', fontsize=10, fontweight='bold', ha='center', va='center', zorder=7)

for cx, cy in ortho_coords:
    ax_intrasom.plot([1, cx], [1, cy], color='#2b5c8f', linewidth=2.5, zorder=3)
    ax_intrasom.text((1+cx)/2, (1+cy)/2, '1.0', color='#2b5c8f', fontsize=8, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2b5c8f', alpha=0.9))

for cx, cy in diag_coords:
    ax_intrasom.plot([1, cx], [1, cy], color='#e0af68', linewidth=2.5, zorder=3)
    ax_intrasom.text((1+cx)/2 + (0.1 if cx>1 else -0.1), (1+cy)/2 + (0.1 if cy>1 else -0.1), '1.0 (bruto)', color='#e0af68', fontsize=7.5, fontweight='bold', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor='#fefae0', edgecolor='#e0af68', alpha=0.9))

plt.tight_layout()
fig3_path = os.path.join(fig_dir, "fig3_diagonal_scaling.pdf")
fig3_png = os.path.join(fig_dir, "fig3_diagonal_scaling.png")
plt.savefig(fig3_path, bbox_inches='tight')
plt.savefig(fig3_png, bbox_inches='tight', dpi=300)
plt.close()
print(f"Saved Figure 3 to {fig3_path}")
