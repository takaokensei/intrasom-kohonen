import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib import cm
import json
import shutil

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import load_synthetic_control, CLASS_NAMES

import intrasom
from intrasom.visualization import PlotFactory

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

# Color palette for the 6 classes
CLASS_COLORS = {
    "Normal": "#1f77b4",          # Muted Blue
    "Cyclic": "#aec7e8",          # Light Blue
    "Increasing Trend": "#ff7f0e", # Orange
    "Decreasing Trend": "#ffbb78", # Light Orange/Peach
    "Upward Shift": "#2ca02c",     # Green
    "Downward Shift": "#d62728"    # Red
}

def plot_time_series_by_class(X, y, save_path="outputs/figures/time_series_classes.png"):
    """Plots all time series divided by classes, showing individual lines, mean, and std dev."""
    print("Plotting time series by class...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    
    time_steps = np.arange(1, 61)
    
    for i, class_name in enumerate(CLASS_NAMES):
        ax = axes[i]
        class_mask = (y == class_name)
        X_class = X[class_mask]
        
        # Plot individual series in light gray
        for _, row in X_class.iterrows():
            ax.plot(time_steps, row.values, color='#d3d3d3', alpha=0.3, linewidth=0.8)
            
        # Plot mean series
        mean_series = X_class.mean(axis=0)
        std_series = X_class.std(axis=0)
        
        ax.plot(time_steps, mean_series, color=CLASS_COLORS[class_name], linewidth=2.5, label=f"Média: {class_name}")
        ax.fill_between(time_steps, mean_series - std_series, mean_series + std_series, 
                        color=CLASS_COLORS[class_name], alpha=0.2, label="Desvio Padrão")
        
        ax.set_title(f"Classe: {class_name}", fontsize=14, fontweight='bold', color='#333333')
        ax.set_ylabel("Valor", fontsize=11)
        if i >= 4:
            ax.set_xlabel("Passo de Tempo", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Time series plot saved to {save_path}")

def plot_dimensionality_reduction(X, y, save_path="outputs/figures/dim_reduction_comparison.png"):
    """Plots PCA and t-SNE projections as comparison baselines."""
    print("Plotting dimensionality reduction projections...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    # Scale data
    X_scaled = StandardScaler().fit_transform(X)
    
    # Run PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for class_name in CLASS_NAMES:
        mask = (y == class_name)
        color = CLASS_COLORS[class_name]
        
        # PCA
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=class_name, alpha=0.8, edgecolors='none', s=45)
        # t-SNE
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=class_name, alpha=0.8, edgecolors='none', s=45)
        
    ax1.set_title("Projeção PCA (2 Componentes)", fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"PC1 (Var Expl: {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    ax1.set_ylabel(f"PC2 (Var Expl: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    ax1.legend(loc="best", frameon=True)
    
    ax2.set_title("Projeção t-SNE", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Dimensão 1", fontsize=12)
    ax2.set_ylabel("Dimensão 2", fontsize=12)
    ax2.legend(loc="best", frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Dimensionality reduction plot saved to {save_path}")

def generate_som_plots(size_name="10x10"):
    """Generates standard U-Matrix, dominant class maps, and purity maps for a given SOM model."""
    print(f"Generating plots for SOM {size_name}...")
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(workspace_dir, "outputs", "maps")
    
    # Load model
    X, y = load_synthetic_control()
    
    neurons_file = os.path.join(maps_dir, f"SOM_{size_name}_neurons.parquet")
    results_file = os.path.join(maps_dir, f"SOM_{size_name}_results.parquet")
    params_file = os.path.join(maps_dir, f"params_SOM_{size_name}.json")
    
    if not (os.path.exists(neurons_file) and os.path.exists(results_file) and os.path.exists(params_file)):
        print(f"Files for SOM {size_name} not found, skipping.")
        return
        
    neurons_df = pd.read_parquet(neurons_file)
    results_df = pd.read_parquet(results_file)
    params = json.load(open(params_file))
    
    # Load SOM using IntraSOM load
    som = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
    
    # 1. Standard U-Matrix and Hits map from PlotFactory
    plot_f = PlotFactory(som)
    
    # Save U-Matrix without hits
    plot_f.plot_umatrix(figsize=(10, 8), hits=False, save=True, file_name=f"umatrix_{size_name}")
    # Save U-Matrix with hits
    plot_f.plot_umatrix(figsize=(10, 8), hits=True, save=True, file_name=f"umatrix_hits_{size_name}")
    
    # 2. Custom Hex Grid Plot for Dominant Class and Purity
    # Each sample has a BMU in results_df (which is 1-indexed, matching BMU column).
    # Let's map samples to their class labels
    results_df['Class'] = y
    
    # Group by BMU and Class to count
    counts = results_df.groupby(['BMU', 'Class']).size().unstack(fill_value=0)
    
    # Total samples per BMU
    totals = counts.sum(axis=1)
    
    # Dominant class per BMU
    dominant_class = counts.idxmax(axis=1)
    # Purity per BMU (dominant count / total count)
    purity = counts.max(axis=1) / totals
    
    # Map size
    cols, rows = som.mapsize
    
    # Let's get visual coordinates for each neuron (1 to cols*rows)
    # generate_hex_lattice returns (cols * rows, 2)
    coords = plot_f.generate_hex_lattice(cols, rows)
    
    # Plot Dominant Class Map
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    
    # Draw background grid
    for idx in range(cols * rows):
        bmu_idx = idx + 1
        cx, cy = coords[idx]
        
        # Determine color and text
        if bmu_idx in totals.index and totals[bmu_idx] > 0:
            cname = dominant_class[bmu_idx]
            face_color = CLASS_COLORS[cname]
            pur = purity[bmu_idx]
            num_samples = totals[bmu_idx]
            label_text = f"N{bmu_idx}\n{cname[:4]}\n{num_samples}s ({pur*100:.0f}%)"
            alpha = 0.85
        else:
            face_color = "#f0f0f0" # empty neuron
            label_text = f"N{bmu_idx}\nVazio"
            alpha = 0.3
            
        hex_patch = RegularPolygon((cx * 2, cy * 2), numVertices=6, radius=1.05/np.sqrt(3),
                                   facecolor=face_color, edgecolor='#444444', alpha=alpha, linewidth=0.5)
        ax.add_patch(hex_patch)
        ax.text(cx * 2, cy * 2, label_text, ha='center', va='center', fontsize=6, fontweight='bold', color='#111111')
        
    ax.set_xlim(coords[:, 0].min() * 2 - 1.5, coords[:, 0].max() * 2 + 1.5)
    ax.set_ylim(coords[:, 1].min() * 2 - 1.5, coords[:, 1].max() * 2 + 1.5)
    ax.axis('off')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='#333333', label=cname) for cname, color in CLASS_COLORS.items()]
    legend_elements.append(Patch(facecolor='#f0f0f0', edgecolor='#444444', label='Vazio', alpha=0.5))
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.0), title="Classes Dominantes", fontsize=11, title_fontsize=12)
    
    ax.set_title(f"Mapa de Classes Dominantes e Pureza - SOM {size_name}", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_dir = os.path.join(workspace_dir, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, f"som_{size_name}_dominant_classes.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dominant class map for {size_name} saved to outputs/figures/")

def generate_collage_component_planes():
    """Generates a collage of component planes using PlotFactory for the 10x10 SOM model."""
    print("Generating Component Planes collage...")
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    maps_dir = os.path.join(workspace_dir, "outputs", "maps")
    
    # Load 10x10 model
    X, y = load_synthetic_control()
    neurons_df = pd.read_parquet(os.path.join(maps_dir, "SOM_10x10_neurons.parquet"))
    params = json.load(open(os.path.join(maps_dir, "params_SOM_10x10.json")))
    som = intrasom.SOMFactory.load_som(data=X, trained_neurons=neurons_df, params=params)
    
    plot_f = PlotFactory(som)
    
    # Component planes collage.
    os.makedirs("Plots/Component_Planes", exist_ok=True)
    # Or plot_f.component_plot_collage will automatically build for all.
    # Let's run component_plot_collage.
    plot_f.component_plot_collage(grid=(3, 2), wich=[0, 14, 29, 44, 59])
    
    # Let's see where it saved. By default, it saves in Plots/Component_Planes or Plots/Collage.
    # Let's check folders in Plots/ and copy the collage file to outputs/figures
    fig_dir = os.path.join(workspace_dir, "outputs", "figures")
    for root, dirs, files in os.walk("Plots"):
        for file in files:
            if file.endswith(".png") or file.endswith(".pdf"):
                shutil.move(os.path.join(root, file), os.path.join(fig_dir, f"som_component_{file}"))
                
    print("Component Planes collage completed.")

def run_all_plots():
    X, y = load_synthetic_control()
    plot_time_series_by_class(X, y)
    plot_dimensionality_reduction(X, y)
    
    # Let's generate plots for key maps: 10x10, 15x15, 20x20
    for size_name in ["10x10", "15x15", "20x20"]:
        generate_som_plots(size_name)
        
    generate_collage_component_planes()
    
    # Move all files from Plots directory recursively to outputs/figures
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(workspace_dir, "outputs", "figures")
    
    if os.path.exists("Plots"):
        print("Moving generated plots to outputs/figures/...")
        for root, dirs, files in os.walk("Plots"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(fig_dir, file)
                    if os.path.exists(dest_file):
                        os.remove(dest_file)
                    shutil.move(src_file, dest_file)
        shutil.rmtree("Plots")
        
    if os.path.exists("Plotagens"):
        shutil.rmtree("Plotagens")
        
    print("\nAll plots generated and saved to outputs/figures/")

if __name__ == "__main__":
    run_all_plots()
