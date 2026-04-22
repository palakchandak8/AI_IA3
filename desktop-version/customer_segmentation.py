"""
Customer Segmentation using K-Means Clustering
Dataset: Mall Customer Segmentation - Kaggle
Author: Team
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────
PALETTE = ['#1A3C5E', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

# ─────────────────────────────────────────────
# 3. DATA LOADING & EXPLORATION
# ─────────────────────────────────────────────
def load_and_explore(path: str) -> pd.DataFrame:
    """Load CSV and print basic statistics."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print("=== Dataset Info ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive stats:\n{df.describe().round(2)}")
    # Standardise column name
    df.rename(columns={
        'Annual Income (k$)': 'Annual_Income',
        'Spending Score (1-100)': 'Spending_Score',
        'CustomerID': 'CustomerID',
        'Genre': 'Gender'
    }, inplace=True)
    return df


# ─────────────────────────────────────────────
# 4. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def plot_eda(df: pd.DataFrame, out_dir: str = "."):
    """Generate EDA plots and save as PNG."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Exploratory Data Analysis – Mall Customer Dataset",
                 fontsize=16, fontweight='bold', y=1.01)

    # 1. Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[0, 0].pie(gender_counts, labels=gender_counts.index,
                   colors=[PALETTE[0], PALETTE[2]], autopct='%1.1f%%',
                   startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0, 0].set_title('Gender Distribution', fontweight='bold')

    # 2. Age distribution
    axes[0, 1].hist(df['Age'], bins=20, color=PALETTE[1], edgecolor='white', linewidth=0.8)
    axes[0, 1].set_title('Age Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Age'); axes[0, 1].set_ylabel('Count')

    # 3. Annual Income
    axes[0, 2].hist(df['Annual_Income'], bins=20, color=PALETTE[2], edgecolor='white', linewidth=0.8)
    axes[0, 2].set_title('Annual Income Distribution (k$)', fontweight='bold')
    axes[0, 2].set_xlabel('Annual Income (k$)')

    # 4. Spending Score
    axes[1, 0].hist(df['Spending_Score'], bins=20, color=PALETTE[3], edgecolor='white', linewidth=0.8)
    axes[1, 0].set_title('Spending Score Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Spending Score (1–100)')

    # 5. Income vs Score
    axes[1, 1].scatter(df['Annual_Income'], df['Spending_Score'],
                       alpha=0.6, color=PALETTE[0], edgecolors='white', linewidth=0.5, s=60)
    axes[1, 1].set_title('Income vs Spending Score', fontweight='bold')
    axes[1, 1].set_xlabel('Annual Income (k$)'); axes[1, 1].set_ylabel('Spending Score')

    # 6. Correlation heatmap
    corr = df[['Age', 'Annual_Income', 'Spending_Score']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                ax=axes[1, 2], linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[1, 2].set_title('Correlation Matrix', fontweight='bold')

    plt.tight_layout()
    path = f"{out_dir}/eda_plots.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")
    return path


# ─────────────────────────────────────────────
# 5. OPTIMAL K – ELBOW + SILHOUETTE
# ─────────────────────────────────────────────
def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 11), out_dir: str = "."):
    """Plot Elbow curve and Silhouette scores to select K."""
    inertia, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Optimal K Selection", fontsize=14, fontweight='bold')

    ax1.plot(list(k_range), inertia, 'o-', color=PALETTE[0], linewidth=2, markersize=7)
    ax1.set_title('Elbow Method (Within-Cluster SSE)', fontweight='bold')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.axvline(x=5, color=PALETTE[4], linestyle='--', alpha=0.7, label='K=5 (chosen)')
    ax1.legend()

    ax2.plot(list(k_range), sil_scores, 's-', color=PALETTE[2], linewidth=2, markersize=7)
    ax2.set_title('Silhouette Score vs K', fontweight='bold')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.axvline(x=5, color=PALETTE[4], linestyle='--', alpha=0.7, label='K=5 (chosen)')
    ax2.legend()

    plt.tight_layout()
    path = f"{out_dir}/elbow_silhouette.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"Best K by Silhouette: {best_k}  |  Chosen K: 5")
    return 5, inertia, sil_scores


# ─────────────────────────────────────────────
# 6. K-MEANS CLUSTERING
# ─────────────────────────────────────────────
def run_kmeans(df: pd.DataFrame, features: list, k: int = 5):
    """Fit K-Means, return df with cluster labels and scaler."""
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)

    df = df.copy()
    df['Cluster'] = labels

    sil = silhouette_score(X_scaled, labels)
    db  = davies_bouldin_score(X_scaled, labels)
    print(f"\n=== Clustering Metrics ===")
    print(f"Silhouette Score : {sil:.4f}  (higher = better, max 1)")
    print(f"Davies-Bouldin   : {db:.4f}  (lower  = better)")
    return df, km, scaler, X_scaled, sil, db


# ─────────────────────────────────────────────
# 7. VISUALISATION – 2D CLUSTERS
# ─────────────────────────────────────────────
SEGMENT_LABELS = {
    0: "Careful Spenders",
    1: "Standard Customers",
    2: "Target Customers",
    3: "Careless Spenders",
    4: "Sensible Customers"
}

def plot_clusters_2d(df: pd.DataFrame, out_dir: str = "."):
    """Annual Income vs Spending Score coloured by cluster."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for c in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c]
        ax.scatter(sub['Annual_Income'], sub['Spending_Score'],
                   color=PALETTE[c % len(PALETTE)], label=SEGMENT_LABELS.get(c, f'Cluster {c}'),
                   alpha=0.8, s=80, edgecolors='white', linewidth=0.7)

    ax.set_title('K-Means Clustering: Annual Income vs Spending Score',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1–100)', fontsize=12)
    ax.legend(title='Segments', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)

    path = f"{out_dir}/clusters_2d.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")
    return path


# ─────────────────────────────────────────────
# 8. VISUALISATION – 3D PCA
# ─────────────────────────────────────────────
def plot_clusters_pca(df: pd.DataFrame, X_scaled: np.ndarray, out_dir: str = "."):
    """3-feature PCA projection coloured by cluster."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df2 = df.copy()
    df2['PC1'] = coords[:, 0]
    df2['PC2'] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in sorted(df2['Cluster'].unique()):
        sub = df2[df2['Cluster'] == c]
        ax.scatter(sub['PC1'], sub['PC2'],
                   color=PALETTE[c % len(PALETTE)], label=SEGMENT_LABELS.get(c, f'Cluster {c}'),
                   alpha=0.8, s=70, edgecolors='white', linewidth=0.6)

    ax.set_title('PCA Projection of Customer Segments', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.legend(title='Segments', fontsize=9)
    ax.grid(True, alpha=0.3)

    path = f"{out_dir}/clusters_pca.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")
    return path


# ─────────────────────────────────────────────
# 9. CLUSTER PROFILING
# ─────────────────────────────────────────────
def profile_clusters(df: pd.DataFrame, out_dir: str = "."):
    """Build cluster statistics table and bar plots."""
    profile = df.groupby('Cluster').agg(
        Count=('Cluster', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Income=('Annual_Income', 'mean'),
        Avg_Score=('Spending_Score', 'mean')
    ).round(1)
    profile['Label'] = [SEGMENT_LABELS[i] for i in profile.index]
    profile['% Female'] = df.groupby('Cluster')['Gender'].apply(
        lambda x: (x == 'Female').mean() * 100).round(1).values
    print("\n=== Cluster Profiles ===")
    print(profile.to_string())

    # Grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cluster Profiles – Average Features", fontsize=13, fontweight='bold')
    metrics = [('Avg_Age', 'Average Age', 'Years'),
               ('Avg_Income', 'Average Annual Income', 'k$'),
               ('Avg_Score', 'Average Spending Score', 'Score')]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(profile['Label'], profile[col],
                      color=PALETTE[:len(profile)], edgecolor='white', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(profile['Label'], rotation=30, ha='right', fontsize=9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = f"{out_dir}/cluster_profiles.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")
    return path, profile


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys

    DATA_PATH = "Mall_Customers.csv"
    OUT_DIR   = "plots"
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Step 1: Load ---
    df = load_and_explore(DATA_PATH)

    # --- Step 2: EDA ---
    plot_eda(df, OUT_DIR)

    # --- Step 3: Feature selection & scaling for elbow ---
    features = ['Annual_Income', 'Spending_Score']
    X = df[features].values
    scaler = StandardScaler()
    X_scaled_2 = scaler.fit_transform(X)

    # --- Step 4: Find K ---
    best_k, inertia, sil_scores = find_optimal_k(X_scaled_2, out_dir=OUT_DIR)

    # --- Step 5: Cluster (2-feature) ---
    df, km, scaler_fit, X_scaled_fit, sil, db = run_kmeans(df, features, k=5)

    # --- Step 6: Plots ---
    plot_clusters_2d(df, OUT_DIR)

    # 3-feature version for PCA
    features_3 = ['Age', 'Annual_Income', 'Spending_Score']
    _, _, _, X3_scaled, _, _ = run_kmeans(df, features_3, k=5)
    plot_clusters_pca(df, X3_scaled, OUT_DIR)

    # --- Step 7: Profiles ---
    profile_path, profile_df = profile_clusters(df, OUT_DIR)

    print("\n✅ All done! Charts saved to ./plots/")
    print(f"   Silhouette: {sil:.4f}  |  Davies-Bouldin: {db:.4f}")
