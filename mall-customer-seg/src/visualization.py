import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PLOTS_DIR

def save_cluster_plot(X, labels, k, output_dir=PLOTS_DIR):
    """
    Saves a 2D scatterplot of clustering result for a given k.
    """
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(X, pd.DataFrame):
        x = X['Annual Income (k$)']
        y = X['Spending Score (1-100)']
    else:
        x = X[:, 1]  # Annual Income (k$)
        y = X[:, 2]  # Spending Score (1-100)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=labels, palette='tab10')
    plt.title(f"K-Means Clustering (k={k})")
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend(title="Cluster")
    plt.grid(True)

    output_path = os.path.join(output_dir, f"kmeans_k_{k}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📸 Saved cluster plot for k={k} to {output_path}")

def plot_clusters_2d(X, labels, x_label='Annual Income (k$)', y_label='Spending Score (1-100)'):
    """
    Displays a 2D scatterplot of the final clustering result.
    """
    if isinstance(X, pd.DataFrame):
        x = X[x_label]
        y = X[y_label]
    else:
        x, y = X[:, 1], X[:, 2]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=labels, palette='tab10')
    plt.title("Final K-Means Clustering")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.show()
    
def plot_pca_clusters(X_pca, labels, output_path="results/plots/pca_dbscan_clusters.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca["PC1"], y=X_pca["PC2"], hue=labels, palette="tab10", legend="full")
    plt.title("PCA Projection with Cluster Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"📸 Saved PCA cluster plot to {output_path}")
    
    
def plot_dbscan_clusters(X, labels, feature_x='Annual Income (k$)', feature_y='Spending Score (1-100)', output_path='../results/plots/dbscan_original_space.png'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(X, pd.DataFrame):
        x = X[feature_x]
        y = X[feature_y]
    else:
        x = X[:, 1]  # assuming income is at index 1
        y = X[:, 2]  # spending score at index 2

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, hue=labels, palette='tab10', legend='full')
    plt.title("DBSCAN Clusters (Original Feature Space)")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"📸 Saved DBSCAN cluster plot to {output_path}")
