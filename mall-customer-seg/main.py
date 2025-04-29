import pandas as pd
import numpy as np
from src.config import PROCESSED_DATA_PATH
from src.clustering import (
    compute_elbow, plot_elbow, train_kmeans,
    apply_pca, train_dbscan
)
from src.visualization import (
    save_cluster_plot, plot_clusters_2d, plot_dbscan_clusters,
)
from src.evaluate_model import compute_silhouette_scores, plot_silhouette_scores, evaluate_dbscan_silhouette
'''
# Load preprocessed data
X_scaled = pd.read_csv(PROCESSED_DATA_PATH)

# Define range of k to test
k_range = range(2, 11)

# -------------------------
# 💠 KMEANS CLUSTERING PIPELINE
# -------------------------

# 1. Elbow Method
inertias = compute_elbow(X_scaled, k_range)
plot_elbow(k_range, inertias)

# 2. Save Cluster Plots for Each k
for k in k_range:
    model, labels = train_kmeans(X_scaled, n_clusters=k)
    save_cluster_plot(X_scaled, labels, k)

# 3. Silhouette Scores
silhouette_scores = compute_silhouette_scores(X_scaled, k_range)
plot_silhouette_scores(k_range, silhouette_scores)

# 4. Final KMeans
k_optimal = 6  # Based on elbow + silhouette
print(f"🎯 Training final KMeans with optimal k={k_optimal}")
final_model, final_labels = train_kmeans(X_scaled, n_clusters=k_optimal)
plot_clusters_2d(X_scaled, final_labels)'''

# -------------------------
# 🧪 DBSCAN CLUSTERING PIPELINE
# -------------------------

# Load data
X_scaled = pd.read_csv(PROCESSED_DATA_PATH).values

best_score = -1
best_model = None
best_labels = None
best_params = {}

# Grid search over eps and min_samples
eps_values = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
min_samples_values = [3, 4, 5, 6]

for eps in eps_values:
    for min_samples in min_samples_values:
        print(f"\n🔍 Trying DBSCAN with eps={eps}, min_samples={min_samples}")
        model, labels = train_dbscan(X_scaled, eps=eps, min_samples=min_samples)

        score = evaluate_dbscan_silhouette(X_scaled, np.array(labels))
        print(f"🔢 Silhouette Score (no noise): {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_labels = labels
            best_params = {'eps': eps, 'min_samples': min_samples}

# Save best plot
print(f"\n✅ Best DBSCAN config: eps={best_params['eps']}, min_samples={best_params['min_samples']}, silhouette={best_score:.4f}")
plot_dbscan_clusters(X_scaled, best_labels, output_path="../results/plots/best_dbscan.png")
