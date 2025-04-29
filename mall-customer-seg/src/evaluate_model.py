from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
import numpy as np

def compute_silhouette_scores(X, k_range, random_state=42):
    scores = []
    for k in k_range:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)
    return scores

def plot_silhouette_scores(k_range, scores, output_path="../results/plots/silhouette_scores.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, marker='o')
    plt.title("Silhouette Scores for Different k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved silhouette score plot to {output_path}")
    
    
def evaluate_dbscan_silhouette(X, labels):
    # Remove noise points
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return -1  # not enough clusters
    return silhouette_score(X[mask], labels[mask])
