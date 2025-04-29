from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA   
import pandas as pd

def compute_elbow(X_scaled, k_range=range(1, 11), random_state=42):
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    return inertias

def plot_elbow(k_range, inertias):
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)
    plt.show()

def train_kmeans(X, n_clusters=5, random_state=42):
    print(f"🎯 Training KMeans with k={n_clusters}")
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    model.fit(X)
    print("✅ KMeans training complete.")
    return model, model.labels_

def train_dbscan(X, eps=0.5, min_samples=5):
    print(f"🎯 Training DBSCAN with eps={eps} and min_samples={min_samples}")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(labels == -1)
    print(f"✅ DBSCAN clustering complete. Found {n_clusters} clusters")
    print(f"🗑️ Noise points: {n_noise}")
    return model, labels

def apply_pca(X, n_components=2):
    print(f"🔍 Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    print(f"✅ PCA complete. Explained variance ratio: {explained}")
    return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    