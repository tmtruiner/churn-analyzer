import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def prepare_for_clustering(X: pd.DataFrame, categorical_features: list = None) -> tuple:
    X_numeric = X.copy()

    if categorical_features is None:
        categorical_features = X_numeric.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_features:
        if col in X_numeric.columns:
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    return X_scaled, scaler, X_numeric.columns.tolist()


def transform_for_clustering(X: pd.DataFrame, scaler: StandardScaler,
                             categorical_features: list = None) -> np.ndarray:
    X_numeric = X.copy()

    if categorical_features is None:
        categorical_features = X_numeric.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_features:
        if col in X_numeric.columns:
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes

    X_scaled = scaler.transform(X_numeric)

    return X_scaled


def find_optimal_clusters(X_scaled: np.ndarray, max_clusters: int = 10,
                         method: str = 'kmeans') -> dict:
    results = {
        'n_clusters': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    for n in range(2, max_clusters + 1):
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n, random_state=42, n_init=10)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n)
        else:
            continue

        labels = clusterer.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, labels)

        results['n_clusters'].append(n)
        results['silhouette'].append(silhouette)
        results['davies_bouldin'].append(davies_bouldin)
        results['calinski_harabasz'].append(calinski_harabasz)

    return results


def perform_kmeans_clustering(X_scaled: np.ndarray, n_clusters: int = 4,
                             random_state: int = 42) -> tuple:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    return labels, kmeans


def perform_dbscan_clustering(X_scaled: np.ndarray, eps: float = 0.5,
                             min_samples: int = 5) -> tuple:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    return labels, dbscan


def perform_hierarchical_clustering(X_scaled: np.ndarray, n_clusters: int = 4,
                                   linkage: str = 'ward') -> tuple:
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(X_scaled)

    return labels, hierarchical


def analyze_clusters(X: pd.DataFrame, labels: np.ndarray, target: pd.Series = None) -> pd.DataFrame:
    X_clustered = X.copy()
    X_clustered['cluster'] = labels

    if target is not None:
        X_clustered['target'] = target

    cluster_stats = []

    for cluster_id in np.unique(labels):
        cluster_data = X_clustered[X_clustered['cluster'] == cluster_id]

        stats = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(X_clustered) * 100
        }

        numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['cluster', 'target']]

        for col in numeric_cols:
            stats[f'{col}_mean'] = cluster_data[col].mean()
            stats[f'{col}_median'] = cluster_data[col].median()

        if target is not None:
            churn_rate = cluster_data['target'].mean()
            stats['churn_rate'] = churn_rate
            stats['churn_count'] = cluster_data['target'].sum()

        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def visualize_clusters_2d(X_scaled: np.ndarray, labels: np.ndarray,
                         title: str = "Визуализация кластеров (PCA)",
                         output_path: str = None):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                       c='black', marker='x', s=50, label='Шум')
        else:
            ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1],
                       c=[color], label=f'Кластер {label}', s=50, alpha=0.6)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(output_path, dpi=150)

    return fig


def plot_cluster_metrics(results: dict, output_path: str = "plots/cluster_metrics.png"):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(results['n_clusters'], results['silhouette'], marker='o')
    axes[0].set_xlabel('Количество кластеров')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['n_clusters'], results['davies_bouldin'], marker='o', color='orange')
    axes[1].set_xlabel('Количество кластеров')
    axes[1].set_ylabel('Davies-Bouldin Index')
    axes[1].set_title('Davies-Bouldin Index (меньше = лучше)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(results['n_clusters'], results['calinski_harabasz'], marker='o', color='green')
    axes[2].set_xlabel('Количество кластеров')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Score')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def add_cluster_features(X: pd.DataFrame, labels: np.ndarray,
                        method: str = 'one_hot') -> pd.DataFrame:
    X_with_clusters = X.copy()

    if method == 'one_hot':
        cluster_dummies = pd.get_dummies(labels, prefix='cluster')
        X_with_clusters = pd.concat([X_with_clusters, cluster_dummies], axis=1)
    elif method == 'label':
        X_with_clusters['cluster'] = labels
    elif method == 'target_mean':
        pass

    return X_with_clusters


def cluster_then_classify_pipeline(X: pd.DataFrame, y: pd.Series,
                                  n_clusters: int = 4,
                                  clustering_method: str = 'kmeans',
                                  use_cluster_as_feature: bool = True) -> tuple:
    X_scaled, scaler, feature_names = prepare_for_clustering(X)

    if clustering_method == 'kmeans':
        labels, cluster_model = perform_kmeans_clustering(X_scaled, n_clusters)
    elif clustering_method == 'dbscan':
        labels, cluster_model = perform_dbscan_clustering(X_scaled)
    elif clustering_method == 'hierarchical':
        labels, cluster_model = perform_hierarchical_clustering(X_scaled, n_clusters)
    else:
        raise ValueError(f"Неизвестный метод кластеризации: {clustering_method}")

    if use_cluster_as_feature:
        X_with_clusters = add_cluster_features(X, labels, method='one_hot')
    else:
        X_with_clusters = X.copy()

    return X_with_clusters, labels, cluster_model, scaler
