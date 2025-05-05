# comparison.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score 
from config import BOW_OUTPUT_DIR, ACT2VEC_OUTPUT_DIR, GRAPH2VEC_OUTPUT_DIR
import os
from utils import evaluate_clustering

def load_cluster_results(method_name):
    """Загрузка результатов кластеризации для указанного метода"""
    output_dir = {
        'BOW': BOW_OUTPUT_DIR,
        'Act2Vec': ACT2VEC_OUTPUT_DIR,
        'Graph2Vec': GRAPH2VEC_OUTPUT_DIR
    }[method_name]
    
    print(f"Загрузка данных для метода {method_name} из {output_dir}")  # Отладочный вывод
    
    clusters_path = os.path.join(output_dir, f"{method_name.lower()}_clusters.npy")
    embeddings_path = os.path.join(output_dir, f"{method_name.lower()}_embeddings.npy")
    
    print(f"Путь к кластерам: {clusters_path}")
    print(f"Путь к эмбеддингам: {embeddings_path}")
    
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Файл {clusters_path} не найден")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Файл {embeddings_path} не найден")
    
    clusters = np.load(clusters_path)
    embeddings = np.load(embeddings_path)
    
    return clusters, embeddings

def calculate_metrics(clusters, embeddings):
    """Расчет метрик качества кластеризации"""
    metrics = {}
    unique_clusters = set(clusters)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    # Silhouette Score
    if n_clusters > 1:
        metrics['silhouette'] = silhouette_score(embeddings, clusters)
    else:
        metrics['silhouette'] = None
    
    # Davies-Bouldin Index (чем меньше, тем лучше)
    if n_clusters > 1:
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, clusters)
    else:
        metrics['davies_bouldin'] = None
    
    # Calinski-Harabasz Index (чем больше, тем лучше)
    if n_clusters > 1:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, clusters)
    else:
        metrics['calinski_harabasz'] = None
    
    # Процент шума
    metrics['noise_ratio'] = (clusters == -1).mean()
    
    # Количество кластеров
    metrics['n_clusters'] = n_clusters
    
    return metrics

def plot_metrics_comparison(df):
    """Визуализация сравнения метрик"""
    plt.figure(figsize=(18, 5))
    
    # Silhouette Score
    plt.subplot(1, 4, 1)
    sns.barplot(data=df, x='method', y='silhouette')
    plt.title('Silhouette Score (↑ лучше)')
    plt.ylim(0, 1)
    
    # Davies-Bouldin Index
    plt.subplot(1, 4, 2)
    sns.barplot(data=df, x='method', y='davies_bouldin')
    plt.title('Davies-Bouldin (↓ лучше)')
    
    # Calinski-Harabasz Index
    plt.subplot(1, 4, 3)
    sns.barplot(data=df, x='method', y='calinski_harabasz')
    plt.title('Calinski-Harabasz (↑ лучше)')
    
    # Noise Ratio
    plt.subplot(1, 4, 4)
    sns.barplot(data=df, x='method', y='noise_ratio')
    plt.title('% шума (↓ лучше)')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'comparison', 'metrics_comparison.png'), dpi=300)
    plt.close()

def compare_methods():
    methods = ['BOW', 'Act2Vec', 'Graph2Vec']
    results = []
    
    for method in methods:
        # Загрузка кластеров и меток
        clusters_train = np.load(f"{method}_output/clusters_train.npy")
        clusters_test = np.load(f"{method}_output/clusters_test.npy")
        true_labels_train = np.load("true_labels_train.npy")  # Ваши метки
        true_labels_test = np.load("true_labels_test.npy")
        
        # Оценка
        metrics_train = evaluate_clustering(true_labels_train, clusters_train)
        metrics_test = evaluate_clustering(true_labels_test, clusters_test)
        
        results.append({
            'method': method,
            'f1_train': metrics_train['f1_score'],
            'f1_test': metrics_test['f1_score'],
            'n_clusters_train': len(set(clusters_train)),
            'n_clusters_test': len(set(clusters_test))
        })
    
    # Сохранение и визуализация
    df_results = pd.DataFrame(results)
    df_results.to_csv("results_comparison.csv", index=False)
    print(df_results)

def plot_metrics_comparison(df):
    """Визуализация сравнения метрик"""
    plt.figure(figsize=(15, 5))
    
    # Silhouette Score
    plt.subplot(1, 3, 1)
    sns.barplot(data=df, x='method', y='silhouette')
    plt.title('Сравнение Silhouette Score')
    plt.ylim(0, 1)
    
    # Noise Ratio
    plt.subplot(1, 3, 2)
    sns.barplot(data=df, x='method', y='noise_ratio')
    plt.title('Сравнение % шума')
    plt.ylim(0, 1)
    
    # Number of clusters
    plt.subplot(1, 3, 3)
    sns.barplot(data=df, x='method', y='n_clusters')
    plt.title('Сравнение количества кластеров')
    
    plt.tight_layout()
    plt.savefig(os.path.join('output', 'comparison', 'metrics_comparison.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    compare_methods()