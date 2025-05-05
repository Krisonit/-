import os
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.io as pio
import umap.umap_ as umap
import hdbscan
from datetime import datetime
import warnings
import numpy as np
from collections import Counter
import ast
from sklearn.metrics import f1_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def load_processed_data(file_path):
    """Загрузка данных через pandas"""
    try:
        return pd.read_csv(file_path)  
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        raise

def extract_traces(df, include_string_representation=False):
    """Извлечение трасс из датафрейма"""
    traces = df.copy()
    if include_string_representation and 'trace_str' not in traces.columns:
        traces['trace_str'] = traces['trace'].apply(lambda x: ' '.join(x))
    return traces


def perform_clustering(X, params, is_sparse=False, random_state=42, method_name=""):
    """Улучшенная кластеризация с UMAP и HDBSCAN"""
    # Уменьшение размерности
    reducer = umap.UMAP(
        n_components=2,
        random_state=random_state,
        n_neighbors=params.get('umap_n_neighbors', 15),
        min_dist=params.get('umap_min_dist', 0.1)
    )
    
    # Преобразуем в плотный формат только если это разреженная матрица
    X_embedded = reducer.fit_transform(X.toarray() if is_sparse and hasattr(X, 'toarray') else X)
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        cluster_selection_epsilon=params.get('cluster_selection_epsilon', 0.1),
        metric=params.get('metric', 'euclidean'),
        cluster_selection_method=params.get('cluster_selection_method', 'eom')
    )
    clusters = clusterer.fit_predict(X_embedded)
    
    print(f"\nРезультаты кластеризации ({method_name}):")
    print(f"Кластеров: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
    print(f"Шум: {100*(clusters == -1).mean():.1f}%")
    
    if len(set(clusters)) > 1:
        score = silhouette_score(X_embedded, clusters)
        print(f"Silhouette: {score:.3f}")
    
    return X_embedded, clusters, clusterer

def visualize_clusters(X_2d, clusters, output_dir, method_name):
    """Универсальная визуализация кластеров"""
    viz_df = pd.DataFrame(X_2d, columns=['x', 'y'])
    viz_df['cluster'] = clusters.astype(str)
    
    # Статичная визуализация
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=viz_df,
        x='x',
        y='y',
        hue='cluster',
        palette='viridis',
        alpha=0.7
    )
    plt.title(f"Кластеризация трасс пациентов ({method_name} + HDBSCAN)")
    plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_clusters.png"), dpi=300)
    plt.close()
    
    # Интерактивная визуализация
    fig = px.scatter(
        viz_df,
        x='x',
        y='y',
        color='cluster',
        title=f"Кластеризация трасс пациентов ({method_name} + HDBSCAN)",
        width=1000,
        height=800
    )
    pio.write_html(fig, os.path.join(output_dir, f"{method_name.lower()}_clusters_interactive.html"))

def save_cluster_analysis(traces, clusters, output_dir, method_name, vectorizer=None):
    """Универсальный анализ кластеров"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{method_name}_analysis_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write(f"=== Анализ кластеров ({method_name}) ===\n")
        f.write(f"Всего кластеров: {len(set(clusters)) - (1 if -1 in clusters else 0)}\n")
        f.write(f"Шумовые точки: {sum(clusters == -1)}\n\n")
        
        traces['cluster'] = clusters
        valid_clusters = traces[traces['cluster'] != -1]
        
        for cluster in valid_clusters['cluster'].unique():
            cluster_data = valid_clusters[valid_clusters['cluster'] == cluster]
            f.write(f"Кластер {cluster} (размер: {len(cluster_data)}):\n")
            
            if method_name == "BOW" and vectorizer:
                # Для BOW: топ слов
                cluster_text = ' '.join(cluster_data['trace_str'])
                bow = vectorizer.transform([cluster_text])
                top_words = sorted(
                    [(word, bow[0, idx]) for word, idx in vectorizer.vocabulary_.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
                f.write(f"Топ слов: {[w[0] for w in top_words]}\n\n")
            else:
                # Для Act2Vec/Graph2Vec: топ событий из последовательности
                if 'trace' in cluster_data.columns:
                    top_events = cluster_data['trace'].explode().value_counts().head(10)
                    f.write(f"Топ событий: {top_events.to_dict()}\n\n")
                elif 'action' in cluster_data.columns:
                    top_events = cluster_data['action'].value_counts().head(10)
                    f.write(f"Топ событий: {top_events.to_dict()}\n\n")
                else:
                    f.write("Не удалось определить топ событий (отсутствуют данные о последовательностях)\n\n")


def create_pseudo_labels(traces_df, n_clusters=3):
    """Создает псевдо-метки на основе частых событий в трассах (альтернативная реализация)"""
    from collections import Counter
    import ast
    import numpy as np
    
    pseudo_labels = np.zeros(len(traces_df), dtype=int)
    
    # Если данные уже содержат кластеры (старая версия)
    if 'cluster' in traces_df.columns:
        for cluster_id in traces_df['cluster'].unique():
            if cluster_id != -1:
                cluster_traces = traces_df[traces_df['cluster'] == cluster_id]['trace']
                all_events = []
                for trace in cluster_traces:
                    try:
                        all_events.extend(ast.literal_eval(trace))
                    except (ValueError, SyntaxError):
                        continue
                if all_events:
                    pseudo_labels[traces_df['cluster'] == cluster_id] = cluster_id
        return pseudo_labels
    
    # Новая версия - создаем метки через KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    X = vectorizer.fit_transform(traces_df['trace'].apply(lambda x: ' '.join(x)))
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)

def evaluate_clustering(true_labels, clusters):
    """Оценка кластеризации с помощью F1-score и других метрик"""
    metrics = {}
    if true_labels is not None:
        metrics['f1_score'] = f1_score(true_labels, clusters, average='weighted')
        metrics['adjusted_rand'] = adjusted_rand_score(true_labels, clusters)
    return metrics

def split_data(traces, true_labels=None, test_size=0.3, random_state=42):
    """Разделение данных на train/test"""
    if true_labels is None:
        return train_test_split(traces, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(traces, true_labels, test_size=test_size, random_state=random_state)