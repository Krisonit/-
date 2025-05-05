import numpy as np
import os
import pickle
import pandas as pd
from utils import perform_clustering, visualize_clusters, save_cluster_analysis
from config import GRAPH2VEC_OUTPUT_DIR, HDBSCAN_PARAMS_GRAPH2VEC, CLEANED_LOG_PATH
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_graph2vec_artifacts():
    """Загрузка модели и векторов"""
    with open(os.path.join(GRAPH2VEC_OUTPUT_DIR, "graph2vec_model.pkl"), 'rb') as f:
        model = pickle.load(f)
    X = np.load(os.path.join(GRAPH2VEC_OUTPUT_DIR, "graph2vec_vectors.npy"))
    return model, X

def save_cluster_results(clusters, X_embedded, output_dir):
    """Сохранение результатов кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "graph2vec_clusters.npy"), clusters)
    np.save(os.path.join(output_dir, "graph2vec_embeddings.npy"), X_embedded)

def main():
    # Загрузка данных
    model, X = load_graph2vec_artifacts()
    df = pd.read_csv(CLEANED_LOG_PATH)
    traces = df.copy()

    # Кластеризация
    X_embedded, clusters, clusterer = perform_clustering(
        X, 
        HDBSCAN_PARAMS_GRAPH2VEC,
        is_sparse=False
    )

    # Сохранение результатов
    save_cluster_results(clusters, X_embedded, GRAPH2VEC_OUTPUT_DIR)
    visualize_clusters(X_embedded, clusters, GRAPH2VEC_OUTPUT_DIR, "graph2vec")
    
    # Анализ кластеров
    traces['cluster'] = clusters
    save_cluster_analysis(traces, clusters, GRAPH2VEC_OUTPUT_DIR, "graph2vec")
    
if __name__ == "__main__":
    main()