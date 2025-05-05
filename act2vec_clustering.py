import numpy as np
import os
import pandas as pd
from gensim.models import Word2Vec
from utils import perform_clustering, visualize_clusters, save_cluster_analysis
from config import ACT2VEC_OUTPUT_DIR, HDBSCAN_PARAMS_ACT2VEC, CLEANED_LOG_PATH
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_act2vec_artifacts():
    """Загрузка модели, векторов и параметров"""
    model = Word2Vec.load(os.path.join(ACT2VEC_OUTPUT_DIR, "act2vec_model.model"))
    X = np.load(os.path.join(ACT2VEC_OUTPUT_DIR, "act2vec_vectors.npy"))
    return model, X

def save_cluster_results(clusters, X_embedded, output_dir):
    """Сохранение результатов кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "act2vec_clusters.npy"), clusters)
    np.save(os.path.join(output_dir, "act2vec_embeddings.npy"), X_embedded)

def main():
    # Загрузка данных
    model, X = load_act2vec_artifacts()
    df = pd.read_csv(CLEANED_LOG_PATH)
    traces = df.copy()

    # Кластеризация
    X_embedded, clusters, clusterer = perform_clustering(
        X, 
        HDBSCAN_PARAMS_ACT2VEC,
        is_sparse=False
    )

    # Сохранение результатов
    save_cluster_results(clusters, X_embedded, ACT2VEC_OUTPUT_DIR)
    visualize_clusters(X_embedded, clusters, ACT2VEC_OUTPUT_DIR, "act2vec")
    
    # Анализ кластеров
    traces['cluster'] = clusters
    save_cluster_analysis(traces, clusters, ACT2VEC_OUTPUT_DIR, "act2vec")
    

if __name__ == "__main__":
    main()