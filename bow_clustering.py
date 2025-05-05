import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from utils import (
    perform_clustering, 
    visualize_clusters, 
    save_cluster_analysis, 
    extract_traces, 
    evaluate_clustering,
    create_pseudo_labels  # Теперь импортируется корректно
)
from config import BOW_OUTPUT_DIR, HDBSCAN_PARAMS_BOW, CLEANED_LOG_PATH

def load_bow_artifacts():
    """Загрузка BOW артефактов с train/test"""
    X_train = np.load(os.path.join(BOW_OUTPUT_DIR, "bow_vectors_train.npy"))
    X_test = np.load(os.path.join(BOW_OUTPUT_DIR, "bow_vectors_test.npy"))
    with open(os.path.join(BOW_OUTPUT_DIR, "bow_vectorizer.pkl"), 'rb') as f:
        vectorizer = pickle.load(f)
    return X_train, X_test, vectorizer

def main():
    # Загрузка данных
    X_train, X_test, vectorizer = load_bow_artifacts()
    df = pd.read_csv(CLEANED_LOG_PATH)
    traces = extract_traces(df, include_string_representation=True)
    
    # Разделение данных
    traces_train, traces_test = train_test_split(traces, test_size=0.3, random_state=42)
    
    # Создание псевдо-меток
    true_labels_train = create_pseudo_labels(traces_train)
    true_labels_test = create_pseudo_labels(traces_test)
    
    # Кластеризация
    X_embedded_train, clusters_train, _ = perform_clustering(
        X_train, 
        HDBSCAN_PARAMS_BOW,
        method_name="BOW (Train)"
    )
    
    X_embedded_test, clusters_test, _ = perform_clustering(
        X_test, 
        HDBSCAN_PARAMS_BOW,
        method_name="BOW (Test)"
    )
    
    # Оценка
    metrics_train = evaluate_clustering(true_labels_train, clusters_train)
    metrics_test = evaluate_clustering(true_labels_test, clusters_test)
    print(f"Train F1: {metrics_train.get('f1_score', 0):.3f}, Test F1: {metrics_test.get('f1_score', 0):.3f}")
    
    # Сохранение
    traces_train['cluster'] = clusters_train
    traces_test['cluster'] = clusters_test
    save_cluster_analysis(traces_train, clusters_train, BOW_OUTPUT_DIR, "bow_train", vectorizer)
    save_cluster_analysis(traces_test, clusters_test, BOW_OUTPUT_DIR, "bow_test", vectorizer)

if __name__ == "__main__":
    main()