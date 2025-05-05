import os

# Пути
CLEANED_LOG_PATH = "prepared_data.csv"  
BOW_OUTPUT_DIR = "output/bow_results"
ACT2VEC_OUTPUT_DIR = "output/act2vec_results"
GRAPH2VEC_OUTPUT_DIR = "output/graph2vec_results"
os.makedirs(GRAPH2VEC_OUTPUT_DIR, exist_ok=True)
os.makedirs(BOW_OUTPUT_DIR, exist_ok=True)
os.makedirs(ACT2VEC_OUTPUT_DIR, exist_ok=True)

# Параметры моделей
BOW_PARAMS = {
    'min_df': 2,
    'max_df': 0.5,
    'ngram_range': (1, 1),
    'stop_words': None
}

HDBSCAN_PARAMS_BOW = {
    'min_cluster_size': 5,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.1,
    'cluster_selection_method': 'eom',
    'metric': 'euclidean'
}

ACT2VEC_PARAMS = {
    'vector_size': 64,
    'window': 5,
    'min_count': 2,
    'epochs': 20,
    'sg': 1,  
    'hs': 0,  
    'negative': 5
}

HDBSCAN_PARAMS_ACT2VEC = {
    'min_cluster_size': 7,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.1,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom'
}

GRAPH2VEC_PARAMS = {
    'dimensions': 128,
    'wl_iterations': 4,
    'min_count': 2,
    'epochs': 32
}

HDBSCAN_PARAMS_GRAPH2VEC = {
    'min_cluster_size': 7,
    'min_samples': 2,
    'cluster_selection_epsilon': 0.1,
    'metric': 'euclidean',
    'cluster_selection_method': 'eom'
}