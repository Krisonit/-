
import numpy as np
import os
from gensim.models import Word2Vec
from sklearn.preprocessing import Normalizer
from utils import load_processed_data, extract_traces
from config import CLEANED_LOG_PATH, ACT2VEC_OUTPUT_DIR, ACT2VEC_PARAMS

def train_act2vec(traces, params=ACT2VEC_PARAMS):
    """Обучение модели act2vec на последовательностях действий"""
    # Используем колонку 'trace' которая содержит списки действий
    sentences = traces['trace'].tolist()
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=params['min_count'],
        epochs=params['epochs'],
        sg=params['sg'],
        hs=params['hs'],
        negative=params['negative'],
        seed=42
    )
    print(f"Act2Vec обучена. Размер словаря: {len(model.wv)}")
    return model

def vectorize_traces(traces, model):
    """Векторизация трасс с L2-нормализацией"""
    X = np.array([
        np.mean([model.wv[event] for event in trace if event in model.wv], axis=0)
        if any(event in model.wv for event in trace)
        else np.zeros(ACT2VEC_PARAMS['vector_size'])
        for trace in traces['trace']
    ])
    return Normalizer(norm='l2').fit_transform(X)

def save_vectors(X, output_dir):
    """Сохранение векторизованных данных"""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "act2vec_vectors.npy"), X)

def main():
    # Загрузка и подготовка данных
    df = load_processed_data(CLEANED_LOG_PATH)
    traces = extract_traces(df)
    
    # Обучение модели
    model = train_act2vec(traces)
    
    # Векторизация данных
    X = vectorize_traces(traces, model)
    
    # Сохранение результатов
    model.save(os.path.join(ACT2VEC_OUTPUT_DIR, "act2vec_model.model"))
    save_vectors(X, ACT2VEC_OUTPUT_DIR)
    
    print(f"Модель и векторы сохранены в {ACT2VEC_OUTPUT_DIR}")

if __name__ == "__main__":
    main()