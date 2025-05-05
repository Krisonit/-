import pandas as pd
import pm4py
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    print("1. ЗАГРУЗКА ДАННЫХ И ФОРМИРОВАНИЕ ЛОГА")
    
    try:
        # Загрузка CSV с явным преобразованием DateTime
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Проверка наличия колонок
        required_columns = ['patient', 'action', 'org:resource', 'DateTime']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")
        
        # Преобразование DateTime в datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        if df['DateTime'].isna().any():
            raise ValueError("Некорректные значения в колонке DateTime")
        
        # Сортировка по пациенту и времени
        df = df.sort_values(['patient', 'DateTime'])
        
        # Переименование колонок для совместимости с pm4py
        df = df.rename(columns={
            'patient': 'case:concept:name',
            'action': 'concept:name',
            'DateTime': 'time:timestamp',
            'org:resource': 'org:resource'
        })
        
        # Конвертация в EventLog
        event_log = pm4py.convert_to_event_log(df)
        print(f"Загружено {len(event_log)} кейсов (трасс)")
        
        # Пример вывода первых 3 трасс
        print("\nПримеры трасс:")
        for i, trace in enumerate(event_log[:3]):
            print(f"\nКейс {i + 1} (пациент {trace.attributes['concept:name']}):")
            for event in trace:
                print(f"  {event['time:timestamp']} — {event['concept:name']} ({event['org:resource']})")
        
        return event_log
    
    except Exception as e:
        print(f"\nОШИБКА ПРИ ЗАГРУЗКЕ: {str(e)}")
        raise

def clean_data(event_log):
    print("\n2. ОЧИСТКА ДАННЫХ")
    initial_count = len(event_log)
    
    try:
        # Удаление пустых трасс
        filtered_log = pm4py.filter_log(lambda trace: len(trace) > 0, event_log)
        
        # Удаление дубликатов
        filtered_log = pm4py.remove_duplicates(filtered_log)
        
        print(f"Удалено {initial_count - len(filtered_log)} кейсов (дубликаты/пустые)")
        print(f"Осталось {len(filtered_log)} кейсов после очистки")
        return filtered_log
    
    except Exception as e:
        print(f"Ошибка при очистке данных: {str(e)}")
        return event_log

def prepare_for_clustering(event_log):
    print("\n3. ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
    
    try:
        # Извлечение трасс в DataFrame
        traces = []
        for trace in event_log:
            patient = trace.attributes['concept:name']
            actions = [event['concept:name'] for event in trace]
            resources = [event['org:resource'] for event in trace]
            combined_actions = [f"{act} ({res})" for act, res in zip(actions, resources)]
            timestamps = [event['time:timestamp'] for event in trace]
            
            # Вычисление временных характеристик
            time_deltas = pd.Series(timestamps).diff().dt.total_seconds().dropna()
            
            traces.append({
                'patient': patient,
                'event_count': len(trace),
                'mean_time_between': time_deltas.mean() if len(time_deltas) > 0 else 0,
                'std_time_between': time_deltas.std() if len(time_deltas) > 0 else 0,
                'unique_actions': len(set(actions)),
                'unique_resources': len(set(resources)),
                'trace': combined_actions,
                'trace_str': ' → '.join(combined_actions),
                'actions_str': ' → '.join(actions),
                'resources_str': ' → '.join(resources)
            })
        
        # Создание DataFrame
        features_df = pd.DataFrame(traces)
        
        # Нормализация числовых признаков
        numeric_cols = ['event_count', 'mean_time_between', 'std_time_between', 
                       'unique_actions', 'unique_resources']
        scaler = StandardScaler()
        features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
        
        # Сохранение в CSV
        features_df.to_csv('prepared_data.csv', index=False)
        print("\nПодготовленные данные сохранены в файл 'prepared_data.csv'")
        
        # Вывод информации о данных
        print("\nСтатистика подготовленных данных:")
        print(features_df[numeric_cols].describe())
        
        return features_df
    
    except Exception as e:
        print(f"Ошибка при подготовке данных: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Загрузка и подготовка данных
        raw_log = load_and_prepare_data("Hospital.csv")
        
        # Очистка данных
        cleaned_log = clean_data(raw_log)
        
        # Подготовка для кластеризации
        prepared_data = prepare_for_clustering(cleaned_log)
        
        # Вывод примеров подготовленных данных
        print("\nПримеры подготовленных данных:")
        print(prepared_data[['patient', 'event_count', 'trace_str']].head(3))
        
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
        print("Проверьте структуру входного файла (требуются колонки: patient, action, org:resource, DateTime).")