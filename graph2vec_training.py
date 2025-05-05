import numpy as np
import os
from karateclub import Graph2Vec
from utils import load_processed_data, extract_traces
from config import CLEANED_LOG_PATH, GRAPH2VEC_OUTPUT_DIR, GRAPH2VEC_PARAMS
import networkx as nx
from collections import defaultdict
import pickle

def traces_to_graphs(traces):
    """Convert traces to properly indexed directed graphs"""
    # Используем колонку 'trace' которая содержит списки действий
    all_events = set()
    for trace in traces['trace']:
        all_events.update(trace)
    
    # Create numeric mapping starting from 0
    event_to_idx = {event: idx for idx, event in enumerate(sorted(all_events))}
    
    graphs = []
    for trace in traces['trace']:
        G = nx.DiGraph()
        
        # Add nodes with consistent numeric indices
        for event in trace:
            node_id = event_to_idx[event]
            G.add_node(node_id, label=event)  # Сохраняем метку события
        
        # Add edges with weights
        edge_weights = defaultdict(int)
        for i in range(len(trace)-1):
            src = event_to_idx[trace[i]]
            dst = event_to_idx[trace[i+1]]
            edge_weights[(src, dst)] += 1
        
        for (src, dst), weight in edge_weights.items():
            G.add_edge(src, dst, weight=weight)
        
        # Ensure graph is connected (required by Graph2Vec)
        if not nx.is_weakly_connected(G):
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
        
        # Ensure nodes are numbered 0 to n-1 without gaps
        if len(G.nodes()) > 0:
            mapping = {node: idx for idx, node in enumerate(sorted(G.nodes()))}
            G = nx.relabel_nodes(G, mapping)
        
        graphs.append(G)
    
    return graphs

def train_graph2vec(graphs, params=GRAPH2VEC_PARAMS):
    """Train Graph2Vec model with proper initialization"""
    model = Graph2Vec(
        dimensions=params['dimensions'],
        wl_iterations=params['wl_iterations'],
        min_count=params['min_count'],
        epochs=params['epochs'],
        seed=42
    )
    
    model.fit(graphs)
    return model

def save_model_and_vectors(model, output_dir):
    """Save model and vectors"""
    os.makedirs(output_dir, exist_ok=True)
    X = np.array(model.get_embedding())
    np.save(os.path.join(output_dir, "graph2vec_vectors.npy"), X)
    
    # Save model using pickle
    with open(os.path.join(output_dir, "graph2vec_model.pkl"), 'wb') as f:
        pickle.dump(model, f)

def main():
    # Load and prepare data
    df = load_processed_data(CLEANED_LOG_PATH)
    traces = extract_traces(df)
    
    # Convert traces to properly indexed graphs
    graphs = traces_to_graphs(traces)
    
    # Train Graph2Vec model
    model = train_graph2vec(graphs)
    
    # Save results
    save_model_and_vectors(model, GRAPH2VEC_OUTPUT_DIR)
    
    print(f"Model and vectors saved to {GRAPH2VEC_OUTPUT_DIR}")

if __name__ == "__main__":
    main()