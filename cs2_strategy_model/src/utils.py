import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def load_map_transitions(path: str) -> dict:
    """
    Reads a CSV of map transitions and returns a dictionary suitable for MarkovMap.
    CSV should have columns: from,to
    """
    df = pd.read_csv(path)
    transitions = {}
    for from_loc in df['from'].unique():
        transitions[from_loc] = df[df['from'] == from_loc]['to'].tolist()
    return transitions

def plot_markov_graph(transitions: dict):
    """
    Plots a directed graph from a transitions dictionary.
    """
    G = nx.DiGraph()
    for src, dst_list in transitions.items():
        for dst in dst_list:
            G.add_edge(src, dst)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(G, with_labels=True, node_size=2000, font_size=8)
    plt.title("Markov Map Graph")
    plt.show()

def preprocess_state(df: pd.DataFrame, scaler=None):
    """
    Scales input features for win probability model.
    Returns scaled numpy array. If scaler is None, returns original df values.
    """
    if scaler is not None:
        return scaler.transform(df)
    return df.values
