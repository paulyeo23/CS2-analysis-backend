import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from collections import defaultdict
from pyvis.network import Network
from sklearn.metrics import confusion_matrix


class MarkovMap:
    def __init__(self, mapfile: str):
    #def __init__(self, transitions: dict):
        #refactor for reading the prob matrix
        print("Starting Markov Map")
        tm = self.build_transitions_from_csv(mapfile)
        self.transitions = tm["neighbors"]
        self.probabilities = tm["probs"]
        self.states = list(self.transitions.keys())
        self.transition_matrix = self.build_prob_matrix()
        #self.transitions = transitions
        #self.transitions = self.build_transitions_from_csv(mapfile)
        #print(self.transitions)
        #self.states = list(self.transitions.keys())
        #self.transition_matrix = self.build_equal_prob_matrix() #this is default behavior
        #this would need to be changed with the trained model

    def build_equal_prob_matrix(self):
        n = len(self.states)
        matrix = np.zeros((n, n))
        for i, s in enumerate(self.states):
            neighbors = self.transitions[s]
            for next_s in neighbors:
                j = self.states.index(next_s)
                matrix[i, j] = 1 / len(neighbors)
        return pd.DataFrame(matrix, index=self.states, columns=self.states)

    def build_prob_matrix(self):
        n = len(self.states)
        mat = np.zeros((n, n))
        #print(f"states is : {n} and matrix is : {mat.shape}")

        for s in self.states:
            i = self.states.index(s)
            next_states = self.transitions[s]
            #print(f"do we have probs: {self.probabilities}")
            if self.probabilities is None:
                #equal prob
                #print(f"length is : {len(next_states)}")
                if len(next_states) > 0:
                    #calc the prob
                    p = 1 / len(next_states)
                    for nstate in next_states:
                        j = self.states.index(nstate)
                        #add the prob
                        mat[i][j] = p
            else:
                #use real probs
                #print(f"using real probs: {self.probabilities[s]}")
                probs = self.probabilities[s]
                for next_s, prob in zip(next_states, probs):
                    j = self.states.index(next_s)
                    #append the probs
                    mat[i][j] = prob

        return pd.DataFrame(mat, index=self.states, columns=self.states)

    def next_location(self, current):
        probs = self.transition_matrix.loc[current]
        return np.random.choice(self.transition_matrix.columns, p=probs)

    def simulate_path(self, start="T Spawn", steps=10):
        path = [start]
        for _ in range(steps):
            start = self.next_location(start)
            path.append(start)
        return path

    def plot_graph(self):
        G = nx.DiGraph()
        for s, nexts in self.transitions.items():
            for n in nexts:
                G.add_edge(s, n)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx(G, with_labels=True, node_size=2000, font_size=8)
        plt.show()

    def build_transitions_from_csv(self,filepath: str) -> dict:
        print(filepath)
        #refactor this one to add the probabilities
        #df = pd.read_csv(filepath)
        #transitions = defaultdict(list)
        #for _, row in df.iterrows():
        #    transitions[row["from"]].append(row["to"])
        # Convert defaultdict back to dict
        #return dict(transitions)
        df = pd.read_csv(filepath)
        transitions = defaultdict(list)
        probs = defaultdict(list)
        #print(f"cols: {df.columns}")
        if "probability" in df.columns:
            # use weighted probabilities
            for i, row in df.iterrows():
                #print(f"row: {row} trans: {transitions} probs: {probs}")
                transitions[row["from"]].append(row["to"])
                probs[row["from"]].append(row["probability"])
            return {"neighbors": dict(transitions), "probs": dict(probs)}
        else:
            for i, row in df.iterrows():
                transitions[row["from"]].append(row["to"])
            return {"neighbors": dict(transitions), "probs": None}


    def plot_transition_graph(self):

        G = nx.DiGraph()

        # Add edges with weights (probabilities)
        for src in self.transition_matrix.index:
            for dst in self.transition_matrix.columns:
                prob = self.transition_matrix.loc[src, dst]
                if prob > 0:
                    G.add_edge(src, dst, weight=prob)

        # Layout — spring layout is general, but you can use fixed positions if you want
        pos = nx.spring_layout(G, seed=42, k=0.7)

        # Node colors
        node_colors = plt.cm.cool(np.linspace(0, 1, len(G.nodes())))

        # Edge widths proportional to probability
        edge_weights = [d['weight'] * 8 for (_, _, d) in G.edges(data=True)]

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, alpha=0.9)

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=edge_weights, arrows=True, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.2')

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black')

        # Draw edge labels with probabilities
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=8)

        plt.title("CS2 Map — Markov State Transitions with Probabilities")
        plt.axis('off')
        plt.show()

    def plot_transition_graph_interactive(self, output_file="markov_graph.html"):
        # Create interactive network
        #print("came here")
        net = Network(height="850px", width="100%", directed=True)
        net.barnes_hut()
        degree = {state: 0 for state in self.states}
        #print(f"degrees of node is {degree}")

        for src in self.transition_matrix.index:
            for dst in self.transition_matrix.columns:
                if self.transition_matrix.loc[src, dst] > 0:
                    degree[src] += 1
                    #print(f"src: {src}, dst: {dst} has degree : {degree[src]}")

        max_deg = max(degree.values()) if degree else 1
        #print(f"max_deg to calc the max node circle: {max_deg}")
        colormap = cm.get_cmap('plasma')  #  'plasma', 'cool', 'turbo'
        norm = colors.Normalize(vmin=0, vmax=max_deg)
        #print("so far here")
        for state in self.states:
            deg = degree[state]
            #size = 15 + (deg / max_deg) * 45
            size = 40 + (deg / max_deg) * 65
            #print(f"degree size of my node {state} : {deg} : {size}")
            rgba = colormap(norm(deg))
            hex_color = colors.to_hex(rgba)
            #print(f"i have everyting to add node {state}, {size}, {hex_color}")
            net.add_node(
                state,
                label=state,
                font={'size': 40},
                size=size,
                color=hex_color,
                title=f"Zone: {state}\nConnections: {deg}"
            )
        #what are my edges, add them here
        for src in self.transition_matrix.index:
            for dst in self.transition_matrix.columns:
                prob = self.transition_matrix.loc[src, dst]
                if prob > 0:
                    net.add_edge(
                        src,
                        dst,
                        value=prob,
                        weight=1+prob*5,
                        fontweight=2+prob*5,
                        title=f"Probability: {prob:.2f}",
                        label=f"{prob:.2f}"
                    )
        net.write_html(output_file)
        print(f"Interactive graph saved to: {output_file}")



    def setup_testing(self,probability_file):
        df = pd.read_csv(probability_file)
        trans = {}

        for i, row in df.iterrows():
            src = row["from"]
            dst = row["to"]
            prob = row["probability"]

            if src not in trans:
                trans[src] = {}
            trans[src][dst] = prob

        return trans

    def markov_predict_one(self,trans_prob_dict, current_zone):
        #print(f"Current Zone: {current_zone}")
        if current_zone not in trans_prob_dict:
            #print(f"not found")
            return None
        next_probs = trans_prob_dict[current_zone]
        if len(next_probs) == 0:
            return None
        #choose the zone with highest probability
        best_zone = max(next_probs, key=next_probs.get)
        return best_zone

    def run_markov_on_test(self, trans_dict, test_df, current_col="current_zone"):
        preds = []
        for i, row in test_df.iterrows():
            cur = row[current_col]
            p = self.markov_predict_one(trans_dict, cur)
            preds.append(p)
        test_df["markov_pred"] = preds
        return test_df

    def compute_accuracy(self, df, pred_col="markov_pred", label_col="next_zone"):
        correct = 0
        total = 0
        for i, row in df.iterrows():
            pred = row[pred_col]
            true = row[label_col]
            if pred is not None:
                total += 1
                if pred == true:
                    correct += 1
        if total == 0:
            return 0
        return correct / total



# Simplified Dust2 transition model
'''default_transitions = {
    "T Spawn": ["Outside Tunnels", "Outside Long", "Mid"],
    "Outside Tunnels": ["Upper B", "T Spawn"],
    "Upper B": ["Lower B", "B Site", "Outside Tunnels"],
    "Lower B": ["Mid", "Upper B"],
    "Mid": ["Lower B", "Catwalk", "CT Mid"],
    "Catwalk": ["Short A", "Mid"],
    "Short A": ["A Plat", "CT Spawn"],
    "A Plat": ["Short A", "Long A"],
    "Long A": ["Pit", "A Plat", "CT Spawn"],
    "Pit": ["Long A"],
    "CT Spawn": ["CT Mid", "Short A", "Long A"],
    "CT Mid": ["B Doors", "Mid", "CT Spawn"],
    "B Doors": ["CT Mid", "B Site"],
    "B Site": ["Upper B", "B Doors"],
    "Outside Long": ["Long Doors", "T Spawn"],
    "Long Doors": ["Outside Long", "Long A"],
}'''
