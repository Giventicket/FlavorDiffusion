import json
import os
import random
import pandas as pd
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

def set_seed(seed):
    """
    Set seed for reproducibility across random operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For CUDA operations

def graph_reader(input_nodes, input_edges):
    """
    Function to read the graph from the path.
    :param input_nodes: Path to the node CSV file.
    :param input_edges: Path to the edge CSV file.
    :return: Two NetworkX graphs (graph_ingr_comp, graph_ingr_ingr).
    """
    print("\n\n##########################################################################")
    print("### Creating Graphs...")
    graph = nx.Graph()

    print("Nodes Loaded...%s..." % format(input_nodes))
    df_nodes = pd.read_csv(input_nodes)
    for _, row in tqdm(df_nodes.iterrows(), total=len(df_nodes), desc="Processing Nodes"):
        node_id, name, _id, node_type, is_hub = row.values.tolist()
        graph.add_node(node_id, name=name, id=node_id, type=node_type, is_hub=is_hub)

    print("Edges Loaded...%s..." % format(input_edges))
    df_edges = pd.read_csv(input_edges)
    for _, row in tqdm(df_edges.iterrows(), total=len(df_edges), desc="Processing Edges"):
        id_1, id_2, score, edge_type = row.values.tolist()
        if edge_type == 'ingr-ingr':
            graph.add_edge(id_1, id_2, weight=score, type=edge_type)
        else:  # ingr-fcomp, ingr-dcomp
            graph.add_edge(id_1, id_2, weight=1, type=edge_type)

    return graph

def graph_to_dict(graph, graph_type):
    """
    Convert NetworkX graph to dictionary format with type annotation.
    """
    return {
        "type": graph_type,  # "ingr_comp" or "ingr_ingr"
        "nodes": [{"id": n, **d} for n, d in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in graph.edges(data=True)]
    }

import random

def extract_subgraph(graph, num_nodes=100):
    sampled_nodes = random.sample(list(graph.nodes), min(num_nodes, len(graph.nodes)))
    subgraph = graph.subgraph(sampled_nodes).copy()
    return subgraph

if __name__ == "__main__":
    seed = 1
    set_seed(seed)

    input_nodes = "/workspace/FlavorGraph/input/nodes_191120.csv"
    input_edges = "/workspace/FlavorGraph/input/edges_191120.csv"
    graph = graph_reader(input_nodes, input_edges)
    
    output_file = "/workspace/FlavorGraph/Flavor_Diffusion/output_subgraphs/recon.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    subgraphs_data = []
    num_samples = 100000
    
    for i in tqdm(range(num_samples), desc="Generating subgraphs"):
        subgraph_ingr_comp = extract_subgraph(graph, num_nodes=100)
        subgraphs_data.append(graph_to_dict(subgraph_ingr_comp, "all"))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subgraphs_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(subgraphs_data)} subgraphs to {output_file}")
