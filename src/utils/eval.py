import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from nltk.cluster import KMeansClusterer
import nltk

def graph_reader(input_nodes, input_edges):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    print("\n\n##########################################################################")
    print("### Creating Graphs...")
    graph = nx.Graph()
    
    print("Nodes Loaded...%s..." % format(input_nodes))
    df_nodes = pd.read_csv(input_nodes)
    for index, row in tqdm(df_nodes.iterrows(), total=len(df_nodes)):
        node_id, name, _id, node_type, is_hub = row.values.tolist()
        graph.add_node(node_id, name=name, id=node_id, type=node_type, is_hub=is_hub)

    print("Edges Loaded...%s..." % format(input_edges))
    df_edges = pd.read_csv(input_edges)
    for index, row in tqdm(df_edges.iterrows(), total=len(df_edges)):
        #print(row.values.tolist())
        id_1, id_2, score, edge_type = row.values.tolist()
        graph.add_edge(id_1, id_2, weight=score, type=edge_type)

    print("\nThe whole graph - ingredients, food-like compounds, drug-like compounds")
    print("# of nodes in graph: %d" % nx.number_of_nodes(graph))
    print("# of edges in graph: %d" % nx.number_of_edges(graph))

    return graph

def evaluate(args, graph, vectors):
    """
    Downstream Applications
    Evaluation
    """
    print("\nEvaluation...")

    node2node_name={}
    node_name2node={}

    for node in graph.nodes():
        node_info = graph.nodes[node]
        node_name = node_info['name']
        node2node_name[node] = node_name
        node_name2node[node_name] = node

    csv = "/workspace/FlavorGraph/input/node_classification_hub.csv"
    df = pd.read_csv(csv)
    categories = df.columns

    node_name2vec={}
    for node in vectors:
        node_name = node2node_name[int(node)]
        node_name2vec[node_name] = vectors[node]

    X=[]
    y=[]
    for category in categories:
        ingredients = df[category].values
        for name in ingredients:
            try:
                vec = node_name2vec[name]
                X.append(vec)
                y.append(category)
            except:
                print(name)


    nmis = []
    for idx in range(20):
        nmi = train(X, y, seed = idx)
        print(idx, nmi)
        nmis.append(nmi)
    
    print("nmi mean: %f" % (np.mean(nmis)))
    print("nmi std: %f" % (np.std(nmis)))
    
    return

def train(X, y, seed=None):
    rng = random.Random(seed) if seed is not None else random.Random()

    NUM_CLUSTERS = 8
    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, 
        distance=nltk.cluster.util.cosine_distance,  
        repeats=100, 
        normalise=True, 
        avoid_empty_clusters=True,
        rng=rng
    )

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    
    nmi = normalized_mutual_info_score(assigned_clusters, y)
    return nmi
