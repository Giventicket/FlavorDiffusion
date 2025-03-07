import pandas as pd
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from FlavorDiffusion import FlavorDiffusion

import pandas as pd
from tqdm import tqdm

from utils.plotter import plot_embedding
from utils.eval import evaluate, graph_reader

from pprint import pprint

def get_emb_size(input_nodes):
    df_nodes = pd.read_csv(input_nodes)
    
    st = set()
    for _, row in tqdm(df_nodes.iterrows(), total=len(df_nodes), desc="Processing Nodes"):
        node_id, name, _id, node_type, is_hub = row.values.tolist()
        st.add(node_id)
    
    return max(st) + 1 if st else 0

def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  
  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--ckpt_path', type=str, default=None)
  
  parser.add_argument('--input_nodes', type=str, default="/workspace/FlavorGraph/input/nodes_191120.csv")
  parser.add_argument('--input_edges', type=str, default="/workspace/FlavorGraph/input/edges_191120.csv")
  
  args = parser.parse_args()
  return args

def main(args):
  pl.seed_everything(42)
  model_class = FlavorDiffusion
  emb_size = get_emb_size(args.input_nodes)
  print("emb_size:", emb_size)  
  
  checkpoint = torch.load(args.ckpt_path, map_location="cpu")
  state_dict = checkpoint["state_dict"]
  embeddings = state_dict["model.node_embed.weight"]
  print(type(embeddings), embeddings.device, embeddings.shape)
  
  graph = graph_reader(args.input_nodes, args.input_edges)
  
  embed_dict = dict()
  embedding = embeddings.numpy()
  
  for node in graph.nodes():
    w = int(node)
    try:
      embed_dict[w] = embedding[w]
    except:
      print("something is wrong with", w)    
        
  plot_embedding(args, graph, vectors = embed_dict)
  # evaluate(args, graph, vectors = embed_dict)

if __name__ == '__main__':
  args = arg_parser()
  main(args)
