"""The handler for training and evaluation."""

import os
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info

from FlavorDiffusion import FlavorDiffusion

import pandas as pd
from tqdm import tqdm

def get_emb_size(input_nodes):
    df_nodes = pd.read_csv(input_nodes)
    
    st = set()
    for _, row in tqdm(df_nodes.iterrows(), total=len(df_nodes), desc="Processing Nodes"):
        node_id, name, _id, node_type, is_hub = row.values.tolist()
        st.add(node_id)
    
    return max(st) + 1 if st else 0

def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a TSP dataset.')
  parser.add_argument('--storage_path', type=str, required=True)
  parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_train_concorde.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split (used for MIS).")
  parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde.txt')
  parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde.txt')

  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--num_epochs', type=int, default=50)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--weight_decay', type=float, default=0.0)
  parser.add_argument('--lr_scheduler', type=str, default='constant')

  parser.add_argument('--num_workers', type=int, default=16)
  parser.add_argument('--fp16', action='store_true')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='gaussian') # gaussian
  parser.add_argument('--diffusion_schedule', type=str, default='cosine') # gaussian
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1000)
  parser.add_argument('--inference_schedule', type=str, default='cosine') # gaussian
  parser.add_argument('--inference_trick', type=str, default=None)
  
  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--aggregation', type=str, default='sum')

  parser.add_argument('--project_name', type=str, default='flavor_diffusion')
  parser.add_argument('--logger_name', type=str, default=None)
  parser.add_argument('--ckpt_path', type=str, default=None)
  parser.add_argument('--resume_weight_only', action='store_true')

  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_valid_only', action='store_true')
  
  parser.add_argument('--input_nodes', type=str, default="/workspace/FlavorGraph/input/nodes_191120.csv")
  
  parser.add_argument('--CSP_train', default=False, action="store_true")
  
  args = parser.parse_args()
  return args

def main(args):
  pl.seed_everything(42)

  epochs = args.num_epochs
  model_class = FlavorDiffusion
  emb_size = get_emb_size(args.input_nodes)
  print("emb_size:", emb_size)
  model = model_class(emb_size, param_args=args)

  checkpoint_callback = ModelCheckpoint(
      monitor='val_mse_loss', 
      mode='min',
      save_top_k=3, 
      save_last=True,
      every_n_epochs=1,
      filename = f'FlavorDiffusion-' + "{epoch:02d}-{val_mse_loss:.4f}",
  )
  
  lr_callback = LearningRateMonitor(logging_interval='step')

  trainer = Trainer(
      accelerator="cuda",
      # devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      devices=[0],
      max_epochs=epochs,
      callbacks=[checkpoint_callback, lr_callback],
      check_val_every_n_epoch=1,
      strategy=DDPStrategy(static_graph=True),
      precision="16-mixed" if args.fp16 else 32,
  )

  rank_zero_info(
      f"{'-' * 100}\n"
      f"{str(model.model)}\n"
      f"{'-' * 100}\n"
  )

  ckpt_path = args.ckpt_path

  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=ckpt_path)

    if args.do_test:
      trainer.test(ckpt_path=checkpoint_callback.best_model_path)

  elif args.do_test:
    trainer.validate(model, ckpt_path=ckpt_path)
    if not args.do_valid_only:
      trainer.test(model, ckpt_path=ckpt_path)

if __name__ == '__main__':
  args = arg_parser()
  main(args)
