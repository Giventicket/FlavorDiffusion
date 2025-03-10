"""Lightning module for training the DIFUSCO TSP model."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from flavor_datasets.graph_dataset import GraphDataset
from utils.diffusion_schedulers import InferenceSchedule

from models.gnn_encoder import GNNEncoder
from utils.lr_schedulers import get_schedule_fn
from utils.diffusion_schedulers import GaussianDiffusion

import pickle
import numpy as np
import csv
from tqdm import tqdm


def load_augmentive_features(emb_size):
    PICKLE_PATH = "/workspace/FlavorGraph/input/node2fp_revised_1120.pickle"
    print("Loading Chemical Vectors from ", PICKLE_PATH)
    with open(PICKLE_PATH, "rb") as handle:
        binary_dict = pickle.load(handle)
    print("Number of Binary Vectors Available: ", len(binary_dict.keys()))
    print("Number of Nodes in Graph: ", emb_size)

    augmentive_matrix = []
    binary_mask = []

    for idx in range(emb_size):
        try:
            binary_vector = list(binary_dict[idx])
            binary_mask.append([1])
        except:
            binary_vector = [0.0 for _ in range(881)]
            binary_mask.append([0])
        augmentive_matrix.append(binary_vector)

    binary_mask = np.array(binary_mask).astype(float)
    augmentive_matrix = np.array(augmentive_matrix).astype(float)
    vector_length = augmentive_matrix.shape[1]

    return torch.tensor(augmentive_matrix, requires_grad=False).float().to("cuda"), vector_length, torch.tensor(binary_mask, requires_grad=False).float().to("cuda")


class FlavorDiffusion(pl.LightningModule):
  def __init__(self, emb_size, param_args=None):
    super(FlavorDiffusion, self).__init__()
    
    self.args = param_args
    self.diffusion_type = self.args.diffusion_type
    self.diffusion_schedule = self.args.diffusion_schedule
    self.diffusion_steps = self.args.diffusion_steps

    if self.diffusion_type == 'gaussian':
      out_channels = 1
      self.diffusion = GaussianDiffusion(
          T=self.diffusion_steps, schedule=self.diffusion_schedule)
    else:
      raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

    self.model = GNNEncoder(
        n_layers=self.args.n_layers,
        hidden_dim=self.args.hidden_dim,
        out_channels=out_channels,
        aggregation=self.args.aggregation,
        use_activation_checkpoint=self.args.use_activation_checkpoint,
        emb_size=emb_size
    )
    self.num_training_steps_cached = None
    
    self.save_hyperparameters(self.args)  # save config file with pytorch lightening

    self.train_dataset = GraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.training_split),
    )

    self.test_dataset = GraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
    )

    self.validation_dataset = GraphDataset(
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
    )
    
    self.CSP_train = self.args.CSP_train
    
    if self.CSP_train:
      self.aug_embeddings, self.aug_dimension, self.binary_masks = load_augmentive_features(emb_size)
      self.encoder_aug = nn.Linear(self.args.hidden_dim, self.aug_dimension)
      self.criterion_aug = nn.BCEWithLogitsLoss()
    
    """
    self.recon_edge = {}
    for i in range(emb_size):
        for j in range(emb_size):
            self.recon_edge.setdefault(i, {}).setdefault(j, [])
    """
      
  def train_dataloader(self):
    batch_size = self.args.batch_size
    train_dataloader = GraphDataLoader(
        self.train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=self.args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

  def val_dataloader(self):
    batch_size = 1
    print("Validation dataset size:", len(self.validation_dataset))
    val_dataloader = GraphDataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader

  def gaussian_posterior(self, target_t, t, pred, xt):
    """Sample (or deterministically denoise) from the Gaussian posterior for a given time step.
       See https://arxiv.org/pdf/2010.02502.pdf for details.
    """
    
    diffusion = self.diffusion
    if target_t is None:
      target_t = t - 1
    else:
      target_t = torch.from_numpy(target_t).view(1)

    atbar = diffusion.alphabar[t]
    atbar_target = diffusion.alphabar[target_t]

    if self.args.inference_trick is None or t <= 1:
      # Use DDPM posterior
      at = diffusion.alpha[t]
      z = torch.randn_like(xt)
      atbar_prev = diffusion.alphabar[t - 1]
      beta_tilde = diffusion.beta[t - 1] * (1 - atbar_prev) / (1 - atbar)

      xt_target = (1 / np.sqrt(at)).item() * (xt - ((1 - at) / np.sqrt(1 - atbar)).item() * pred)
      xt_target = xt_target + np.sqrt(beta_tilde).item() * z
    elif self.args.inference_trick == 'ddim':
      xt_target = np.sqrt(atbar_target / atbar).item() * (xt - np.sqrt(1 - atbar).item() * pred)
      xt_target = xt_target + np.sqrt(1 - atbar_target).item() * pred
    else:
      raise ValueError('Unknown inference trick {}'.format(self.args.inference_trick))

    return xt_target
  
  def gaussian_training_step(self, batch, batch_idx):
    _, points, adj_matrix = batch
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    adj_matrix = adj_matrix.clamp(-1, 1)  # clamping

    xt, epsilon = self.diffusion.sample(adj_matrix, t)
    xt = xt.clamp(-1, 1) 
    
    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    
    epsilon_pred = self.model(
        points.to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device)
    )
    epsilon_pred = epsilon_pred.squeeze(1)
    
    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
    return loss
  
  def training_step(self, batch, batch_idx):
    self.model.train()
    
    if self.diffusion_type == 'gaussian':
      loss = self.gaussian_training_step(batch, batch_idx)
        
      if self.args.CSP_train:
        _, points, _ = batch
        
        binary_masks = self.binary_masks.to(points.device)
        aug_embeddings = self.aug_embeddings.to(points.device)
        
        points_mask = binary_masks[points].bool().squeeze(-1)
        extracted_points = points[points_mask]

        extracted_embeddings = self.model.node_embed(extracted_points)
        feature_aug = self.encoder_aug(extracted_embeddings)
        
        tgt_aug = aug_embeddings[extracted_points]
        loss_aug = self.criterion_aug(feature_aug, tgt_aug)

        return loss + loss_aug
        
      return loss

    else:
      raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

  def gaussian_denoise_step(self, points, xt, t, device, target_t=None):
    with torch.no_grad():
      
      t = torch.from_numpy(t).view(1)
    
      pred = self.model(
          points.to(device),
          t.float().to(device),
          xt.float().to(device)
      )
      
      pred = pred.permute((0, 2, 3, 1)).contiguous()
      
      pred = pred.squeeze(-1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      xt = xt.clamp(-1, 1)
      
      return xt

  def test_step(self, batch, batch_idx, split='test'):
      self.model.eval()
      device = batch[-1].device
      
      real_batch_idx, points, gt_adj = batch
      
      xt = torch.randn_like(gt_adj.float())
      
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(
          inference_schedule=self.args.inference_schedule,
          T=self.diffusion.T,
          inference_T=steps,
      )

      skip = self.diffusion.T // steps
      assert self.diffusion.T % steps == 0, f"self.diffusion.T ({self.diffusion.T}) must be divisible by steps ({steps})."
      
      diffusion_results = []

      # Diffusion iterations
      for i in range(steps):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1])  # tIdx => T - (i + 1) * skip, s
          t2 = np.array([t2])  # tIdx => T - (i) * skip, t
          
          if self.diffusion_type == 'gaussian':
              xt = self.gaussian_denoise_step(
                  points, xt, t1.astype(int), device, target_t=t2.astype(int)
              )
          else:
              raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

          diffusion_results.append({
              "step": i,
              "points": points.cpu().numpy(),
              "xt": xt.cpu().detach().numpy() * 0.5 + 0.5,  # 변환 후 저장
              "gt_adj": gt_adj.cpu().detach().numpy()
          })
      
      np.save(f"diffusion_results_{real_batch_idx.cpu().numpy()}.npy", diffusion_results, allow_pickle=True)
      
      if self.diffusion_type == 'gaussian':
          adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
          adj_mat = np.clip(adj_mat, 0, 1)
      else:
          raise ValueError(f"Unknown diffusion type {self.diffusion_type}")

      adj_pred = torch.tensor(adj_mat, dtype=torch.float32, device=device)
      gt_adj = torch.tensor(gt_adj, dtype=torch.float32, device=device)
      
      """
      for i in range(len(adj_pred[0])):
        for j in range(len(adj_pred[0][0])):
            u = points[0][i].item()
            v = points[0][j].item()
            self.recon_edge[u][v].append(adj_pred[0][i][j].item())
      """
      
      mse_loss = F.mse_loss(adj_pred, gt_adj, reduction='mean')

      # Logging
      metrics = {
          f"{split}_mse_loss": mse_loss.item(),
      }

      for k, v in metrics.items():
          self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)

      return metrics

  def validation_step(self, batch, batch_idx):
    self.model.eval()
    return self.test_step(batch, batch_idx, split='val')
  
  def on_validation_epoch_end(self):
    pass
    """
    mean_std_edge = {}
    for u in self.recon_edge:
        mean_std_edge[u] = {}
        for v in self.recon_edge[u]:
            values = self.recon_edge[u][v]
            if values:
                mean = np.mean(values)
                std = np.std(values)
            else:
                mean, std = 0, 0
            mean_std_edge[u][v] = {'mean': mean, 'std': std}

    with open("./edge_mean_std.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["u", "v", "mean", "std"])

        for u in tqdm(mean_std_edge):
            for v in mean_std_edge[u]:
                writer.writerow([u, v, mean_std_edge[u][v]['mean'], mean_std_edge[u][v]['std']])
    """
    
  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached
  
  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }
  