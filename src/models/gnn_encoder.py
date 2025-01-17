import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
import torch.utils.checkpoint as activation_checkpoint
import torch.nn.init as init

class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, mode="residual"):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
    Returns:
        Updated node and edge features
    """
    batch_size, num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates)  # B x V x H

    # Normalize node features
    h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
    ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h

    e = self.norm_e(
        e.view(batch_size * num_nodes * num_nodes, hidden_dim)
    ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, graph, gates, mode=None):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if (mode or self.aggregation) == "mean":
      return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
    elif (mode or self.aggregation) == "max":
      return torch.max(Vh, dim=2)[0]
    else:
      return torch.sum(Vh, dim=2)


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               use_activation_checkpoint=False, emb_size = -1,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    
    self.node_embed = nn.Embedding(emb_size, hidden_dim)
    init.xavier_uniform_(self.node_embed.weight)
    
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
      
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, x, graph, timesteps):
    """
    Args:
        x: Input node index (B x V)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    x = self.node_embed(x)
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    graph = torch.ones_like(graph).long()

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x, e = layer(x, e, graph, mode="direct")
      e = e + time_layer(time_emb)[:, None, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
    e = self.out(e.permute((0, 3, 1, 2)))
    return e

  
  def forward(self, x, timesteps, graph=None):
    return self.dense_forward(x, graph, timesteps)
      
