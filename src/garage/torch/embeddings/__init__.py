"""PyTorch embedding modules for meta-learning algorithms."""

from garage.torch.embeddings.mlp_encoder import MLPEncoder
from garage.torch.embeddings.oracle_encoder import OracleEncoder
__all__ = ['MLPEncoder', 'OracleEncoder']
