"""PyTorch embedding modules for meta-learning algorithms."""

from garage.torch.embeddings.mlp_encoder import MLPEncoder
from garage.torch.embeddings.identity_encoder import IdentityEncoder
from garage.torch.embeddings.contrastive_encoder import ContrastiveEncoder
__all__ = ['MLPEncoder', 'IdentityEncoder', 'ContrastiveEncoder']
