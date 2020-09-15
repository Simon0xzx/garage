"""PyTorch Policies."""
from garage.torch.policies.context_conditioned_policy import (
    ContextConditionedPolicy)
from garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.torch.policies.policy import Policy
from garage.torch.policies.tanh_gaussian_mlp_policy import (
    TanhGaussianMLPPolicy)
from garage.torch.policies.goal_conditioned_policy import GoalConditionedPolicy
from garage.torch.policies.tanh_gaussian_context_emphasized_policy import TanhGaussianContextEmphasizedPolicy
from garage.torch.policies.curl_policy import CurlPolicy
from garage.torch.policies.curl_shaped_policy import CurlShapedPolicy

__all__ = [
    'DeterministicMLPPolicy',
    'GaussianMLPPolicy',
    'Policy',
    'TanhGaussianMLPPolicy',
    'ContextConditionedPolicy',
    'GoalConditionedPolicy',
    'TanhGaussianContextEmphasizedPolicy',
    'CurlPolicy',
    'CurlShapedPolicy'
]
