import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch as gt
from geotorch.manifolds import Stiefel

class WarpKernel(nn.Module):
    """Low-rank Riemannian metric that shapes the hidden-state landscape."""
    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def metric(self) -> torch.Tensor:
        """Global positive-definite metric G = I + A^T A."""
        return torch.eye(self.A.size(1), device=self.A.device) + self.A.T @ self.A

    def depth(self, h: torch.Tensor) -> torch.Tensor:
        """Scalar potential: smaller = deeper in reward valley."""
        G = self.metric()
        return torch.einsum("bd,dd,bd->b", h, G, h)


class GeoLinear(nn.Module):
    """Riemannian low-rank linear layer on the Stiefel manifold."""
    def __init__(self, in_features: int, out_features: int, rank: int = None):
        super().__init__()
        rank = min(in_features, out_features) if rank is None else rank

        # Stiefel manifold parameters: U^T U = I, V^T V = I
        self.U = gt.ManifoldParameter(
            torch.randn(out_features, rank),
            manifold=Stiefel()
        )
        self.V = gt.ManifoldParameter(
            torch.randn(in_features, rank),
            manifold=Stiefel()
        )
        # Euclidean parameter for singular values
        self.log_s = nn.Parameter(torch.zeros(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Efficient low-rank forward: O(B·d·r + B·r·d)
        h = F.linear(x, self.V)                          # (B, rank)
        return F.linear(h, (self.U * self.log_s.exp()).T) # (B, out)

    def check_orthogonality(self) -> dict:
        """Numerical verification of Stiefel constraints."""
        return {
            "U^T U": torch.norm(self.U.T @ self.U - torch.eye(self.U.size(1), device=self.U.device)).item(),
            "V^T V": torch.norm(self.V.T @ self.V - torch.eye(self.V.size(1), device=self.V.device)).item()
        }
