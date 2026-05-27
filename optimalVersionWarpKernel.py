import torch
import torch.nn as nn
import torch.nn.functional as F
import geotorch as gt
from geotorch.manifolds import Stiefel


class WarpKernel(nn.Module):
    """Low-rank Riemannian metric that shapes the hidden-state landscape.
    
    Defines G = I + A^T A, a positive-definite metric tensor.
    Use with centered/normalized hidden states for meaningful depth signals.
    """
    def __init__(self, dim: int, rank: int = 8):
        super().__init__()
        if rank > dim:
            raise ValueError(f"rank ({rank}) must be <= dim ({dim})")
        # Small init keeps G ≈ I (perturbative regime)
        self.A = nn.Parameter(torch.randn(rank, dim) * 0.01)

    def metric(self) -> torch.Tensor:
        """Global positive-definite metric G = I + A^T A."""
        return torch.eye(self.A.size(1), device=self.A.device) + self.A.T @ self.A

    def depth(self, h: torch.Tensor) -> torch.Tensor:
        """Scalar potential: smaller = deeper in reward valley.
        
        Computes ‖h‖_G² = h^T G h. Assumes h is centered near zero.
        """
        G = self.metric()
        return torch.einsum("bd,dd,bd->b", h, G, h)


class GeoLinear(nn.Module):
    """Riemannian low-rank linear layer on the Stiefel manifold.
    
    Factorizes W ≈ U · diag(exp(log_s)) · V^T with U^T U = I, V^T V = I.
    Requires a Riemannian optimizer (e.g., geotorch.optim.RiemannianAdam).
    """
    def __init__(self, in_features: int, out_features: int, rank: int = None):
        super().__init__()
        rank = min(in_features, out_features) if rank is None else rank
        
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank ({rank}) must be <= min(in_features={in_features}, "
                f"out_features={out_features}) for Stiefel manifold constraints"
            )

        # Stiefel manifold parameters: U^T U = I, V^T V = I
        self.U = gt.ManifoldParameter(
            torch.randn(out_features, rank),
            manifold=Stiefel()
        )
        self.V = gt.ManifoldParameter(
            torch.randn(in_features, rank),
            manifold=Stiefel()
        )
        # Euclidean parameter for singular values (log-parametrized for positivity)
        self.log_s = nn.Parameter(torch.zeros(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Efficient low-rank forward: O(B·d·r + B·r·out)."""
        h = F.linear(x, self.V)                          # (B, rank)
        return F.linear(h, (self.U * self.log_s.exp()).T) # (B, out)

    def check_orthogonality(self) -> dict:
        """Numerical verification of Stiefel constraints."""
        device = self.U.device
        return {
            "U^T U": torch.norm(
                self.U.T @ self.U - torch.eye(self.U.size(1), device=device)
            ).item(),
            "V^T V": torch.norm(
                self.V.T @ self.V - torch.eye(self.V.size(1), device=device)
            ).item(),
        }

    def __repr__(self):
        return (f"GeoLinear(in={self.V.size(0)}, out={self.U.size(0)}, "
                f"rank={self.log_s.size(0)})")
