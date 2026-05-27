def train_geometric_step(model: GeoLinear, kernel: WarpKernel, 
                         x: torch.Tensor, reward: torch.Tensor,
                         optimizer: gt.optim.RiemannianAdam):
    """Single Riemannian update with reward-shaped depth loss."""
    optimizer.zero_grad()
    
    # Forward through geometric layer
    h = model(x)
    
    # WarpKernel shapes the potential landscape
    depth = kernel.depth(h)
    
    # Reward scales the valley depth: +reward deepens, -reward shallows
    loss = -(reward * depth).mean()
    
    # Riemannian backward pass
    loss.backward()
    
    # Geodesic step on Stiefel manifolds + Euclidean step on log_s & A
    optimizer.step()
    
    return loss.item()

# --- Setup ---
dim = 768
rank = 8
model = GeoLinear(dim, dim, rank=rank)
kernel = WarpKernel(dim, rank=rank)

# RiemannianAdam automatically routes manifold vs Euclidean parameters
optimizer = gt.optim.RiemannianAdam(
    list(model.parameters()) + list(kernel.parameters()),
    lr=3e-4,
    weight_decay=1e-2
)

# --- Dummy training loop ---
for step in range(50):
    x_batch = torch.randn(32, dim)
    reward = torch.randn(32) * 2  # simulate +1/-1 reward signal
    
    loss = train_geometric_step(model, kernel, x_batch, reward, optimizer)
    
    if step % 10 == 0:
        ortho = model.check_orthogonality()
        print(f"Step {step:3d} | Loss: {loss:.4f} | U^TU-I: {ortho['U^T U']:.2e} | V^TV-I: {ortho['V^T V']:.2e}")
