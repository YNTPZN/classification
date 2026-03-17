import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al.).
    z: (B, D) normalized embeddings
    y: (B,) int labels
    """
    if z.dim() != 2:
        raise ValueError("z must be (B, D)")
    if y.dim() != 1:
        y = y.reshape(-1)

    device = z.device
    B = z.size(0)
    sim = (z @ z.T) / temperature  # (B,B)

    # Mask out self-similarity
    logits_mask = torch.ones((B, B), device=device, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)

    y = y.contiguous()
    pos_mask = (y[:, None] == y[None, :]) & logits_mask  # positives excluding self

    # For numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Mean over positives per anchor
    pos_count = pos_mask.sum(dim=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / (pos_count + 1e-12)

    # Only anchors that have at least one positive
    valid = pos_count > 0
    if valid.any():
        return loss[valid].mean()
    return torch.zeros((), device=device)

