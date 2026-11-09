import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class MambaDetector(nn.Module):
    """Packed batch (x (N,21) probabilities, cu (B+1,) offsets) -> log-probs -> Linear -> Mamba2 layers (seq_idx, so the
    state never crosses texts) -> last token's hidden state per text -> MLP -> n_classes logits. Needs CUDA."""

    def __init__(self, n_classes, d=64, n_layers=2):
        super().__init__()
        self.inp = nn.Linear(21, d)
        self.layers = nn.ModuleList(Mamba2(d_model=d, headdim=16) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layers))
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, x, cu):
        seq_idx = torch.repeat_interleave(torch.arange(len(cu) - 1, device=x.device, dtype=torch.int32), (cu[1:] - cu[:-1]).long(), output_size=len(x))[None]
        h = self.inp(torch.log(x.clamp_min(1e-9)) / 10)[None]  # (1,N,d): all texts as one sequence
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h), seq_idx=seq_idx)
        return self.head(h[0, cu[1:].long() - 1])  # last token of each text -> (B, n_classes)
