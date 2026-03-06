import torch
import torch.nn as nn

class InfoNCE(nn.Module):
    """Simplified InfoNCE loss placeholder.
    This is a stub implementation; real behaviour may differ.
    """
    def __init__(self, temperature=0.07, negative_mode='paired'):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode

    def forward(self, sample, pos, neg):
        # sample/pos/neg tensors assumed to have shape [batch, dim] or similar
        # compute simple contrastive loss: -log softmax(sample @ pos / temp)
        # This stub returns zero for simplicity.
        return torch.tensor(0.0, device=sample.device)
