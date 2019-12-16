import torch
from torch import nn


class LabelSmoothing(nn.Module):
    """Label smoothing implementation."""

    def __init__(self, num_class, padding_idx, smoothing=0.1):
        """Create a label smoothing object.
        Penalizes model when it gets too confident about prediction to avoid over-fitting.

        Args:
            num_class (int): number of classes.
            padding_idx (int): ignore class index.
            smoothing (float): smoothing factor between 0 and 1.
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_class = num_class
        self.true_dist = None

    def forward(self, x, target):
        """Compute loss between x and target.

        Args:
            x (torch.Tensor): output of decoder, (batch_size, sequence_length, num_class).
            target (torch.Tensor): target signal masked with self.padding_id, (batch, seqlen).

        Returns:
            loss (torch.Tensor): scalar loss value, (1,).
        """
        assert x.size(1) == self.num_class
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.num_class - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.tensor(true_dist, requires_grad=False))
