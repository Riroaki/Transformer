from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    """Generator implementation."""

    def __init__(self, feat_dim, vocab_size):
        """Create a generator object.

        Args:
            feat_dim (int): dimension of input vector.
            vocab_size (int): dimension of output logits, which represent each word's probability.
        """
        super(Generator, self).__init__()
        self.w = nn.Linear(feat_dim, vocab_size)

    def forward(self, x):
        """Compute logits for words.

        Args:
            x (torch.Tensor): output from decoder, (..., feat_dim).

        Returns:
            logits (torch.Tensor): logits of each word, (..., vocab_size)
        """
        return F.softmax(self.w(x), dim=-1)
