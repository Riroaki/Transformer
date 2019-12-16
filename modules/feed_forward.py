from torch import nn
from torch.nn import functional as F


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward layer implementation."""

    def __init__(self, feat_dim, hidden_dim, dropout_rate=0.1):
        """Create a position-wise feed forward layer object.

        Args:
            feat_dim (int): dimension of input / out.
            hidden_dim (int): number of hidden units.
            dropout_rate (float): rate of dropout.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(feat_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, feat_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Perform one step.

        Args:
            x (torch.Tensor): input tensor, (..., feat_dim).

        Returns:
            out (torch.Tensor): output tensor, (..., feat_dim).
        """
        return self.w2(self.dropout(F.relu(self.w1(x))))
