from torch import nn


class ResidualLayer(nn.Module):
    """A residual connection followed by a layer norm.
    The layer normalization is performed before calculations in layer.
    """

    def __init__(self, feat_dim, dropout_rate):
        """Create a residual layer object.

        Args:
            feat_dim (int): number of input features.
            dropout_rate (float): dropout rate.
        """
        super(ResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, layer):
        """Apply residual connection to any sublayer with the same size.

        Args:
            x (torch.Tensor): input tensor, (..., feat_dim).
            layer (torch.nn.Module): any layer.

        Returns:
            layer_out (torch.Tensor): output tensor, (...).
        """
        return x + self.dropout(layer(self.norm(x)))
