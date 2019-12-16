from torch import nn
from modules.clones import clones


class Encoder(nn.Module):
    """Encoder module, basically stack of N encoder-layers."""

    def __init__(self, layer, n):
        """Create an encoder object.

        Args:
            layer (torch.nn.Module): encoder layer.
            n (int): number of layers.
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.feat_dim)

    def forward(self, x, mask):
        """Pass the input and mask to all layers in turn.

        Args:
            x (torch.Tensor): input tensor, (..., feat_dim).
            mask (torch.Tensor): mask, (sequence_length, sequence_length).

        Returns:
            out (torch.Tensor): output of encoder, (..., feat_dim).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
