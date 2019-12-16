import torch
from torch import nn
from modules.clones import clones
from modules.residual import ResidualLayer


class EncoderLayer(nn.Module):
    """Encoder layer implementation."""

    def __init__(self, feat_dim, self_attn, feed_forward, dropout_rate):
        """Create a encoder layer object.

        Args:
            feat_dim (int): dimension of input layer, which equals to d_model.
            self_attn (torch.nn.Module): self attention layer.
            feed_forward (torch.nn.Module): point-wise feed forward layer.
            dropout_rate (float): rate of dropout.
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forwward = feed_forward
        self.residuals = clones(ResidualLayer(feat_dim, dropout_rate), 2)
        self.feat_dim = feat_dim

    def forward(self, x, mask):
        """Forward one step of encoder.

        Args:
            x (torch.Tensor): input tensor, (..., feat_dim).
            mask (torch.Tensor): input mask, (sequence_length, sequence_length).

        Returns:
            out (torch.Tensor): output of layer, (..., feat_dim).
        """
        x = self.residuals[0](x, lambda i: self.self_attn(i, i, i, mask))
        return self.residuals[1](x, self.feed_forwward)
