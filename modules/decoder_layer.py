from torch import nn
from modules.residual import ResidualLayer
from modules.clones import clones


class DecoderLayer(nn.Module):
    """Decoder layer implementation."""

    def __init__(self, feat_dim, self_attn, src_attn, feed_forward,
                 dropot_rate):
        """Create a decoder layer object.

        Args:
            feat_dim (int): dimension of input / outpt.
            self_attn (torch.nn.Module): self attention layer.
            src_attn (torch.nn.Module): source attention layer.
            feed_forward (torch.nn.Module): position-wise feed forward layer.
            dropot_rate (float): rate of dropout.
        """
        super(DecoderLayer, self).__init__()
        self.feat_dim = feat_dim
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.residuals = clones(ResidualLayer(feat_dim, dropot_rate), 3)

    def forward(self, x, target_mask, memory, memory_mask):
        """Forward one step of decoder.

        Args:
            x (torch.Tensor): aligned target embeddings, (batch_size, sequence_length, feat_dim).
            target_mask (torch.Tensor): input tokens mask, (sequence_length, sequence_length).
            memory (torch.Tensor): encoded memory, (batch_size, sequence_length, feat_dim).
            memory_mask (torch.Tensor): mask for encoded memory.

        Returns:
            out (torch.Tensor): output of encoded layer, (batch_size, sequence_length, feat_dim).
        """
        x = self.residuals[0](x, lambda i: self.self_attn(i, i, i, target_mask))
        x = self.residuals[1](x, lambda i: self.src_attn(i, memory, memory,
                                                         memory_mask))
        return self.residuals[2](x, self.feed_forward)
