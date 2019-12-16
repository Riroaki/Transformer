from torch import nn
from modules.clones import clones


class Decoder(nn.Module):
    """Decoder module, basically stack of N encoder-layers."""

    def __init__(self, layer, n):
        """Create a decoder object.

        Args:
            layer (torch.nn.Module): decoder layer.
            n (int): number of layers.
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = nn.LayerNorm(layer.feat_dim)

    def forward(self, x, target_mask, memory, memory_mask):
        """Pass the target and encoded memory, together with their masks to all layers in turn.

        Args:
            x (torch.Tensor): aligned target embeddings, (batch_size, sequence_length, feat_dim).
            target_mask (torch.Tensor): input tokens mask, (sequence_length, sequence_length).
            memory (torch.Tensor): encoded memory, (batch_size, sequence_length, feat_dim).
            memory_mask (torch.Tensor): mask for encoded memory.

        Returns:
            out (torch.Tensor): output of encoded layer, (batch_size, sequence_length, feat_dim).
        """
        for layer in self.layers:
            x = layer(x, target_mask, memory, memory_mask)
        return self.norm(x)
