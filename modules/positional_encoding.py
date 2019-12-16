import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding implementation."""

    def __init__(self, emb_dim, dropout_rate, max_len=5000):
        """Create a positional encoding layer object.

        Args:
            emb_dim (int): dimension of embedding.
            dropout_rate (float): rate of dropout.
            max_len (int): maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) *
                             -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Freeze the encoding vector.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Get the positional encoding of a given batch of sequences.

        Args:
            x (torch.Tensor): original embedding of sentences, (batch_size, sequence_length, emb_dim).

        Returns:
            out (torch.Tensor): embedding with positional encoding, (batch_size, sequence_length, emb_dim).
        """
        x = x + torch.tensor(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
