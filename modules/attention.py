import math
from torch import nn


def attention(query, key, value, mask=None, dropout=None):
    """Scaled Dot Prodction Attention implmentation.

    Args:
        query (torch.Tensor): batch of query matrix, (..., sequence_length, d_k).
        key (torch.Tensor): batch of key matrix, (..., sequence_length, d_k).
        value (torch.Tensor): batch of value matrix, (..., sequence_length, d_k).
        mask (torch.ByteTensor, optional): mask of attention, (sequence_length, sequence_length).
        dropout (torch.nn.Dropout, optional): dropout for attention map.

    Returns:
        attention_map (torch.Tensor): attentive map, (..., sequence_length, sequence_length).
        attended_output (torch.Tensor): output of attention, (..., sequence_length, d_k).
    """
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    # Mask out 0 slots.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_map = scores.softmax(dim=-1)
    # Dropout on attention map.
    if dropout is not None:
        attn_map = dropout(attn_map)
    return attn_map, attn_map.matmul(value)


class MultiHeadedAttention(nn.Module):
    """Multi-headed Attention implementation."""

    def __init__(self, num_head, feat_dim, dropout_rate=0.1):
        """Create a multi-headed attention layer.

        Args:
            num_head (int): number of heads.
            feat_dim (int): number of input features.
            dropout_rate (float): rate of dropout.
        """
        super(MultiHeadedAttention, self).__init__()
        assert feat_dim % num_head == 0
        # Assume value_dim == key_dim
        self.d_k = feat_dim // num_head
        self.h = num_head
        self.w_q = nn.Linear(feat_dim, feat_dim)
        self.w_k = nn.Linear(feat_dim, feat_dim)
        self.w_v = nn.Linear(feat_dim, feat_dim)
        self.w_out = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout_rate)
        # Store attention map
        self.attn_map = None

    def forward(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention'.

        Args:
            query (torch.Tensor): batch of query matrix, (batch_size, sequence_length, query_dim).
            key (torch.Tensor): batch of key matrix, (batch_size, sequence_length, key_dim).
            value (torch.Tensor): batch of value matrix, (batch_size, sequence_length, value_dim).
            mask (torch.ByteTensor, optional): Mask of attention, (sequence_length, sequence_length).

        Returns:
            out (torch.Tensor): attended and transformed value, (batch_size, sequence_length, feat_dim).
        """
        num_batch = query.size(0)
        q = self.w_q(query).view(num_batch, -1, self.h, self.d_k)
        k = self.w_k(key).view(num_batch, -1, self.h, self.d_k)
        v = self.w_v(value).view(num_batch, -1, self.h, self.d_k)
        # After transpose: (batch_size, num_head, sequence_length, d_k).
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        self.attn_map, x = attention(q, k, v, mask, self.dropout)
        # Transpose again to recover dimensions: (batch_size, sequence_length, feat_dim)
        x = x.transpose(1, 2).contiguous().view(num_batch, -1,
                                                self.h * self.d_k)
        return self.w_out(x)
