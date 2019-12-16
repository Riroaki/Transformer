from copy import deepcopy
from torch import nn
from modules import *


class Transformer(nn.Module):
    """Transformer full model implementation."""

    def __init__(self, src_vocab, model_dim, max_len, encoder_layers,
                 decoder_layers, tgt_vocab, ff_hidden, attn_head,
                 num_class, padding_idx, smoothing=0.1, dropout_rate=0.1):
        """Create a transformer model object.

        Args:
            src_vocab (int): size of source vocabulary.
            model_dim (int): dimension of input / output / embedding of each layer in the model.
            max_len (int): maximum sequence length.
            encoder_layers (int): number of encoder layers.
            decoder_layers (int): number of decoder layers.
            tgt_vocab (int): size of target vocabulary.
            ff_hidden (int): number of hidden units in point-wise feed forward layer.
            attn_head (int): number of heads in attention layers.
            num_class (int): number of classes of labels.
            padding_idx (int): ignore class index.
            smoothing (float): smoothing factor between 0 and 1.
            dropout_rate (float): rate of dropout in each layer.
        """
        super(Transformer, self).__init__()
        # Make prototype of common modules.
        pe = PositionalEncoding(model_dim, dropout_rate, max_len)
        attn = MultiHeadedAttention(attn_head, model_dim)
        ff = PositionwiseFeedForward(model_dim, ff_hidden, dropout_rate)
        # Register modules.
        self.src_emb = nn.Sequential(Embeddings(model_dim, src_vocab),
                                     deepcopy(pe))
        self.tgt_emb = nn.Sequential(Embeddings(model_dim, tgt_vocab),
                                     deepcopy(pe))
        self.encoder = Encoder(
            EncoderLayer(model_dim, deepcopy(attn), deepcopy(ff), dropout_rate),
            encoder_layers)
        self.decoder = Decoder(
            DecoderLayer(model_dim, deepcopy(attn), deepcopy(attn),
                         deepcopy(ff), dropout_rate), decoder_layers)
        self.generator = Generator(model_dim, num_class)
        self.label_smooth = LabelSmoothing(num_class, padding_idx, smoothing)

    def init_weights(self):
        """Initialize parameters in model using Xavier method."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        """Compute forward propagation.

        Args:
            src (torch.Tensor): batch of padded word ids, (batch_size, sequence_length).
            tgt (torch.Tensor): batch of padded output ids, (batch_size, sequence_length).

        Returns:
            loss (torch.Tensor): scalar loss value, (1,).
        """
        src_mask = subsequent_mask(src.size(1))
        tgt_mask = subsequent_mask(tgt.size(1))
        memory = self.encoder(self.src_emb(src), src_mask)
        logits = self.generator(
            self.decoder(self.tgt_emb(tgt), tgt_mask, memory, src_mask))
        return self.label_smooth(logits, tgt)
