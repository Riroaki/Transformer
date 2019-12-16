import math
from torch import nn


class Embeddings(nn.Module):
    """Embedding implementation."""

    def __init__(self, emb_dim, vocab_size):
        """Create an embedding object.

        Args:
            emb_dim (int): dimension of embedding vector.
            vocab_size (int): size of vocabulary.
        """
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size,
                                embedding_dim=emb_dim)
        self.scale = math.sqrt(emb_dim)

    def forward(self, ids):
        """Perform scaled embedding.

        Args:
            ids (torch.Tensor): input token ids, (batch_size, sequence_length).

        Returns:
            embeddings (torch.Tensor): embedding vectors, (batch_size, sequence_length, embedding_size)
        """
        return self.emb(ids) * self.scale
