import torch


def subsequent_mask(sequence_length):
    """Return a mask matrix of sequence.

    Args:
        sequence_length (int): length of full sequence.

    Returns:
        mask (torch.Tensor): mask, (sequence_length, sequence_length).
    """
    return torch.tril(torch.full((sequence_length, sequence_length), True))
