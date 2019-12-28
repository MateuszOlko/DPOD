import torch


def get_mask(tensor):
    """
    Args:
        tensor: prediction tensor of dim (batch_size, classes, H, W)

    Returns: mask of dim (batch_size, H, W)
    """
    return torch.argmax(tensor, dim=1)
