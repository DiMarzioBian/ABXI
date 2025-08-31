import torch


def get_absolute_pos_idx(mask: torch.Tensor,
                         ) -> torch.Tensor:
    """
    Generate position index, default ignoring padding and masking index 0.
    Input mask is non-padded mask
    """
    mask = mask.long().squeeze(-1)
    return mask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]) * mask
