import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

def run_blocks(blocks, x, use_checkpoint=False):
    for block in blocks:
        if use_checkpoint:
            x = checkpoint(block, x, use_reentrant=False)
        else:
            x = block(x)
    return x