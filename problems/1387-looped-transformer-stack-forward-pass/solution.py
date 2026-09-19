import torch
import torch.nn as nn


class LoopedStack(nn.Module):
    def __init__(self, blocks, n_loops):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.n_loops = n_loops

    def forward(self, x, n_loops=None):
        loops = self.n_loops if n_loops is None else n_loops

        for _ in range(loops):
            for block in self.blocks:
                x = block(x)

        return x

    def block_applications(self, n_loops=None):
        loops = self.n_loops if n_loops is None else n_loops
        return len(self.blocks) * loops

    def shared_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def unrolled_parameter_count(self, n_loops=None):
        loops = self.n_loops if n_loops is None else n_loops
        return self.shared_parameter_count() * loops