import torch
import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom

def subnet_fc(in_ch, out_ch):
    return nn.Sequential(
        nn.Linear(in_ch, 1024),
        nn.ReLU(),
        nn.Linear(1024, out_ch)
    )

def build_inn(num_segments: int):
    """
    Build a FrEIA invertible neural network for rig/anchor mapping.
    Args:
        num_segments: Number of rig segments (N)
    Returns:
        INN model (rig <-> anchor)
    """
    dim = (num_segments + 1) * 4
    nodes = [InputNode(dim, name='input')]
    for _ in range(12):
        nodes.append(Node(nodes[-1], RNVPCouplingBlock, {'subnet_constructor': subnet_fc}))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': 0}))
    nodes.append(OutputNode(nodes[-1], name='output'))
    inn = ReversibleGraphNet(nodes, verbose=False)
    return INNWrapper(inn)

class INNWrapper(nn.Module):
    def __init__(self, inn):
        super().__init__()
        self.inn = inn

    def forward(self, x):
        return self.inn(x)

    def inverse(self, y):
        return self.inn(y, rev=True)

    def to(self, *args, **kwargs):
        self.inn = self.inn.to(*args, **kwargs)
        return super().to(*args, **kwargs)
