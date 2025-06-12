import torch.nn as nn
from functools import partial
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom

def subnet_fc(in_ch, out_ch, hid_ch):
    return nn.Sequential(
        nn.Linear(in_ch, hid_ch),
        nn.ReLU(),
        nn.Linear(hid_ch, out_ch)
    )

def build_inn(config: dict):
    """
    Build a FrEIA invertible neural network for rig/anchor mapping.
    Args:
        config: Configuration dictionary. Must contain:
            - NUM_SEGMENTS: int
            - INN_DEPTH: int
            - INN_WIDTH: int
    Returns:
        INN model (rig <-> anchor)
    """
    num_segments = config["NUM_SEGMENTS"]
    inn_depth = int(config["INN_DEPTH"])
    hid_ch = int(config["INN_WIDTH"])

    dim = (num_segments + 1) * 4
    subnet = partial(subnet_fc, hid_ch=hid_ch)

    nodes = [InputNode(dim, name='input')]
    for _ in range(inn_depth):
        nodes.append(Node(nodes[-1], RNVPCouplingBlock, {'subnet_constructor': subnet}))
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
