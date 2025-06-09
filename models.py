import torch
import torch.nn as nn
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import RNVPCouplingBlock, PermuteRandom



def subnet_fc(in_ch, out_ch):
    return nn.Sequential(
        nn.Linear(in_ch, 256),
        nn.ReLU(),
        nn.Linear(256, out_ch)
    )


def build_inn(num_segments):
    dim = (num_segments + 1) * 4  # FIXED input/output dim for invertible net
    nodes = [InputNode(dim, name='input')]

    for _ in range(6):
        nodes.append(Node(nodes[-1], RNVPCouplingBlock, {'subnet_constructor': subnet_fc}))
        nodes.append(Node(nodes[-1], PermuteRandom, {'seed': 0}))

    nodes.append(OutputNode(nodes[-1], name='output'))
    return ReversibleGraphNet(nodes, verbose=False)



class INNWrapper(nn.Module):
    def __init__(self, inn):
        super().__init__()
        self.inn = inn

    def forward(self, x):
        return self.inn(x)  # rig → anchor

    def inverse(self, y):
        return self.inn(y, rev=True)  # anchor → rig


class SimpleMLP(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        h_dim = 1024
        self.fc = nn.Sequential(
            nn.Linear(input_dims, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, output_dims)
        )

    def forward(self, x):
        return self.fc(x)
