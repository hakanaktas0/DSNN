from typing import Optional

import torch
from torch_geometric.utils import  remove_self_loops, coalesce, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np


def get_phase(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                           normalization: Optional[str] = 'sym',
                           dtype: Optional[int] = None,
                           num_nodes: Optional[int] = None,
                           q: Optional[float] = 0.25,
                           return_lambda_max: bool = False):
    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes, "add")

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter(edge_weight_sym, row, dim=0, dim_size=num_nodes, reduce='sum')

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])
    return edge_weight_q

def get_edge_attr(edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None,
                           normalization: Optional[str] = 'sym',
                           dtype: Optional[int] = None,
                           num_nodes: Optional[int] = None,
                           return_lambda_max: bool = False):
    if normalization is not None:
        assert normalization in ['sym'], 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    theta_attr = torch.cat([edge_weight, -edge_weight], dim=0)
    sym_attr = torch.cat([edge_weight, edge_weight], dim=0)
    edge_attr = torch.stack([sym_attr, theta_attr], dim=1)

    edge_index_sym, edge_attr = coalesce(edge_index, edge_attr, num_nodes, "add")

    return edge_attr

    edge_weight_sym = edge_attr[:, 0]
    edge_weight_sym = edge_weight_sym/2

    row, col = edge_index_sym[0], edge_index_sym[1]
    deg = scatter(edge_weight_sym, row, dim=0, dim_size=num_nodes, reduce='sum')

    edge_weight_q = torch.exp(1j * 2 * np.pi * q * edge_attr[:, 1])
    return edge_weight_q
