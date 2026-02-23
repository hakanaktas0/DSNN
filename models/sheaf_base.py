# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch_sparse
import numpy as np

class SheafDiffusion(nn.Module):
    """Base class for sheaf diffusion models."""

    def __init__(self, edge_index, args,directed_edge_index=None):
        super(SheafDiffusion, self).__init__()

        assert args['d'] > 0
        self.d = args['d']
        self.edge_index = edge_index
        self.directed_edge_index = directed_edge_index
        self.add_lp = args['add_lp']
        self.add_hp = args['add_hp']

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.hidden_dim = args['hidden_channels'] * self.final_d
        self.device = args['device']
        self.graph_size = args['graph_size']
        self.layers = args['layers']
        self.normalised = args['normalised']
        self.deg_normalised = args['deg_normalised']
        self.nonlinear = not args['linear']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        self.left_weights = args['left_weights']
        self.right_weights = args['right_weights']
        self.sparse_learner = args['sparse_learner']
        self.use_act = args['use_act']
        if args['pred_task'] == 'node_classification':
            self.input_dim = args['input_dim']
            self.output_dim = args['output_dim']
        else:
            if args['num_class_link'] == 2:
                self.input_dim = 2
                self.output_dim = 2
            if args['num_class_link'] == 3:
                self.input_dim = 2
                self.output_dim = 3
        self.hidden_channels = args['hidden_channels']
        self.layers = args['layers']
        self.sheaf_act = args['sheaf_act']
        self.second_linear = args['second_linear']
        self.orth_trans = args['orth']
        self.use_edge_weights = args['edge_weights']
        self.t = args['max_t']
        self.time_range = torch.tensor([0.0, self.t], device=self.device)
        self.laplacian_builder = None


        # Complex Sheaf Stuff
        self.complex_separate_linears = args['complex_separate_linears']  # DONE
        self.complex_separate_dropout = args['complex_separate_dropout'] # DONE
        self.complex_add_block_diag = args['complex_add_block_diag'] # DONE
        self.complex_copy_values = args['complex_copy_values'] # DONE
        self.complex_q = args['complex_q'] # DONE
        self.complex_activation = args['complex_activation'] # DONE
        self.complex_use_intro_act = args['complex_use_intro_act'] # DONE
        self.complex_x_dropout = args['complex_x_dropout'] # DONE
        self.complex_start = args['complex_start'] # DONE
        self.trainable_q = args['trainable_q']

        # task
        self.pred_task = args['pred_task']


    def complex_sparse_multiplication(self, L, x):
        # Assuming L is provided as:
        # L = (L_indices, L_values) where:
        # L_indices: a 2 x N tensor of indices (on cuda:0)
        # L_values: an N-element tensor of complex values (on cuda:0)
        # Extract the real and imaginary parts of the sparse matrix L

        #     L_indices = L[0]
        # L_values = L[1]

        # Size of the square sparse matrix
        m, n = x.size(0), x.size(0)
        # Create the real and imaginary sparse tensors for L
        L_real_sparse = torch.sparse_coo_tensor(L[0], L[1].real, size=(m, n), dtype=torch.float).to(x.device)
        L_imag_sparse = torch.sparse_coo_tensor(L[0], L[1].imag, size=(m, n), dtype=torch.float).to(x.device)
        if self.constrained_direction:
            # multiply here by the direction in a elementwise
            # relu is a limitation
            L_imag_sparse = torch.mul(torch.abs(L_imag_sparse), self.direction)
        # Separate the real and imaginary parts of the dense complex tensor x
        x_real = x.real
        x_imag = x.imag

        # Perform the four sparse multiplications using torch_sparse.spmm directly with L_indices and L_values parts.
        # out_real = (L_real * x_real) - (L_imag * x_imag)
        # Perform sparse matrix multiplication for the real and imaginary parts separately
        out_real = torch_sparse.spmm(L_real_sparse._indices(), L_real_sparse._values(), m, n, x_real) - \
                   torch_sparse.spmm(L_imag_sparse._indices(), L_imag_sparse._values(), m, n, x_imag)

        out_imag = torch_sparse.spmm(L_real_sparse._indices(), L_real_sparse._values(), m, n, x_imag) + \
                   torch_sparse.spmm(L_imag_sparse._indices(), L_imag_sparse._values(), m, n, x_real)
        # Combine the real and imaginary parts into a complex tensor
        return torch.complex(out_real, out_imag)

    def update_edge_index(self, edge_index):
        assert edge_index.max() <= self.graph_size
        self.edge_index = edge_index
        self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        sheaf_learners, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learners.append(param)
            else:
                others.append(param)
        assert len(sheaf_learners) > 0
        assert len(sheaf_learners) + len(others) == len(list(self.parameters()))
        return sheaf_learners, others

    def left_right_linear(self, x, left=None, right=None):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x
    def edge_q_to_adj(self,edge_q):
        edge_index = self.edge_index.detach().cpu().numpy()
        adj_mat = torch.zeros((np.max(edge_index)+1,np.max(edge_index)+1),dtype=torch.complex64)
        for i in range(edge_index.shape[1]):
            edge = edge_index[:,i]
            adj_mat[edge[0],edge[1]] = edge_q[i]
        return adj_mat.to(self.edge_index.device)
    def L_to_lap(self,L):
        edge_index = L[0].detach().cpu().numpy()
        laplacian = torch.zeros((np.max(edge_index)+1,np.max(edge_index)+1),dtype=L[1].dtype)
        for i in range(edge_index.shape[1]):
            laplacian[edge_index[0,i],edge_index[1,i]] = L[1][i]
        return laplacian.to(self.edge_index.device)