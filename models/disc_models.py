# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_sparse
import numpy as np

from torch import nn
from models.sheaf_base import SheafDiffusion
from models import laplacian_builders as lb
from models.sheaf_models import  ComplexLocalConcatSheafLearner, ComplexLocalConcatSheafLearnerVariant,ComplexEdgeWeightLearner, LocalConcatSheafLearnerVariant, LocalConcatSheafLearner, EdgeWeightLearner
from utils.magnetic_laplacian import get_phase, get_edge_attr
from utils.complex_activations import  complex_relu_layer,complex_elu_layer
from utils.complex_dropout import complex_merged_dropout, complex_separate_dropout
from utils.func import sparse_kron, sparse_eye, efficient_sparse_kron


def sparse_to_dense(sparse_tensor: torch.Tensor) -> torch.Tensor:
    assert sparse_tensor.layout == torch.sparse_coo

    indices = sparse_tensor.indices()          # [2, nnz]
    values = sparse_tensor.values()            # [nnz]
    shape = sparse_tensor.shape

    dense = torch.zeros(shape, dtype=values.dtype, device=values.device)

    dense.index_put_(
        tuple(indices),
        values,
        accumulate=True
    )

    return dense

class DiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteDiagSheafDiffusion, self).__init__(edge_index, args,directed_edge_index)
        assert args['d'] > 0


        if self.complex_separate_linears:
            self.lin_right_weights_imag = nn.ModuleList()
            self.lin_left_weights_imag = nn.ModuleList()

        if self.right_weights:
            self.lin_right_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data, gain=0.5)
                if self.complex_separate_linears:
                    self.lin_right_weights_imag.append(
                        nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                    nn.init.orthogonal_(self.lin_right_weights_imag[-1].weight.data, gain=0.5)

        if self.left_weights:
            self.lin_left_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
                if self.complex_separate_linears:
                    self.lin_left_weights_imag.append(nn.Linear(self.final_d, self.final_d, bias=False))
                    nn.init.eye_(self.lin_left_weights_imag[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        if self.use_act:
            if self.complex_activation == 'relu':
                self.complex_act = complex_relu_layer()
            elif self.complex_activation == 'elu':
                self.complex_act = complex_elu_layer()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearnerVariant(self.final_d,
                                                                          self.hidden_channels,
                                                                          out_shape=(self.d,),
                                                                          sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))



        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        if self.complex_start and self.complex_separate_linears:
            self.lin1_real = nn.Linear(self.input_dim, self.hidden_dim)
            self.lin1_imag = nn.Linear(self.input_dim, self.hidden_dim)

            if self.second_linear:
                self.lin12_real = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.lin12_imag = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
            if self.second_linear:
                self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 4, self.output_dim)


        if self.complex_separate_dropout:
            self.complex_dropout = complex_separate_dropout
        else:
            self.complex_dropout = complex_merged_dropout

        phase = get_phase(self.directed_edge_index, q=self.complex_q)

        # DENSE
        # adj_edge_q = self.edge_q_to_adj(edge_q)
        # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=edge_q.device)
        # self.kron_prod = torch.kron(adj_edge_q, ones_mat)


        # SPARSE IMPLEMENTATION
        sparse_phase = torch.sparse_coo_tensor(self.edge_index,phase,size=(self.graph_size,self.graph_size),device=self.device)
        ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device)
        self.kron_prod_phase = efficient_sparse_kron(sparse_phase.coalesce(),ones_mat.to_sparse_coo().coalesce())

        if self.complex_add_block_diag:
            # diag = torch.eye(self.graph_size, device=self.device)
            # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device)
            # block_diag = torch.kron(diag, ones_mat)  # 915 x 915

            diag = sparse_eye(self.graph_size, device=self.device)
            ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device).to_sparse_coo()
            block_diag = efficient_sparse_kron(diag.coalesce(), ones_mat.coalesce())
            self.kron_prod_phase += block_diag



    def forward(self, x,index=None):
        if self.complex_start:
            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real
            x = self.complex_dropout(x, p=self.input_dropout, training=self.training)
            if self.complex_separate_linears:
                x_real = self.lin1_real(x.real)
                x_imag = self.lin1_imag(x.imag)
                x = x_real.to(torch.cfloat)
                x.imag = x_imag
            else:
                real = self.lin1(x.real)
                imag = self.lin1(x.imag)
                x = torch.complex(real, imag)

            if self.use_act and self.complex_use_intro_act:
                real, imag = self.complex_act(x.real, x.imag)
                x = torch.complex(real, imag)

            x = self.complex_dropout(x, p=self.complex_x_dropout, training=self.training)
            if self.second_linear:
                if self.complex_separate_linears:
                    x_real = self.lin12_real(x.real)
                    x_imag = self.lin12_imag(x.imag)
                    x = torch.complex(x_real, x_imag)
                else:
                    x_real = self.lin12(x.real)
                    x_imag = self.lin12(x.imag)
                    x = torch.complex(x_real, x_imag)

            x = x.view(self.graph_size * self.final_d, -1)
        else:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act and self.complex_use_intro_act:
                x = F.elu(x)
            x = F.dropout(x, p=self.complex_x_dropout, training=self.training)

            if self.second_linear:
                x = self.lin12(x)
            x = x.view(self.graph_size * self.final_d, -1)

            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = self.complex_dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = self.complex_dropout(x, p=self.dropout, training=self.training)

            if self.complex_separate_linears:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights_imag[layer], self.lin_right_weights_imag[layer])
                else:
                # adding the only right option
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights_imag[layer])
                    x = torch.complex(x_real, x_imag)
            else:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights[layer], self.lin_right_weights[layer])
                # adding the only right option
                else:
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights[layer])
                    x = torch.complex(x_real, x_imag)

            sparse_laplacian = torch.sparse_coo_tensor(L[0], L[1], (x.shape[0], x.shape[0],))
            magnet_laplacian = self.kron_prod_phase * sparse_laplacian

            # sparse_locations = sparse_laplacian._indices()
            # kron_values = self.kron_prod[sparse_locations[0], sparse_locations[1]]
            # new_values = sparse_laplacian._values() * kron_values
            # magnet_laplacian = torch.sparse_coo_tensor(sparse_locations, new_values, sparse_laplacian.shape)
            #
            # locations = self.kron_prod.to_sparse_coo()._indices()
            # lapcacian_values = sparse_laplacian[locations[0], locations[1]]
            # new_vals = lapcacian_values * self.kron_prod._values()
            # magnet_laplacian_1 = torch.sparse_coo_tensor(locations,new_vals, sparse_laplacian.shape)

            x = torch_sparse.spmm(magnet_laplacian._indices(), magnet_laplacian._values(), x.size(0), x.size(0), x)
            # x = torch.matmul(magnet_laplacian,x)

            if self.use_act:
                real, imag = self.complex_act(x.real, x.imag)
                x = torch.complex(real, imag)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0
        if self.pred_task == 'node_classification':
            x = torch.cat((x.real, x.imag), dim=-1)
            # To detect the numerical instabilities of SVD.
            assert torch.all(torch.isfinite(x))

            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]], x.imag[index[:, 0]], x.imag[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)


class DiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteBundleSheafDiffusion, self).__init__(edge_index, args,directed_edge_index)
        assert args['d'] > 1
        assert not self.deg_normalised


        if self.complex_separate_linears:
            self.lin_right_weights_imag = nn.ModuleList()
            self.lin_left_weights_imag = nn.ModuleList()

        if self.right_weights:
            self.lin_right_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data, gain=0.5)
                if self.complex_separate_linears:
                    self.lin_right_weights_imag.append(
                        nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                    nn.init.orthogonal_(self.lin_right_weights_imag[-1].weight.data, gain=0.5)

        if self.left_weights:
            self.lin_left_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
                if self.complex_separate_linears:
                    self.lin_left_weights_imag.append(nn.Linear(self.final_d, self.final_d, bias=False))
                    nn.init.eye_(self.lin_left_weights_imag[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()
        if self.use_act:
            if self.complex_activation == 'relu':
                self.complex_act = complex_relu_layer()
            elif self.complex_activation == 'elu':
                self.complex_act = complex_elu_layer()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearnerVariant(self.final_d,
                                                                                 self.hidden_channels,
                                                                                 out_shape=(self.get_param_size(),),
                                                                                 sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

            if self.use_edge_weights:
                self.weight_learners.append(ComplexEdgeWeightLearner(self.hidden_dim, edge_index))


        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        if self.complex_start and self.complex_separate_linears:
            self.lin1_real = nn.Linear(self.input_dim, self.hidden_dim)
            self.lin1_imag = nn.Linear(self.input_dim, self.hidden_dim)

            if self.second_linear:
                self.lin12_real = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.lin12_imag = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
            if self.second_linear:
                self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 4, self.output_dim)



        if self.complex_separate_dropout:
            self.complex_dropout = complex_separate_dropout
        else:
            self.complex_dropout = complex_merged_dropout

        phase = get_phase(self.directed_edge_index, q=self.complex_q)

        # adj_edge_q = self.edge_q_to_adj(edge_q)
        # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=edge_q.device)
        # self.kron_prod = torch.kron(adj_edge_q, ones_mat)

        sparse_phase = torch.sparse_coo_tensor(self.edge_index, phase, size=(self.graph_size, self.graph_size),device=phase.device)
        ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=phase.device)
        self.kron_prod_phase = efficient_sparse_kron(sparse_phase.coalesce(), ones_mat.to_sparse_coo().coalesce())

        if self.complex_add_block_diag:
            # diag = torch.eye(self.graph_size, device=self.device)
            # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device)
            # block_diag = torch.kron(diag, ones_mat)  # 915 x 915

            diag = sparse_eye(self.graph_size, device=self.device)
            ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device).to_sparse_coo()
            block_diag = efficient_sparse_kron(diag.coalesce(), ones_mat.coalesce())
            self.kron_prod_phase += block_diag


    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2


    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x,index=None):
        if self.complex_start:
            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real
            x = self.complex_dropout(x, p=self.input_dropout, training=self.training)
            if self.complex_separate_linears:
                x_real = self.lin1_real(x.real)
                x_imag = self.lin1_imag(x.imag)
                x = x_real.to(torch.cfloat)
                x.imag = x_imag
            else:
                real = self.lin1(x.real)
                imag = self.lin1(x.imag)
                x = torch.complex(real, imag)

            if self.use_act and self.complex_use_intro_act:
                real, imag = self.complex_act(x.real, x.imag)
                x = torch.complex(real, imag)

            x = self.complex_dropout(x, p=self.complex_x_dropout, training=self.training)
            if self.second_linear:
                if self.complex_separate_linears:
                    x_real = self.lin12_real(x.real)
                    x_imag = self.lin12_imag(x.imag)
                    x = torch.complex(x_real, x_imag)
                else:
                    x_real = self.lin12(x.real)
                    x_imag = self.lin12(x.imag)
                    x = torch.complex(x_real, x_imag)

            x = x.view(self.graph_size * self.final_d, -1)
        else:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act and self.complex_use_intro_act:
                x = F.elu(x)
            x = F.dropout(x, p=self.complex_x_dropout, training=self.training)

            if self.second_linear:
                x = self.lin12(x)
            x = x.view(self.graph_size * self.final_d, -1)

            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = self.complex_dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)
            x = self.complex_dropout(x, p=self.dropout, training=self.training)

            if self.complex_separate_linears:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights_imag[layer], self.lin_right_weights_imag[layer])
                else:
                # adding the only right option
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights_imag[layer])
                    x = torch.complex(x_real, x_imag)
            else:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights[layer], self.lin_right_weights[layer])
                # adding the only right option
                else:
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights[layer])
                    x = torch.complex(x_real, x_imag)

            sparse_laplacian = torch.sparse_coo_tensor(L[0], L[1], (x.shape[0], x.shape[0],))
            magnet_laplacian = self.kron_prod_phase * sparse_laplacian

            # sparse_locations = sparse_laplacian._indices()
            # kron_values = self.kron_prod[sparse_locations[0], sparse_locations[1]]
            # new_values = sparse_laplacian._values() * kron_values
            # magnet_laplacian = torch.sparse_coo_tensor(sparse_locations, new_values, sparse_laplacian.shape)
            #
            # locations = self.kron_prod.to_sparse_coo()._indices()
            # lapcacian_values = sparse_laplacian[locations[0], locations[1]]
            # new_vals = lapcacian_values * self.kron_prod._values()
            # magnet_laplacian_1 = torch.sparse_coo_tensor(locations,new_vals, sparse_laplacian.shape)

            x = torch_sparse.spmm(magnet_laplacian._indices(), magnet_laplacian._values(), x.size(0), x.size(0), x)
            # x = torch.matmul(magnet_laplacian,x)

            if self.use_act:
                real, imag = self.complex_act(x.real, x.imag)
                x = torch.complex(real, imag)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        if self.pred_task == 'node_classification':
            x = torch.cat((x.real, x.imag), dim=-1)
            # To detect the numerical instabilities of SVD.
            assert torch.all(torch.isfinite(x))

            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]], x.imag[index[:, 0]], x.imag[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteGeneralSheafDiffusion, self).__init__(edge_index, args,directed_edge_index)
        # assert args['d'] > 1
        # self.directed_edge_index = directed_edge_index
        # self.lin_right_weights = nn.ModuleList()
        # self.lin_left_weights = nn.ModuleList()

        if self.complex_separate_linears:
            self.lin_right_weights_imag = nn.ModuleList()
            self.lin_left_weights_imag = nn.ModuleList()

        if self.right_weights:
            self.lin_right_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data,gain=0.5)
                if self.complex_separate_linears:
                    self.lin_right_weights_imag.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                    nn.init.orthogonal_(self.lin_right_weights_imag[-1].weight.data,gain=0.5)

        if self.left_weights:
            self.lin_left_weights = nn.ModuleList()
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
                if self.complex_separate_linears:
                    self.lin_left_weights_imag.append(nn.Linear(self.final_d, self.final_d, bias=False))
                    nn.init.eye_(self.lin_left_weights_imag[-1].weight.data)


        self.sheaf_learners = nn.ModuleList()

        if self.use_act:
            if self.complex_activation == 'relu':
                self.complex_act = complex_relu_layer()
            elif self.complex_activation == 'elu':
                self.complex_act = complex_elu_layer()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(ComplexLocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))

        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        if self.complex_start and self.complex_separate_linears:
            self.lin1_real = nn.Linear(self.input_dim, self.hidden_dim)
            self.lin1_imag = nn.Linear(self.input_dim, self.hidden_dim)

            if self.second_linear:
                self.lin12_real = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.lin12_imag = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
            if self.second_linear:
                self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 4, self.output_dim)



        if self.complex_separate_dropout:
            self.complex_dropout = complex_separate_dropout
        else:
            self.complex_dropout = complex_merged_dropout

        if self.trainable_q:
            self.train_q = nn.Parameter(torch.Tensor(1).fill_(self.complex_q))
            self.edge_attr = get_edge_attr(self.directed_edge_index)
        else:
            phase = get_phase(self.directed_edge_index, q=self.complex_q)

            # adj_edge_q = self.edge_q_to_adj(edge_q)
            # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=edge_q.device)
            # self.kron_prod = torch.kron(adj_edge_q, ones_mat)

            sparse_phase = torch.sparse_coo_tensor(self.edge_index, phase, size=(self.graph_size, self.graph_size),device=phase.device)
            ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=phase.device)
            self.kron_prod_phase = efficient_sparse_kron(sparse_phase.coalesce(), ones_mat.to_sparse_coo().coalesce())



            #
            if self.complex_add_block_diag:
                # diag = torch.eye(self.graph_size, device=self.device)
                # ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device)
                # block_diag = torch.kron(diag, ones_mat)  # 915 x 915

                diag = sparse_eye(self.graph_size, device=self.device)
                ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=self.device).to_sparse_coo()
                block_diag = efficient_sparse_kron(diag.coalesce(),ones_mat.coalesce())
                self.kron_prod_phase += block_diag
            #


    def forward(self, x,index=None):
        if self.complex_start:
            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real
            x = self.complex_dropout(x, p=self.input_dropout, training=self.training)
            if self.complex_separate_linears:
                x_real = self.lin1_real(x.real)
                x_imag = self.lin1_imag(x.imag)
                x = x_real.to(torch.cfloat)
                x.imag = x_imag
            else:
                real = self.lin1(x.real)
                imag = self.lin1(x.imag)
                x = torch.complex(real, imag)

            if self.use_act and self.complex_use_intro_act:
                real, imag = self.complex_act(x.real, x.imag)
                x = torch.complex(real, imag)

            x = self.complex_dropout(x, p=self.complex_x_dropout, training=self.training)
            if self.second_linear:
                if self.complex_separate_linears:
                    x_real = self.lin12_real(x.real)
                    x_imag = self.lin12_imag(x.imag)
                    x = torch.complex(x_real, x_imag)
                else:
                    x_real = self.lin12(x.real)
                    x_imag = self.lin12(x.imag)
                    x = torch.complex(x_real, x_imag)

            x = x.view(self.graph_size * self.final_d, -1)
        else:
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            x = self.lin1(x)
            if self.use_act and self.complex_use_intro_act:
                x = F.elu(x)
            x = F.dropout(x, p=self.complex_x_dropout, training=self.training)

            if self.second_linear:
                x = self.lin12(x)
            x = x.view(self.graph_size * self.final_d, -1)

            x = x.to(torch.cfloat)
            if self.complex_copy_values:
                x.imag = x.real

        # another option is use the MLP to map the real x to a complex x
        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = self.complex_dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = self.complex_dropout(x, p=self.dropout, training=self.training)

            if self.complex_separate_linears:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights_imag[layer], self.lin_right_weights_imag[layer])
                else:
                # adding the only right option
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights_imag[layer])
                    x = torch.complex(x_real, x_imag)
            else:
                if self.left_weights and self.right_weights:
                    x.real = self.left_right_linear(x.real, self.lin_left_weights[layer], self.lin_right_weights[layer])
                    x.imag = self.left_right_linear(x.imag, self.lin_left_weights[layer], self.lin_right_weights[layer])
                # adding the only right option
                else:
                    x_real = self.left_right_linear(x.real, right=self.lin_right_weights[layer])
                    x_imag = self.left_right_linear(x.imag, right=self.lin_right_weights[layer])
                    x = torch.complex(x_real, x_imag)


            if self.trainable_q:
                phase = torch.exp(1j * 2 * np.pi * self.train_q.clamp(0,0.25) * self.edge_attr[:, 1])
                sparse_phase = torch.sparse_coo_tensor(self.edge_index, phase, size=(self.graph_size, self.graph_size),
                                                       device=phase.device)
                ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32, device=phase.device)
                self.kron_prod_phase = torch.kron(sparse_to_dense(sparse_phase.coalesce()), ones_mat)

                if self.complex_add_block_diag:
                    diag = torch.eye(self.graph_size, device=self.device)
                    ones_mat = torch.ones((self.final_d, self.final_d), dtype=torch.float32,
                                          device=self.device)
                    block_diag = torch.kron(diag, ones_mat)
                    self.kron_prod_phase += block_diag
                sparse_laplacian = torch.sparse_coo_tensor(L[0], L[1], (x.shape[0], x.shape[0],))
                magnet_laplacian = self.kron_prod_phase * sparse_to_dense(sparse_laplacian.coalesce())
                x = torch.matmul(magnet_laplacian, x)
            else:
                sparse_laplacian = torch.sparse_coo_tensor(L[0], L[1], (x.shape[0], x.shape[0],))
                magnet_laplacian =  self.kron_prod_phase * sparse_laplacian
                x = torch_sparse.spmm(magnet_laplacian._indices(), magnet_laplacian._values(), x.size(0), x.size(0), x)
            # sparse_locations = sparse_laplacian._indices()
            # kron_values = self.kron_prod[sparse_locations[0], sparse_locations[1]]
            # new_values = sparse_laplacian._values() * kron_values
            # magnet_laplacian = torch.sparse_coo_tensor(sparse_locations, new_values, sparse_laplacian.shape)
            #
            # locations = self.kron_prod.to_sparse_coo()._indices()
            # lapcacian_values = sparse_laplacian[locations[0], locations[1]]
            # new_vals = lapcacian_values * self.kron_prod._values()
            # magnet_laplacian_1 = torch.sparse_coo_tensor(locations,new_vals, sparse_laplacian.shape)

            # x = torch_sparse.spmm(magnet_laplacian._indices(), magnet_laplacian._values(), x.size(0), x.size(0), x)
            # x = torch.matmul(magnet_laplacian,x)

            if self.use_act:
                real, imag = self.complex_act(x.real,x.imag)
                x = torch.complex(real, imag)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0
        if self.pred_task == 'node_classification':
            x = torch.cat((x.real, x.imag), dim=-1)
            assert torch.all(torch.isfinite(x))

            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]], x.imag[index[:, 0]], x.imag[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

# Real valued implementations from https://github.com/twitter-research/neural-sheaf-diffusion

class DiscreteDiagSheafDiffusionReal(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteDiagSheafDiffusionReal, self).__init__(edge_index, args,directed_edge_index)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                         normalised=self.normalised,
                                                         deg_normalised=self.deg_normalised,
                                                         add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x, index = None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        if self.pred_task == 'node_classification':
            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        #
        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)


class DiscreteBundleSheafDiffusionReal(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteBundleSheafDiffusionReal, self).__init__(edge_index, args,directed_edge_index)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                                                                          self.hidden_channels,
                                                                          out_shape=(self.get_param_size(),),
                                                                          sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))

            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left=None, right=None):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x,index=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0


        if self.pred_task == 'node_classification':
            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        #
        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)


class DiscreteGeneralSheafDiffusionReal(SheafDiffusion):

    def __init__(self, edge_index, args,directed_edge_index):
        super(DiscreteGeneralSheafDiffusionReal, self).__init__(edge_index, args,directed_edge_index)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                                                                          self.hidden_channels,
                                                                          out_shape=(self.d, self.d),
                                                                          sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.pred_task == 'node_classification':
            self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            self.lin2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
    def left_right_linear(self, x, left=None, right=None):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x,index=None):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        if self.pred_task == 'node_classification':
            x = x.reshape(self.graph_size, -1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)
        else:
            x = x.reshape(self.graph_size, -1)
            x = torch.cat((x.real[index[:, 0]], x.real[index[:, 1]]), dim=-1)
            x = self.lin2(x)
            return F.log_softmax(x, dim=1)

        # x = x.reshape(self.graph_size, -1)
        # x = self.lin2(x)
        # return F.log_softmax(x, dim=1)