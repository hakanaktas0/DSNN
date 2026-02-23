import torch
import numpy as np

def sparse_kron(input: torch.Tensor, other: torch.Tensor):
    assert input.ndim == other.ndim
    input_indices = input.indices()
    other_indices = other.indices()

    input_indices_expanded = input_indices.expand(other_indices.shape[1], *input_indices.shape).T * torch.tensor(other.shape).reshape(1,-1,1).to(torch.int32).to(input.device)
    other_indices_expanded = other_indices.expand(input_indices.shape[1], *other_indices.shape).to(torch.int32)
    new_indices = torch.permute(input_indices_expanded + other_indices_expanded, (1,0,2)).reshape(input.ndim,-1)

    new_values = torch.kron(input.values(), other.values())

    if new_indices.ndim == 1:
        new_indices = new_indices.reshape([input.ndim, 0])

    new_shape = [n * m for n, m in zip(input.shape, other.shape)]

    return torch.sparse_coo_tensor(new_indices, new_values, new_shape, dtype=input.dtype, device=input.device)


def efficient_sparse_kron(input: torch.Tensor, other: torch.Tensor):
    assert input.layout == torch.sparse_coo and other.layout == torch.sparse_coo
    assert input.ndim == other.ndim

    D = input.ndim
    device = input.device

    input_indices = input.indices()  # [D, N]
    other_indices = other.indices()  # [D, M]
    input_values = input.values()    # [N]
    other_values = other.values()    # [M]
    N = input_values.numel()
    M = other_values.numel()

    # Compute output shape
    out_shape = [input.shape[i] * other.shape[i] for i in range(D)]

    # Compute all combinations of nonzero positions
    # For indices: broadcast addition of input_idx * other_shape + other_idx
    # Expand to [D, N, 1] and [D, 1, M]
    input_idx_scaled = input_indices.unsqueeze(2) * torch.tensor(other.shape, device=device).view(D, 1, 1)
    other_idx_exp = other_indices.unsqueeze(1)  # [D, 1, M]

    out_indices = input_idx_scaled + other_idx_exp  # [D, N, M]
    out_indices = out_indices.reshape(D, -1)        # [D, N*M]

    # Compute values as outer product: [N, 1] * [1, M] -> [N, M] -> [N*M]
    out_values = (input_values.view(-1, 1) * other_values.view(1, -1)).reshape(-1)

    return torch.sparse_coo_tensor(out_indices, out_values, size=out_shape, dtype=input.dtype, device=device)

def sparse_eye(size,device):
    indices = torch.arange(size)
    indices = torch.stack([indices, indices])
    values = torch.ones(size)
    return torch.sparse_coo_tensor(indices, values, (size, size),device=device)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


#
# from models.baselines import cheb_poly_sparse, hermitian_decomp_sparse
# from models.baselines import GNN, MLP, GCNII, SAGEModel, GPRGNN, GGCN, H2GCN, MixHopNetwork, FAGCN, SigMaNet,  MagNet
# def init_model(model_cls, args, data):
#     if args.model == 'michael':
#         model = model_cls(num_features=args.input_dim,
#                           num_classes=args.output_dim,
#                           hidden_dim=args.hidden_channels,
#                           num_layers=args.layers,
#                           dropout=args.dropout,
#                           conv_type="dir-gcn",
#                           jumping_knowledge=args.jk,
#                           normalize=True,
#                           alpha=args.alpha,
#                           learn_alpha=False)
#     elif args.model == 'MagNet':
#         L = hermitian_decomp_sparse(data.edge_index[0].cpu(), data.edge_index[1].cpu(), data.y.size(-1), args.q,
#                                     norm=True, laplacian=True, max_eigen=2.0, gcn_appr=False)
#         L = cheb_poly_sparse(L, 1)
#         # convert dense laplacian to sparse matrix
#         L_img = []
#         L_real = []
#         for i in range(len(L)):
#             L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(args.device))
#             L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(args.device))  # .to(device) )
#         # Lpalcian calculation
#         model = model_cls(
#             data.x.size(-1),
#             L_real,
#             L_img,
#             K=1,
#             label_dim=args.output_dim,
#             layer=args.layers,
#             activation=True,
#             num_filter=args.hidden_channels,
#             dropout=args.dropout)
#     elif args.model == 'SigMaNet':
#         model = model_cls(
#             num_features=args.input_dim,
#             hidden=args.hidden_channels,
#             label_dim=args.output_dim,
#             layer=args.layers,
#             dropout=args.dropout,
#             edge_index=data.edge_index,
#             X_real=data.x,
#             i_complex=args.i_complex,
#             gcn=args.gcn)
#     elif args.model == 'MLP':
#         model = model_cls(nfeat=args.input_dim,
#                           nlayers=args.layers,
#                           nhidden=args.hidden_channels,
#                           nclass=args.output_dim,
#                           dropout=args.dropout,
#                           use_res=args.use_res)
#     elif args.model == 'GCNII':
#         model = model_cls(nfeat=args.input_dim,
#                           nlayers=args.layers,
#                           nhidden=args.hidden_channels,
#                           nclass=args.output_dim,
#                           dropout=args.dropout,
#                           lamda=args.lamda,
#                           alpha=args.alpha_1,
#                           variant=args.variant,
#                           adj=data.edge_index, )
#     elif args.model == 'Sage':
#         model = model_cls(
#             input_dim=args.input_dim,
#             out_dim=args.output_dim,
#             filter_num=args.hidden_channels,
#             dropout=args.dropout,
#             layer=args.layers,
#             directed=args.directed)
#     elif args.model == "GPRGNN":
#         model = model_cls(
#             nlayers=args.layers,
#             nfeat=args.input_dim,
#             nhidden=args.hidden_channels,
#             nclass=args.output_dim,
#             dropout=args.dropout,
#             dprate_GPRGNN=args.dprate_GPRGNN,
#             alpha_GPRGNN=args.alpha_GPRGNN,
#             Gamma_GPRGNN=args.Gamma_GPRGNN,
#             Init_GPRGNN=args.Init_GPRGNN,
#             ppnp_GPRGNN=args.ppnp_GPRGNN,
#             directed=args.directed,
#         )
#     elif args.model == 'GGCN':
#         use_degree = (args.no_degree) & (not args.row_normalized_adj)
#         use_sign = args.no_sign
#         use_decay = args.no_decay
#         use_bn = (args.use_bn) & (not use_decay)
#         use_ln = (args.use_ln) & (not use_decay) & (not use_bn)
#         model = model_cls(nlayers=args.layers,
#                           nfeat=args.input_dim,
#                           nhidden=args.hidden_channels,
#                           nclass=args.output_dim,
#                           dropout=args.dropout,
#                           decay_rate=args.decay_rate,
#                           exponent=args.exponent,
#                           use_degree=use_degree,
#                           use_sign=use_sign,
#                           use_decay=use_decay,
#                           scale_init=args.scale_init,
#                           deg_intercept_init=args.deg_intercept_init,
#                           use_bn=use_bn,
#                           use_ln=use_ln)
#     elif args.model == 'H2GCN':
#         model = model_cls(
#             feat_dim=args.input_dim,
#             hidden_dim=args.hidden_channels,
#             class_dim=args.output_dim,
#             layer=args.layers,
#             dropout=args.dropout,
#             directed=args.directed,
#             use_relu=args.use_relu)
#     elif args.model == 'MixHop':
#         model = model_cls(
#             feature_number=args.input_dim,
#             class_number=args.output_dim,
#             layers_1=args.layers_1,
#             layers_2=args.layers_2,
#             dropout=args.dropout,
#             edge_index=data.edge_index, )
#     elif args.model == 'FAGCN':
#         model = model_cls(
#             g=data.edge_index,
#             in_dim=args.input_dim,
#             out_dim=args.output_dim,
#             hidden_dim=args.hidden_channels,
#             layer_num=args.layers,
#             dropout=args.dropout,
#             eps=args.eps)
#     else:
#         model = model_cls(data.edge_index, vars(args))
#     return model
#
# def init_model_cls(args):
#     if args.model == 'michael':
#         model_cls = GNN
#     elif args.model == 'MagNet':
#         model_cls = MagNet
#     elif args.model == 'SigMaNet':
#         model_cls = SigMaNet
#     elif args.model == 'MLP':
#         model_cls = MLP
#     elif args.model == 'GCNII':
#         model_cls = GCNII
#     elif args.model == 'Sage':
#         model_cls = SAGEModel
#     elif args.model == 'GPRGNN':
#         model_cls = GPRGNN
#     elif args.model == 'GGCN':
#         model_cls = GGCN
#     elif args.model == 'H2GCN':
#         model_cls = H2GCN
#     elif args.model == 'MixHop':
#         model_cls = MixHopNetwork
#     elif args.model == 'FAGCN':
#         model_cls = FAGCN
#     else:
#         raise ValueError(f'Unknown model {args.model}')
#     return model_cls
# 
