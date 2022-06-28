from functools import total_ordering
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as thgeo
import numpy as np


def graph_laplacian(graph):
    return torch.eye(graph.shape[1], device = graph.device) - graph


def get_eye(N, device, sparse=True):
    if sparse:
        identity_indices = torch.cat([torch.arange(N).reshape(1,-1) for _ in range(2)], dim=0).to(device)
        return torch.sparse_coo_tensor(
            identity_indices,
            torch.ones(N, device=device),
            (N, N)
        )
    else:
        return torch.eye(N, device = device)


def normalise_graph(graph, remove_diagonal = True, self_connection = False, precomputed_sparse_identity_matrix = None):
    N = graph.shape[0]
    if type(graph) == torch.Tensor:
        if not graph.is_sparse:
            eye = torch.eye(N, device=graph.device)
            if remove_diagonal:
                graph = graph - graph.diag() * eye # Remove all self connections
            if self_connection:
                graph = graph + eye
            G_sum = graph.sum(1)
            G_sum[G_sum == 0] = 1
            D = torch.float_power(G_sum, -0.5).float() * eye
            return D @ graph @ D
        else:
            identity_indices = torch.cat([torch.arange(N).reshape(1,-1) for _ in range(2)], dim=0).to(graph.device)
            if self_connection or remove_diagonal:
                if precomputed_sparse_identity_matrix is None:
                    eye = torch.sparse_coo_tensor(
                        identity_indices,
                        torch.ones(N, device=graph.device),
                        (N, N)
                    )
                else:
                    eye = precomputed_sparse_identity_matrix
            if remove_diagonal:
                graph = graph - (graph * eye)
            if self_connection:
                graph = graph + eye
            G_sums = torch.sparse.sum(graph, dim=1).to_dense()
            G_sums[G_sums == 0.] = 1.
            assert len(G_sums) == N, "There are zero-sum rows in graph."
            D = torch.sparse_coo_tensor(
                identity_indices,
                torch.float_power(G_sums, -1),
                (N, N),
                dtype=torch.float32
            )
            return torch.sparse.mm(D, graph)
    elif type(graph) == np.ndarray:
        if self_connection:
            if remove_diagonal: graph = graph - np.diag(graph) * np.eye(N) # Remove all self connections
            graph = graph + np.eye(N)
        D = np.power(graph.sum(1), -0.5) * np.eye(N)
        return D @ graph @ D
    else:
        print(f"Graph type {type(graph)} not recognised.")


class GraphConv(nn.Module):
    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        attention_features: int = 1,
        mask: bool = True,
        mask_weighting: bool = False,
        mask_power: int = 1,
        standard_gcn: bool = False,
        laplacian: bool = False,
        bias: bool = False,
        s: float = None,
        m: int = None,
        dropout: float = None,
    ) -> None:
        super(GraphConv, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p = dropout)
            self.dropout2d = nn.Dropout2d(p = dropout)
        else: self.dropout = None
        self.laplacian = laplacian
        self.mask = mask
        self.mask_weighting = mask_weighting
        self.mask_power = mask_power
        self.standard_gcn = standard_gcn
        self.eye = None
        if not self.laplacian:
            assert (in_features is not None) and (out_features is not None), "Requires in_features and out_features arguments."

            self.t_s = nn.Parameter(torch.empty((attention_features, out_features)), requires_grad=True)
            self.t_r = nn.Parameter(torch.empty((attention_features, out_features)), requires_grad=True)
            self.fc = nn.Linear(in_features = in_features, out_features = out_features, bias = bias)
            self.fc.bias.data.fill_(0.)
            nn.init.kaiming_normal_(self.fc.weight)

            self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)
                self.bias.data.fill_(0.)
            else:
                self.bias = None
        else:
            assert (s is not None) and (m is not None), "Parameters s and m must be given for Laplacian GraphConv."
            self.s = s
            self.m = m

    def get_mask(self, graph, power = 1):
        graph = normalise_graph(graph, remove_diagonal = True, self_connection = False)
        mask = addit = graph
        for _ in range(power - 1):
            if graph.is_sparse:
                addit = torch.sparse.mm(
                    addit,
                    graph
                )
            else:
                addit = addit @ graph
            mask = mask + addit
            
        if graph.is_sparse:
            return normalise_graph(
                mask,
                remove_diagonal=False,
                self_connection=False,
                precomputed_sparse_identity_matrix=self.eye
            ).coalesce()
        else:
            return normalise_graph(
                mask,
                remove_diagonal=False,
                self_connection=False,
            )


    @staticmethod
    def mask_graph(A, mask, add_mask_to_A = False, fill_value = 1e-9, mask_weighting = False, dense_A = False):
        if dense_A and mask.is_sparse: A = A.to_sparse()
        if add_mask_to_A:
            A = A + fill_value * mask.to(bool)

        if mask.is_sparse: A = A.to_dense().sparse_mask(mask)

        if mask_weighting:
            A = A * mask
        elif not mask.is_sparse:
            A[~mask.to(bool)] = 0.
        
        if (not dense_A) or (not mask.is_sparse):
            return A
        else:
            return A.to_dense()


    def neighbourhood_softmax(self, graph, mask=None):
        if mask is not None: graph[~mask] = -10e10 # remove points outside neighbourhood
        return F.softmax(graph, dim=1)


    def get_edge_index_attention(self, H, graph, mask = None, sparse = True):
        C1 = F.leaky_relu(torch.matmul(H, self.t_s.t()), negative_slope=.2)
        C2 = F.leaky_relu(torch.matmul(H, self.t_r.t()).t(), negative_slope=.2)
        if sparse:
            C1, C2 = C1.to_sparse(), C2.to_sparse()
            S = torch.sparse.mm(C1, C2)
            if mask is not None:
                S = self.mask_graph(S, mask, add_mask_to_A = True, mask_weighting = self.mask_weighting)
            return torch.sparse.softmax(S, dim=1)
        else:
            eye = torch.eye(H.shape[0], device=H.device)
            if C1.shape[1] == 1:
                if self.mask_weighting:
                    S = ((mask * C1) + (mask * C2)) / 2
                else:
                    S = (C1 + C2) / 2
                if mask is not None:
                    if mask.is_sparse: S[mask.to_dense() <= 0] = -1e11
                    else: S[mask <= 0] = -1e11
                out = F.softmax(S, dim = 1)
                if self.dropout is not None:
                    out = self.dropout2d(out[None,:,:])[0]
                return out
            else:
                # Multi-head attention
                S = []
                for _ in range(C1.shape[1]):
                    if self.mask_weighting:
                        S_ = ((mask * C1[:,_:_+1]) + (mask * C2[_:_+1,:])) / 2
                    else:
                        S_ = (C1[:,_:_+1] + C2[_:_+1,:]) / 2
                    if mask.is_sparse: S_[~mask.to_dense().to(bool)] = -1e11
                    else: S_[~mask.to(bool)] = -1e11
                    S.append(S_)
                out = [F.softmax(_, dim=1) for _ in S]
                if self.dropout is not None:
                    for i in range(len(out)):
                        out[i] = self.dropout2d(out[i][None,:,:])[0]
                return out


    def _weighted_graph_conv_forward(self, input, graph = None, C = None, backward = False, return_attention = False):
        if backward is False:
            H_ = torch.matmul(input, self.weight.t())
        else:
            if self.bias is not None:
                input = input - self.bias[None, :]
            H_ = torch.matmul(input, self.weight)
        if (self.bias is not None) and (not backward):
            H_ = H_ + self.bias[None, :]
        if self.dropout is not None:
            H_ = self.dropout(H_)

        if (graph is not None) and (C is None):
            if (self.eye is None) or (self.eye.shape != graph.shape):
                self.eye = get_eye(graph.shape[0], graph.device, sparse=graph.is_sparse)

            if self.mask:
                mask = self.get_mask(graph, power = self.mask_power)
            else:
                mask = None

            if self.standard_gcn:
                if self.mask:
                    graph = self.mask_graph(
                        graph,
                        mask,
                        add_mask_to_A = True,
                        mask_weighting = self.mask_weighting,
                        dense_A = not graph.is_sparse
                    )

                if type(graph) == torch.Tensor:
                    C = normalise_graph(
                        graph,
                        remove_diagonal=True,
                        self_connection=True,
                        precomputed_sparse_identity_matrix = self.eye
                    )
                elif type(graph) == list:
                    C = [normalise_graph(
                        _,
                        remove_diagonal=True,
                        self_connection=True,
                        precomputed_sparse_identity_matrix = self.eye
                    ) for _ in graph]
            elif not self.standard_gcn:
                C = self.get_edge_index_attention(H_, graph, mask=mask, sparse=graph.is_sparse)
                if type(C) == torch.Tensor:
                    C = normalise_graph(
                        C,
                        remove_diagonal=False,
                        self_connection=True,
                        precomputed_sparse_identity_matrix = self.eye
                    )
                elif type(C) == list:
                    C = [normalise_graph(
                        _,
                        remove_diagonal=False,
                        self_connection=True,
                        precomputed_sparse_identity_matrix = self.eye
                    ) for _ in C]

        if (type(C) == torch.Tensor):
            if C.is_sparse: out = torch.sparse.mm(C, H_)
            else: out = torch.matmul(C, H_)
        elif type(C) == list:
            out = torch.mean(torch.stack([torch.matmul(_, H_) for _ in C], dim=-1), dim=-1)
        else:
            raise TypeError

        if not return_attention:
            return out
        else:
            return out, C


    def _laplacian_graph_conv_forward(self, input, graph, **_):
        L = graph_laplacian(graph)
        C = torch.matrix_power(torch.eye(L.shape[0], device=L.device) + self.s*L, self.m)

        return torch.matmul(C, input)


    def forward(
        self,
        input: torch.Tensor,
        graph: torch.Tensor = None,
        C: torch.Tensor = None,
        backward: bool = False,
        return_attention: bool = False,
    ):
        return self._weighted_graph_conv_forward(input, graph = graph, C = C, backward = backward, return_attention = return_attention)
