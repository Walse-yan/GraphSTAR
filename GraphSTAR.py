import torch.nn as nn
import torch
from utils import normalize_adj, get_k_nearest_distance_ind_sigma
import torch.nn.functional as F
import random

# # DLPFC数据集
# if args.data_name == 'DLPFC':
#     mid_feature1 = 512
# # STARmap数据集
# if args.data_name == 'STARmap':
#     mid_feature1 = 128
# # osmFish数据集
# if args.data_name == 'osmFish':
#     mid_feature1 = 30
# if args.data_name == 'MERFISH':
#     mid_feature1 = 128


def make_positive_definite(matrix, epsilon=1e-4):
    return matrix + epsilon * torch.eye(matrix.shape[0], device=matrix.device)


def safe_cholesky(matrix, epsilon=1e-4):
    try:
        # 有时候matrix的特征值太小是-10^-7的时候，torch.linalg.cholesky(matrix)不会报错，会求出一个nan值
        U = torch.linalg.cholesky(matrix)
        if torch.isnan(U).any():
            U = torch.linalg.cholesky(matrix + 1e-6 * torch.eye(matrix.shape[0], device=matrix.device))
        return U
    except:
        matrix = make_positive_definite(matrix, epsilon)
        return torch.linalg.cholesky(matrix)


def get_M(H, beta, k_neighbor):
    # 创建单位矩阵 I
    identity_matrix = torch.eye(H.shape[1], device=H.device)
    # 求 βI + H^TH
    symmetric_matrix = beta * identity_matrix + torch.mm(H.t(), H)
    # 求symmetric_matrix的实数特征值
    U = safe_cholesky(symmetric_matrix)
    norm_covariance = torch.mm(H, U.inverse().T)
    values, indices, sigma = get_k_nearest_distance_ind_sigma(norm_covariance, k_neighbor)
    kernel_values = torch.exp(-(values * values) / (2.0 * (sigma * sigma)))
    
    # M = torch.zeros((norm_covariance.shape[0], norm_covariance.shape[0]), device=H.device)
    # M.scatter_(1, indices, kernel_values)
    # M_sparse = M.to_sparse()

    # 重写上面的三句代码
    num_nodes = norm_covariance.shape[0]
    # 获取edge_index
    row_indices = torch.arange(num_nodes, device=H.device).repeat_interleave(k_neighbor)
    col_indices = indices.view(-1)
    indices_tensor = torch.stack([row_indices, col_indices], dim=0)
    # 获取对应的values
    kernel_values_tensor = kernel_values.view(-1)
    M_sparse = torch.sparse_coo_tensor(indices_tensor, kernel_values_tensor, (num_nodes, num_nodes), device=H.device)


    M_sparse = (M_sparse + M_sparse.transpose(0, 1)) * 0.5
    return M_sparse


class Block(nn.Module):
    def __init__(self, in_features, out_features, alpha, beta, k_neighbor, is_learned, dropout_num=0.0, is_end=False):
        super(Block, self).__init__()
        self.is_learned = is_learned
        if self.is_learned:
            self.alpha = nn.Parameter(torch.FloatTensor(1))
            self.beta = nn.Parameter(torch.FloatTensor(1))
            self.alpha.data.fill_(alpha)
            self.beta.data.fill_(beta)
        else:
            self.alpha = alpha
            self.beta = beta
        self.is_end = is_end
        self.k_neighbor = k_neighbor
        self.weight1 = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight2 = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.reset_parameters()
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.dropout_num = dropout_num

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        # nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout_num, training=self.training)
        # 求H_1  H_0=XW
        
        # if self.is_learned:
        #     print("alpha = {}, beta = {}".format(self.alpha, self.beta))
        M = get_M(x.detach(), self.beta, self.k_neighbor)
        # M = get_M(x, self.beta, self.k_neighbor)
        new_adj = self.alpha * adj + normalize_adj(M)
        out = torch.sparse.mm(new_adj, x @ self.weight1)
        # out = x @ self.weight1
        out = self.act1(out)
        # 求H_i (i从2到n)
        M_2 = get_M(out.detach(), self.beta, self.k_neighbor)
        # M_2 = get_M(out, self.beta, self.k_neighbor)
        new_adj2 = self.alpha * adj + normalize_adj(M_2)
        self.weight2.data = (self.weight2 + self.weight2.T) * 0.5
        out = torch.sparse.mm(new_adj2, out @ self.weight2)
        # out = out @ self.weight2
        if not self.is_end:
            out = self.act2(out)
            return out
        else:
            return out, new_adj2.data


# 编码器
class Encoder(nn.Module):
    def __init__(self,
                 input_dim,           # 输入的特征维度
                 hidden_dim,          # encoder输出的特征维度
                 k_neighbor_list,     # 构建M时选择的近邻数，一个block一个k
                 alpha_list, 
                 beta_list,
                 is_learned,          # 是否学习alpha和beta
                 mid_feature_dims=[], # 中间层的维度，里面的个数是block个数-1
                 ):
        super(Encoder, self).__init__()
        self.block_num = len(k_neighbor_list)
        if self.block_num != len(alpha_list) or self.block_num != len(beta_list) or self.block_num != len(mid_feature_dims) + 1:
            raise ValueError("The length of k_neighbor_list, alpha_list, beta_list and (mid_features_dims + 1) should be the same!")
        self.layer = nn.Sequential()
        current_input_dim = input_dim
        current_output_dim = mid_feature_dims[0] if len(mid_feature_dims) > 0 else hidden_dim
        is_end = False
        for i in range(self.block_num):
            if i == self.block_num - 1:     #最后一层不加
                is_end = True
            self.layer.add_module('block{}'.format(i), Block(current_input_dim, current_output_dim, alpha_list[i], beta_list[i], k_neighbor_list[i], is_learned=is_learned, is_end=is_end))
            current_input_dim = current_output_dim
            current_output_dim = mid_feature_dims[i + 1] if i + 1 < len(mid_feature_dims) else hidden_dim

        # self.block1 = ourBlock(args.in_features, mid_feature1, alpha_list[0], beta_list[0], k_nearest=mid_k_neighbor_list[0])
        # self.block2 = ourBlock(args.in_features, hidden_dim, alpha_list[0], beta_list[0], k_nearest=mid_k_neighbor_list[0], is_end=True)

    def forward(self, x, adj):
        out = x
        for index, block in enumerate(self.layer):
            if index == self.block_num - 1:
                out, new_adj = block(out, adj)
            else: out = block(out, adj)
        return out, new_adj


class Decoder(nn.Module):
    def __init__(self,
                 input_dim,      # 输入的特征维度
                 output_dim,      # 输出的特征维度
                 mid_feature_dims,    # 中间层的维度
                 ):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential()
        current_input_dim = input_dim
        current_output_dim = mid_feature_dims[0] if len(mid_feature_dims) > 0 else output_dim
        self.layer_num = len(mid_feature_dims) + 1
        for i in range(self.layer_num):
            self.layer.add_module('Linear{}'.format(i), nn.Linear(current_input_dim, current_output_dim))
            if i != self.layer_num - 1:    # 最后一层不加
                self.layer.add_module('Tanh{}'.format(i), nn.Tanh())
            current_input_dim = current_output_dim
            current_output_dim = mid_feature_dims[i + 1] if i + 1 < len(mid_feature_dims) else output_dim
        
        # self.weight1 = nn.Parameter(torch.FloatTensor(input_dim, mid_feature_dims[0]))
        # self.weight2 = nn.Parameter(torch.FloatTensor(mid_feature_dims[0], output_dim))
        # nn.init.xavier_uniform_(self.weight1)
        # nn.init.xavier_uniform_(self.weight2)

    def forward(self, x, adj):
        # out = torch.spmm(adj, x @ self.weight1)
        # out = F.leaky_relu(out)
        # out = torch.spmm(adj, out @ self.weight2)
        
        out = self.layer(x)
        return out


# 自编码器
class GraphSTAR(nn.Module):
    def __init__(self, 
                 input_dim,          # 输入的特征维度
                 hidden_dim,         # encoder之后的维度
                 k_neighbor_list,    # 构建M时选择的近邻数，确保M是稀疏的
                 alpha_list,         # 每一个block包含一个alpha
                 beta_list,          # 每一个block包含一个beta
                 mid_feature_dims=[],   # 中间层的维度，里面的个数是block个数-1
                 is_learned=False   # 是否学习alpha和beta
                 ):         
        super(GraphSTAR, self).__init__()
        print("in_features = {}, mid_feature1 = {}, cluster_features = {}".format(input_dim, "NULL" if len(mid_feature_dims) == 0 else mid_feature_dims, hidden_dim))
        self.encoder = Encoder(input_dim, hidden_dim, k_neighbor_list, alpha_list, beta_list, is_learned, mid_feature_dims)
        # mid_feature_dims翻过来
        if len(mid_feature_dims) == 0:
            decoder_mid_feature_dims = [512]
        else:
            decoder_mid_feature_dims = mid_feature_dims[::-1]
        self.decoder = Decoder(hidden_dim, input_dim, mid_feature_dims=decoder_mid_feature_dims)

    def forward(self, input, adj):
        embedding, new_adj = self.encoder(input, adj)
        return self.decoder(embedding, new_adj), embedding, new_adj
