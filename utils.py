import torch
import os
import ot
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_completeness_v_measure, pairwise
from torch import nn
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, mutual_info_score, rand_score
from sklearn.metrics.cluster import contingency_matrix
# 用KNN方法构建图，loop为True表示KNN包括节点本身，cosine为True表示利用余弦作为距离指标
def build_graph(spatial, k_nearest=7, loop=True, cosine=False, rad_distance=150, mode='Radius'):
    edge_index = None
    if mode == 'Radius':
        edge_index = radius_graph(spatial, rad_distance, loop=loop)
    elif mode == 'KNN':
        # 通过knn创建图
        # loop添加自环，cosine使用余弦距离
        edge_index = knn_graph(spatial, k=k_nearest, loop=loop, cosine=cosine)
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.ones(edge_index.size(1))
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=torch.Size([num_nodes, num_nodes]))
    # 确保对称, 且边的权重是1，
    adj = adj + adj.transpose(0, 1)
    adj = adj.coalesce()
    mask = adj.values() > 1
    adj.values()[mask] = 1
    return adj


# 对空间位置的邻接矩阵按照特征距离阈值选择是否包含这个边
def adjust_adjacency_matrix(adj, feat, threshold):
    # 计算节点特征之间的欧式距离
    distance = torch.cdist(feat, feat)
    new_adj = adj * (distance < threshold)
    return new_adj


# 计算稀疏邻居矩阵中的平均邻居数
def get_average_neighor(adj):
    node_num = adj.shape[0]
    degree = torch.sparse.sum(adj, dim=1)
    average_degree = torch.sparse.sum(degree) / node_num
    return average_degree


# 标准化图
def normalize_adj(adj, mode='sparse'):
    if mode == 'dense':
        # 求度
        degree = adj.sum(dim=1)
        # 计算D^-0.5
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        # 计算对称归一化的邻接矩阵
        adj_norm = torch.mm(degree_inv_sqrt, torch.mm(adj, degree_inv_sqrt))
    elif mode == 'sparse':
        adj = adj.coalesce()
        inv_sqrt_degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense(), -0.5)
        D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
        new_values = adj.values() * D_value
        adj_norm = torch.sparse_coo_tensor(adj.indices(), new_values, adj.size(), dtype=torch.float)
    else:
        adj_norm = adj
    return adj_norm


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    y_true = y_true.astype(np.int64)
    y_pred = y_predicted.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    w_sum = 0
    for x in range(cluster_number):
        i = ind[0][x]
        j = ind[1][x]
        w_sum += w[i, j]
    return (w_sum * 1.0) / y_pred.size


def nmi(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    nmi_score = normalized_mutual_info_score(y_pred, y_true)
    return nmi_score


# 归一化特征 所有样本同个特征归一化   min-max 0-1然年变成-1到1
def normalized_feat(feat):
    minVals = torch.min(feat, dim=0)[0]
    maxVals = torch.max(feat, dim=0)[0]
    epsilon = 1e-8
    norm_feat = (feat - minVals.unsqueeze(0)) / (maxVals - minVals + epsilon).unsqueeze(0)
    # norm_feat = (norm_feat - 0.5) * 2
    return norm_feat


# 将获取最近的距离
def get_k_nearest_distance_ind_sigma(x, k):
    distance = torch.cdist(x, x)  # 计算成对距离
    # 获取sigma
    sigma = torch.mean(torch.mean(distance, dim=0))
    if sigma == 0.:
        sigma = 1e-6
    # 对每行进行排序，获取前k+1个最小距离的索引(包括自己)
    values, indices = distance.topk(k, dim=1, largest=False)
    return values, indices, sigma


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm="mclust", random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def mclust_R2(embedding, num_cluster, modelNames='EEE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    mclust_res = mclust_res.astype('int')
    # mclust_res = mclust_res.astype('category')
    return mclust_res


def refine_label(adata, radius=50, key='mclust'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    new_type = np.array(new_type)
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def loss_amse(recons, ori):
    SE_keepdim = nn.MSELoss(reduction="none")

    Gamma = ori.data.sign().absolute()
    Q = Gamma.mean(dim=1)
    Gamma = Gamma + (Gamma - 1).absolute() * Q.reshape(-1, 1)

    loss = SE_keepdim(ori, recons) * Gamma

    return loss.mean()

def search_res(adata, n_clusters, method='leiden', use_rep='embedding', start=0.1, end=3.0, increment=0.01):  # 有ground truth
    print('Searching resolution...')
    label = 0
    best_ari, best_res = 0.0, 0.0
    sc.pp.neighbors(adata, n_neighbors=15, use_rep=use_rep)  # 注意邻居
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=42, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity = calculate_clustering_metrics(adata.obs['leiden'], adata.obs['Ground Truth'])
            if ARI > best_ari and count_unique == n_clusters:  # 去掉极端稀少的类
                best_ari = ARI
                best_res = res
            print('resolution={}, cluster number={}, ARI={}, AMI={} NMI={}, Homogeneity={} completeness={} V_measure={} purity={}'.format(res, count_unique, ARI, AMI, NMI, Homogeneity, completeness, V_measure, purity))
            # print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            # break
    print("best_ari:{} best_res:{}".format(best_ari, best_res))
    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!"

    return best_res

def calculate_clustering_metrics(y_true, y_pred):
    MI = mutual_info_score(y_true, y_pred)
    NMI = normalized_mutual_info_score(y_true, y_pred)
    AMI = adjusted_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    RI = rand_score(y_true, y_pred)
    ARI = adjusted_rand_score(y_true, y_pred)
    Homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    V_measure = v_measure_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)

    return MI, NMI, AMI, RI, ARI, Homogeneity, completeness, V_measure, purity

def purity_score(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
    return purity