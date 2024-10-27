import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from utils import mclust_R, build_graph, normalize_adj, get_average_neighor, refine_label
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

root_DLPFC = './data/DLPFC'
root_zebrafish = './data/Stereo-seq zebrafish/zf12_stereoseq.h5ad'
root_STARmap = './data/STARmap'
root_osmFish = './data/osmFish'
root_MERFISH = './data/MERFISH'
root_simulation = '../data/simulation'
root_CBMSTA = './data/CBMSTA'

def get_data_DLPFC(section_id):
    input_dir = os.path.join(root_DLPFC, section_id)
    adata = sc.read_visium(path=input_dir, count_file=section_id + '_filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    # 读取标签
    Ann_df = pd.read_csv(os.path.join(root_DLPFC, section_id, section_id + '_truth.txt'), sep='\t', header=None,
                         index_col=0)
    # 将列命名为Ground Truth
    Ann_df.columns = ['region']
    # 把标签添加到adata中
    adata.obs['region'] = Ann_df.loc[adata.obs_names, 'region']
    return adata


def get_data_zebrafish(slice_id):
    adata = sc.read(root_zebrafish)
    slice_adata = adata[adata.obs['slice'] == slice_id]
    # 添加spatial
    spatial_coordinates = np.vstack((slice_adata.obs['spatial_x'], slice_adata.obs['spatial_y'])).T
    slice_adata.obsm['spatial'] = spatial_coordinates
    slice_adata.obs['Ground Truth'] = slice_adata.obs['layer_annotation']
    return slice_adata


def get_data_STARmap():
    input_dir = os.path.join(root_STARmap, 'STARmap_20180505_BY3_1k.h5ad')
    adata = sc.read(input_dir)
    adata.obs['Ground Truth'] = adata.obs['label']
    return adata


def get_data_osmFish():
    input_dir = os.path.join(root_osmFish, 'osmfish.h5ad')
    adata = sc.read(input_dir)
    return adata


def get_data_MERFISH(section_id):
    input_dir = os.path.join(root_MERFISH, 'MERFISH_{}.h5ad'.format(section_id))
    adata = sc.read(input_dir)
    spatial_coordinates = adata.obsm['spatial']
    x_coord = spatial_coordinates[:, 0]
    y_coord = spatial_coordinates[:, 1]
    adata.obsm['spatial'] = np.vstack((x_coord, -y_coord)).T
    return adata

def get_data_simulation():
    input_dir = os.path.join(root_simulation, 'simulation.h5ad')
    adata = sc.read(input_dir)
    adata.obs['Ground Truth'] = adata.obs['label']
    return adata

def get_data_CBMSTA(slice_name):
    input_dir = os.path.join(root_CBMSTA, slice_name)
    adata = sc.read(input_dir)
    adata.obs['Ground Truth'] = adata.obs['annotation']
    return adata

def process_data(adata, data_name):
    if data_name == 'DLPFC':
        # 去除nan标签的数据
        used_adata = adata[~pd.isnull(adata.obs['region'])]
        adata = used_adata
        # 基因过滤，移除少于在min_cells个细胞中表达的基因
        # sc.pp.filter_genes(adata, min_cells=3)
        # 使用Seurat_v3算法识别高度可变基金，保留n_top_genes个高度可变的基因
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        # 进行总数归一化，使得每个细胞的总表达量缩放到1e4
        sc.pp.normalize_total(adata, target_sum=1e4)
        # 对数转换，有助于数据更符合正态分布
        sc.pp.log1p(adata)
        # 选择预处理后的基因数据作为新基因数据
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
            adata = adata_Vars
    elif data_name == 'STARmap':
        # sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.process_STARmap['n_top_genes'])
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
            adata = adata_Vars
    elif data_name == 'osmFish':
        print("osmFish data has been preprocessed!")
    elif data_name == 'MERFish':
        print("MERFish data has been preprocessed!")
    elif data_name == 'Stereoseq ZESTA':
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
            adata = adata_Vars
    elif data_name == 'CBMSTA':
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
            adata = adata_Vars
    else:
        print("请在这里添加预处理步骤！")
    return adata

# 仅把基因表达量作为自身特征
def get_feature(adata):
    # 获取特征数据
    # 如果adata.X的类型是numpy.ndarray
    if type(adata.X) == np.ndarray:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    else:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    features = torch.from_numpy(X.values).float()
    return features


# 获取自身和邻居特征作为自身新的特征
def get_feature2(adata, adj):
    # 获取特征数据
    # 如果adata.X的类型是numpy.ndarray
    if type(adata.X) == np.ndarray:
        X = pd.DataFrame(adata.X[:, ], index=adata.obs.index, columns=adata.var.index)
    else:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    features = torch.from_numpy(X.values)
    num_nodes = adj.shape[0]
    updated_features = torch.zeros_like(features)
    for i in range(num_nodes):
        neighbors = torch.nonzero(adj[i]).squeeze().flatten()
        for neighbor in neighbors:
            updated_features[i] += features[neighbor]
    return updated_features


def get_adj(adata, mode, k_nearest=7, rad_distance=150, is_norm=True):
    # 直接利用KNN建图->sparse图
    adj = build_graph(torch.LongTensor(adata.obsm['spatial']), k_nearest=k_nearest, rad_distance=rad_distance, mode=mode)
    print("空间位置的平均邻居数:", get_average_neighor(adj))
    if is_norm:
        adj_norm = normalize_adj(adj)
    else:
        adj_norm = adj
    return adj_norm


def get_label(adata):
    # 将ground truth转为标签值
    le = LabelEncoder()
    # labels从Layer_1到Layer_6、WM还有NAN表示没有标签
    labels = le.fit_transform(adata.obs['Ground Truth'])
    labels = torch.from_numpy(labels)
    return labels


# 画图并返回指标
def plot_history(adata, 
                 num_class, 
                 history, 
                 title, 
                 file_name, 
                 data_name,      # 数据集的名字，如"DLPFC"
                 cluster_stride, # 训练时每隔多少个epoch进行一次聚类
                 radius=50, 
                 refinement=False, 
                 font_size=14):
    # history如果不是None
    if history is not None:
        # 画出损失图和acc、nmi
        plt.rcParams["figure.figsize"] = (3, 3)
        plt.suptitle(title, fontsize=font_size)
        ax1 = plt.subplot(121)
        ax1.set_title("Loss")
        ax1.plot(history["loss"], label="train")
        plt.xlabel("Epoch")
        ax1.legend()

        ax2 = plt.subplot(122)
        ax2.set_title("nmi and ami and ari")
        ax2.plot(history["nmi"], label="nmi")
        ax2.plot(history["ari"], label="ari")
        ax2.plot(history['ami'], label='ami')
        plt.xlabel("Epoch * {}".format(cluster_stride))
        ax2.legend()

    loss_root = './res/{}/loss'.format(data_name)
    full_path = os.path.join(loss_root, 'slice_{} loss.png'.format(file_name))
    # 如果loss_root路径不存在，则创建
    if not os.path.exists(loss_root):
        os.makedirs(loss_root)
    plt.savefig(full_path)
    plt.show()

    # 聚类过程
    sc.pp.neighbors(adata, use_rep='embedding')
    # 使用UMAP（Uniform Manifold Approximation and Projection）算法对数据进行降维和可视化
    sc.tl.umap(adata)
    
    # 做标签配对，并把结果放入adata中的label_refined
    # if refinement:
    #     new_type = refine_label(adata, radius=radius, key='predict')
    #     adata.obs['predict'] = new_type
    ARI = adata.uns['metric']['ARI']
    NMI = adata.uns['metric']['NMI']
    AMI = adata.uns['metric']['AMI']

    if data_name == 'DLPFC':
        sc.pl.spatial(adata, color=["predict", "Ground Truth"],
                      title=['slice:{} GraphSTAR(ARI={:.4f})'.format(file_name, ARI), "Ground Truth"],
                      save='slice_{}_spatial_plot.png'.format(file_name))
    elif data_name == 'Stereo-seq zebrafish':
        sc.pl.spatial(adata, spot_size=20, color=["predict", "Ground Truth"],
                      title=['slice:{} GraphSTAR(ARI={:.4f})'.format(file_name, ARI), "Ground Truth"],
                      save='slice_{}_spatial_plot.png'.format(file_name))
    elif data_name == 'STARmap':
        sc.pl.embedding(adata, basis='spatial', color=["predict", "Ground Truth"],
                        title=['GraphSTAR(ARI={:.4f})'.format(ARI), "Ground Truth"],
                        save='slice_{}_spatial_plot.png'.format(file_name))
    elif data_name == 'osmFish':
        plt.rcParams["figure.figsize"] = (2, 4)
        sc.pl.embedding(adata, basis='spatial', color=["predict", "Ground Truth"],
                        title=['GraphSTAR(ARI={:.4f})'.format(ARI), "Ground Truth"],
                        save='slice_{}_spatial_plot.png'.format(file_name))
    elif data_name == 'MERFish':
        sc.pl.spatial(adata, color=["predict", "Ground Truth"],
                      title=['slice:{} GraphSTAR(ARI={:.4f})'.format(file_name, ARI), "Ground Truth"],
                      save='slice_{}_spatial_plot.png'.format(file_name), spot_size=20)
    metric = {"ARI": ARI, "AMI": AMI, "NMI": NMI}
    return metric
