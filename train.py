import os
import torch
import torch.nn as nn
from tqdm import tqdm
import GraphSTAR as gr
from process import get_feature, get_feature2, get_adj, get_label, plot_history
from utils import (normalize_adj, normalized_feat, mclust_R2, mclust_R)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score


def train(adata,                    # 要求有adata.X，标签放在label中，预测结果放在pred中，没有adj就构建
          data_name,                # 数据集名称，如果没有这个数据集，请自行添加代码
          num_class,                # 类别数
          mode='Radius',            # 空间邻接矩阵的构建方式
          start_k_neighbor=7,       # 构建空间近邻的k值
          rad_distance=150,         # 构建空间近邻的radius值
          norm=True,                # 是否对adata.X 特征进行归一化(0-1范围)
          hidden_dim=50,            # 模型的隐藏层维度，用于聚类的embedding
          mid_k_neighbor_list=[7],  # 模型训练时用特征构建的M近邻数
          alpha_list=[0.5],         # alpha_list的个数取决于block的块数
          beta_list=[1.0],          # 同alpha_list
          mid_feature_dims=[],      # 中间层的维度，里面的个数是block个数-1
          is_learned=False,         # 是否学习alpha和beta
          lr=1e-3,                  # 学习率
          reg=1e-5,                 # 正交正则化损失的权重值
          loss_threshold=1e-5,      # 训练的loss阈值，小于这个值就停止训练
          weight_decay=1e-5,        # 权重衰减
          epochs=1000,              # 训练的epoch数
          SEED=42,                  # 随机种子
          file_name='151507',       # 一般是切片的名称，用于保存各种结果文件
          is_record=False,          # 是否记录训练过程中的nmi,ari,ami
          cluster_stride=50,        # 如果记录，每隔多少个epoch进行一次聚类
          refinement_radius=50,     # refinement的半径
          is_refinement=False,      # 是否进行refinement
      #   is_cellCluster=False,     # 是否进行细胞聚类
          device='cpu'):
    # 设置随机种子
    import random
    import numpy as np
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    last_model_root = './res/{}/model/last'.format(data_name)
    features = None
    adj_norm = None
    # 如果adata中有adj，就直接用，没有就构建
    if 'adj' in adata.uns.keys():
        # 判断adj类型是numpy还是torch,如果是numpy就转换成torch
        adj = adata.uns['adj']
        if isinstance(adj, np.ndarray):
            adj = torch.tensor(adata.uns['adj'])
        # 标准化
        if adj.is_sparse:
            adj_norm = normalize_adj(adj).to(device)
        else:
            adj_norm = normalize_adj(adj, mode='dense').to_sparse().to(device)
    elif 'adj_norm' in adata.uns.keys():
        adj_norm = adata.uns['adj_norm']
        if isinstance(adj_norm, np.ndarray):
            adj_norm = torch.tensor(adj_norm)
        if not adj_norm.is_sparse:
            adj_norm = adj_norm.to_sparse()
        adj_norm = adj_norm.to(device)
    else:
        # 保存邻接矩阵到adata中
        adj = get_adj(adata, mode=mode, k_nearest=start_k_neighbor, rad_distance=rad_distance, is_norm=False)
        adata.uns['adj'] = adj.to_dense().numpy()
        adj_norm = get_adj(adata, mode=mode, k_nearest=start_k_neighbor, rad_distance=rad_distance).to(device)
        adata.uns['adj_norm'] = adj_norm.cpu().to_dense().numpy()
    features = get_feature(adata).to(device) 
    labels = get_label(adata)

    if norm:
        features_norm = normalized_feat(features)
    else:
        features_norm = features
    input_dim = features_norm.shape[1]
    model = gr.GraphSTAR(input_dim, hidden_dim, mid_k_neighbor_list, alpha_list, beta_list, mid_feature_dims, is_learned=is_learned).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = {"loss": [], "nmi": [], "ami": [], "ari": []}
    # 训练
    model.train()
    print("begin training")
    for epoch in tqdm(range(1, epochs + 1)):
        optimizer.zero_grad()
        # 获取预测后的邻接矩阵
        out, embedding, new_adj = model(features_norm, adj_norm)
        loss = criterion(out, features_norm)
        orth_loss = torch.zeros(1, device=out.device)
        for name, param in model.named_parameters():
            if 'encoder' in name and 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat.T, param_flat)
                sym -= torch.eye(param_flat.shape[1]).to(param.device)
                orth_loss = orth_loss + (reg * sym.abs().sum())
        loss = loss + orth_loss
        # print(f"\nEpoch: {epoch}\n----------")
        # print(f"Train loss: {loss.item():.5f}")

        history["loss"].append(loss.item())
        if is_record and epoch % cluster_stride == 0:   
            y_pred_init = mclust_R2(embedding.cpu().detach().numpy(), num_cluster=num_class)
            y_pred_init = y_pred_init - 1
            nmi_val = normalized_mutual_info_score(labels.numpy(), y_pred_init)
            ari_val = adjusted_rand_score(labels.numpy(), y_pred_init)
            ami_val = adjusted_mutual_info_score(labels.numpy(), y_pred_init)
            history['nmi'].append(nmi_val)
            history['ari'].append(ari_val)
            history['ami'].append(ami_val)

            adata_temp = adata.copy()
            adata_temp.obsm['embedding'] = embedding.cpu().detach().numpy()
            adata_temp.obs['predict'] = y_pred_init
            adata_temp.obs['predict'] = adata_temp.obs['predict'].astype('category')
            # adata_list.append(adata_temp)

        if loss.item() < loss_threshold:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    # 保存最后一个model参数
    last_model_full_path = os.path.join(last_model_root, 'slice_{} last_model.pth'.format(file_name))
    # 如果last_model_root不存在，则创建
    if not os.path.exists(last_model_root):
        os.makedirs(last_model_root)
    torch.save(model.state_dict(), last_model_full_path)
    if is_record:
        print("\n")
        print("训练过程中最大的nmi值是", max(history['nmi']))
        print("训练过程中最大的ari值是", max(history['ari']))
        print("训练过程中最大的ami值是", max(history['ami']))

        print('\n')
        print("最后一个epoch的ari值是", history['ari'][len(history['ari']) - 1])
        print("最后一个epoch的ami值是", history['ami'][len(history['ami']) - 1])

    # 测试
    model.eval()
    _, last_embedding, new_adj = model(features_norm, adj_norm)
    # 保存学习后的graph
    adata.uns['new_adj'] = new_adj.cpu().detach().to_dense().numpy()
    adata.obsm['embedding'] = last_embedding.cpu().detach().numpy()
    # torch.save(last_embedding.cpu().detach(), 'emb.pth')
    adata = mclust_R(adata, used_obsm='embedding', num_cluster=num_class)
    adata.obs['predict'] = adata.obs['mclust']
    # 计算ARI
    ARI = adjusted_rand_score(adata.obs['predict'], adata.obs['Ground Truth'])
    NMI = normalized_mutual_info_score(adata.obs['predict'], adata.obs['Ground Truth'])
    AMI = adjusted_mutual_info_score(adata.obs['predict'], adata.obs['Ground Truth'])
    metric = {"ARI": ARI, "NMI": NMI, "AMI": AMI}
    adata.uns['metric'] = metric

    if is_record:
        metric = plot_history(adata, num_class, history, "loss when training",
                          file_name=file_name, data_name=data_name, cluster_stride=cluster_stride, radius=refinement_radius, refinement=is_refinement)
    
    
    return adata
