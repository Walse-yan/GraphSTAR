# GraphSTAR:Joint Spatial and Feature-Aware Graphs for Spatial Transcriptomics Analysis in Varying Resolutions

## data origin
* DLPFC:https://drive.google.com/drive/folders/10lhz5VY7YfvHrtV40MwaqLmWz56U9eBP
* CBMSTA:https://db.cngb.org/stomics/cbmsta/
* MERFISH and osmFish:http://sdmbench.drai.cn/

## Requiremts
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch==2.2.2
* cudnn=12.1
* numpy==1.24.3
* scanpy==1.9.8
* anndata=0.9.2
* rpy2==3.5.10
* R=4.4.1

## Parameters and Training Details
This section discusses the parameters and training details of GraphSTAR. The KNN parameter $k_1$ for spatial neighbor construction is generally set to the number of neighbors around the anchor point, while the KNN parameter $k_2$ for feature-aware graph is generally assigned the same value as $k_1$ or slightly larger. The hyperparameter $\beta$ corresponding to feature graph construction, is set to a default value of 1.0. The hyperparameter $\alpha$ which controls the weights of the spatial graph and dynamic feature graph, is defined within a range of 0.1 to 2.0. We will further analyze the influence of the $\alpha$ parameter on model performance in the experimental section. The weight $\lambda$ for the orthogonal loss function is set to a default value of $e^{-5}$, facilitating a balance between reconstruction loss and orthogonal loss. In the autoencoder, the embedding dimension used for clustering is set to 50 by default. GraphSTAR employs the Adam optimizer for model parameter optimization, with a default learning rate $e^{-3}$ of and a default weight decay of $e^{-5}$.
