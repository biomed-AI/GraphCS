import numpy as np
from sklearn.metrics import f1_score
import gc
import scipy.sparse as sp
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
import scanpy as sc
from anndata import AnnData
import umap
import matplotlib.pyplot as plt

base = "./"

def get_umap_for_data(data_name="example"):

    label = pd.read_csv(base+data_name+"/Label.csv", header=0).values.flatten()
    label[label == 'B_cell'] = 'b_cell'
    label[label == 'T cell'] = 't cell'
    label[label == 't cell'] = 't cell'
    label[label == 'activated_stellate'] = 'stellate'
    label[label == 'quiescent_stellate'] = 'stellate'

    embedding_data = pd.read_csv(base + data_name + "/embedding_data.csv", index_col=0, header=0, sep=',').values

    formatting = AnnData(embedding_data)
    formatting.obs["cell_type"] = label

    split_value= name.split('/')
    save_name=split_value[0]+"_"+split_value[1]
    fontsize=15
    sc.pp.neighbors(formatting, n_neighbors=60, use_rep='X', n_pcs=0)#n_pcs
    sc.tl.umap(formatting)
    sc.pl.umap(formatting, color=["cell_type"],
               legend_fontsize=fontsize,
               #legend_loc='EastOutside',
               #save= save_name + ".png"
               )




names=[

    
    "Raw_data/Baron_mouse_Baron_human",
    "Seurat3/Baron_mouse_Baron_human",
     "scGCN/Baron_mouse_Baron_human",
    "GraphCS/Baron_mouse_Baron_human",
    

    "Raw_data/Baron_mouse_segerstolpe",
    "Seurat3/Baron_mouse_segerstolpe",
     "scGCN/Baron_mouse_segerstolpe",
    "GraphCS/Baron_mouse_segerstolpe",


     "Raw_data/Baron_mouse_combination",
     "Seurat3/Baron_mouse_combination",
   "scGCN/Baron_mouse_combination",
    "GraphCS/Baron_mouse_combination",
  
      ]

for name in names:
    print("data_name: ", name)
    get_umap_for_data(data_name=name)


