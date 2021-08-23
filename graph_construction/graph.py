import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale
import scanpy as sc
import bbknn
from anndata import AnnData
import copy
import argparse
import os


def bbknn_construct_graph(datastr_name, data, index1, index2):
    sc.settings.verbosity = 3

    batch1 = ['batch1'] * index1
    batch2 = ['batch2'] * index2
    batch_labels = np.concatenate((batch1, batch2), axis=0)

    formatting = AnnData(data)
    formatting.obs["batch"] = batch_labels
    adata=formatting
    sc.tl.pca(adata)
    bbknn.bbknn(adata)

    sc.tl.leiden(adata, resolution=0.4) # resolution=0.4
    # resolution: A parameter value controlling the coarseness of the clustering.
    # Higher values lead to more clusters
    bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['leiden'])
    sc.pp.pca(adata)
    bbknn.bbknn(adata, batch_key='batch')# ,trim =20

    cell_count=adata.obsp['connectivities'].shape[0]
    graph_mtx=adata.obsp['connectivities'].tocoo()
    f = open("../data/" + datastr_name + ".txt", 'w')
    f.write('{}\n'.format(cell_count))
    rows=graph_mtx.row
    cols=graph_mtx.col
    ratio_value = graph_mtx.data

    for index, value in enumerate(rows):
        if value < index1 and cols[index] < index1:
           f.write('{} {}\n'.format(value, cols[index]))
        elif value > index1 and cols[index] > index1:
            f.write('{} {}\n'.format(value, cols[index]))
        else:
            if ratio_value[index]>0.4: # trim the edges between datasets
                f.write('{} {}\n'.format(value, cols[index]))

    for index in range(cell_count):
        f.write('{} {}\n'.format(index, index))
    f.close()


def convert_data_graph_construction(name):

    data_name1="norm_data_1.csv"
    data_name2 = "norm_data_2.csv"
    data_path = "../process_data/" + name + "/norm_data/"
    train_data = pd.read_csv(data_path + data_name1, header=0, index_col=0)
    test_data = pd.read_csv(data_path + data_name2, header=0, index_col=0)
    data = np.hstack((train_data.values, test_data.values))

    data_value = data.T
    data_value_for_graph=copy.deepcopy(data_value)
    minmax_scale(data_value, feature_range=(0, 1), axis=0, copy=False)
    save_path = "../data/" + name + "_feat"
    np.save(save_path, data_value)

    # construct graph using BBKNN
    batch1_label = "../data/" + name + "/Label1.csv"
    batch2_label = "../data/" + name + "/Label2.csv"
    index1 = len(pd.read_csv(batch1_label).values.flatten())
    index2 = len(pd.read_csv(batch2_label).values.flatten())
    bbknn_construct_graph(name, data_value_for_graph, index1, index2)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="example", help='use normalized.')
    args = parser.parse_args()

    convert_data_graph_construction(args.name)



