import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, minmax_scale
import scanpy as sc
import bbknn
from anndata import AnnData
import copy
import argparse
import os
import h5py
from scipy.sparse import csr_matrix


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
    rows=graph_mtx.row
    cols=graph_mtx.col
    ratio_value = graph_mtx.data

    f = open("../data/" + datastr_name + ".txt", 'w')
    f.write('{}\n'.format(cell_count))

    inter_edge_num = 0
    inter_ratio = []
    for index, value in enumerate(rows):
        if value < index1 and cols[index] < index1:
           f.write('{} {}\n'.format(value, cols[index]))
        elif value > index1 and cols[index] > index1:
            f.write('{} {}\n'.format(value, cols[index]))
        else:
            inter_ratio.append(ratio_value[index])
            if ratio_value[index]>0.4: # trim the edges between datasets
                f.write('{} {}\n'.format(value, cols[index]))
                inter_edge_num += 1

    for index in range(cell_count):
        f.write('{} {}\n'.format(index, index))
    f.close()


    # To make BBKNN compatible with different operating systems
    times = 3
    if inter_edge_num < max(index1, index2):
        f = open("../data/" + datastr_name + ".txt", 'w')
        f.write('{}\n'.format(cell_count))

        inter_ratio.sort(reverse=True)
        ratio = inter_ratio[min(times * max(index1, index2), len(inter_ratio) - 1)]

        for index, value in enumerate(rows):
            if value < index1 and cols[index] < index1:
                f.write('{} {}\n'.format(value, cols[index]))
            elif value > index1 and cols[index] > index1:
                f.write('{} {}\n'.format(value, cols[index]))
            else:
                if ratio_value[index] > ratio:  # trim the edges between datasets
                    f.write('{} {}\n'.format(value, cols[index]))

        for index in range(cell_count):
            f.write('{} {}\n'.format(index, index))
        f.close()




def read_from_h5(data_dir, dataname, ref_or_query_name, number_cells, gene_numbers):

    f = h5py.File(os.path.join(data_dir, ref_or_query_name), 'r')
    keys = list(f.keys())
    keys = f[dataname]
    k2 = [x for x in keys if x not in ['gene_names', 'cell_names']]
    indptr = np.array(f[dataname]['indptr'])
    indices = np.array(f[dataname]['indices'])
    results = np.array(f[dataname]['data'])
    results = sc.AnnData(X=csr_matrix((results, indices, indptr), shape=(number_cells, gene_numbers)).toarray()).X.T
    cell_names=np.array(f[dataname]['barcodes']).astype(str)
    gene_names = np.array(f[dataname]['gene']).astype(str)
    results=pd.DataFrame(results, index=list(gene_names), columns=list(cell_names))
    return results



def convert_data_graph_construction(name):

    data_name1="norm_data_1.csv"
    data_name2 = "norm_data_2.csv"
    data_path = "../process_data/" + name + "/norm_data/"
    #train_data = pd.read_csv(data_path + data_name1, header=0, index_col=0)
    #test_data = pd.read_csv(data_path + data_name2, header=0, index_col=0)
    train_data = read_from_h5(data_path, name, "reference_data_1.h5", 691600, 2000)
    test_data = read_from_h5(data_path, name, "reference_data_2.h5", 141606, 2000)

    data = np.hstack((train_data.values, test_data.values))

    data_value = data.T
    data_value=data_value.astype(np.float64)
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



