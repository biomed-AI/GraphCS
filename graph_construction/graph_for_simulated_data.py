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



def check_graph_error_rate(datastr, batch_info):
    error_count=0
    batch_num=len(batch_info)
    batch=np.concatenate([[i]*batch_info[i] for i in range(batch_num)],axis=0)
    label1 = pd.read_csv('../data/' + datastr + "/Label1.csv").values.flatten()
    label2 = pd.read_csv('../data/' + datastr + "/Label2.csv").values.flatten()
    labels=np.concatenate((label1, label2), axis=0)
    labels = pd.DataFrame(labels)
    label_types=np.unique(labels).tolist()
    rename = {label_types[i]:i for i in range(len(label_types))}
    labels = labels.replace(rename).values.flatten()
    f=lambda x:(x*(x+1)//2)
    batch_error_count=[0 for j in range(f(batch_num))]
    batch_count=[0 for j in range(f(batch_num))]
    graph_path="../data/"+datastr+".txt"
    data=pd.read_table(graph_path).values
    total_count = data.shape[0]
    for value in data:
        edge_con=True
        split_value= value[0].split(' ')
        index1, index2=int(split_value[0]), int(split_value[1])
        x=batch[index1]
        y=batch[index2]
        x,y=max(x,y),min(x,y)
        if labels[index1]!=labels[index2]:
            error_count+=1
            edge_con=False
        batch_count[f(x)+y]+=1
        if not edge_con:
            batch_error_count[f(x)+y]+=1
    file=open(datastr+'_graph1_error.txt','w+')
    file.write("graph error: %.4f\n"%(error_count/total_count))
    for i in range(batch_num):
        for j in range(i+1):
            file.write('The edge count between batch%d and batch%d is %d\n'%(i+1,j+1,batch_count[f(i)+j]))
            file.write('The edge error rate between batch%d and batch%d is %.4f\n'%(i+1,j+1,batch_error_count[f(i)+j]/batch_count[f(i)+j]))
    file.close()




def bbknn_construct_graph(datastr_name, data, batch_info,edge_ratio):
    sc.settings.verbosity = 3
    batch_num=len(batch_info)
    batch_labels=np.concatenate([['batch%d'%(i+1)]*batch_info[i] for i in range(batch_num)],axis=0)
    formatting = AnnData(data)
    formatting.obs["batch"] = batch_labels
    adata=formatting
    sc.tl.pca(adata)
    bbknn.bbknn(adata)

    sc.tl.leiden(adata, resolution=0.4) 
    # resolution: A parameter value controlling the coarseness of the clustering.
    # Higher values lead to more clusters
    bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['leiden'])
    sc.pp.pca(adata)
    bbknn.bbknn(adata, batch_key='batch') 

    cell_count=adata.obsp['connectivities'].shape[0]
    graph_mtx=adata.obsp['connectivities'].tocoo()
    rows=graph_mtx.row
    cols=graph_mtx.col
    ratio_value = graph_mtx.data
    batch=np.concatenate([[i]*batch_info[i] for i in range(batch_num)],axis=0)

    inter_ratio=[[[] for i in range(batch_num)] for j in range(batch_num)]
    inter_num=0
    outer_num=0
    for i in range(len(rows)):
        if batch[rows[i]]!=batch[cols[i]]:
            x=batch[rows[i]]
            y=batch[cols[i]]
            x,y=max(x,y),min(x,y)
            inter_ratio[x][y].append(ratio_value[i])

    times=edge_ratio
    for i in range(batch_num):
        for j in range(batch_num):
            if len(inter_ratio[i][j])>0:
                inter_ratio[i][j].sort(reverse=True)

    f = open("../data/" + datastr_name + ".txt", 'w')
    f.write('{}\n'.format(cell_count))
    print("total edges: ", len(inter_ratio[1][0]))
    print("edge_ratio: ", edge_ratio) 

    for i in range(len(rows)):
        if batch[rows[i]]==batch[cols[i]]:
            f.write('{} {}\n'.format(rows[i], cols[i]))
        else:
            x=batch[rows[i]]
            y=batch[cols[i]]
            x,y=max(x,y),min(x,y)
            ratio=inter_ratio[x][y][min(int(times*max(batch_info[x], batch_info[y])), len(inter_ratio[x][y])-1)]
            if ratio_value[i] > ratio:  # trim the edges between batches
                f.write('{} {}\n'.format(rows[i], cols[i]))

    for index in range(cell_count):
        f.write('{} {}\n'.format(index, index))
    f.close()
    #check_graph_error_rate(datastr_name,batch_info)


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



def convert_data_graph_construction(name, edge_ratio):
 
    data_name1="tpm_data_1.csv"
    data_name2 = "tpm_data_2.csv"
    data_path = "../process_data/" + name + "/tpm_data/"
    train_data = pd.read_csv(data_path + data_name1, header=0, index_col=0)
    test_data = pd.read_csv(data_path + data_name2, header=0, index_col=0)

    hvg_path="../process_data/" + name+"/sel_features.csv"
    hvg_features=pd.read_csv(hvg_path, header=0).values.flatten()
    train_data=train_data.loc[hvg_features, :]
    test_data = test_data.loc[hvg_features,:]

    data = np.hstack((train_data.values, test_data.values))
    data = data.astype(np.float64)
    data_value = data.T

    data_value_for_graph=copy.deepcopy(data_value)
    minmax_scale(data_value, feature_range=(0, 1), axis=0, copy=False)
    save_path = "../data/" + name + "_feat"
    np.save(save_path, data_value)

    # construct graph using BBKNN
    # Batch information is saved in the pre-processing phase
    f=open("../data/" + name + "/batch_info.txt",'r')
    batch_info=list(f)[0].split(' ')[:-1]
    f.close()
    batch_info=list(map(int,batch_info))
    bbknn_construct_graph(name, data_value, batch_info, edge_ratio)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="example", help='Data_name.')
    # edge_ratio=100 represents not trimming inter-edges
    parser.add_argument('--edge_ratio', type=float, default=100, help='Parameters for controlling the number of inter-edges between batches.')
    args = parser.parse_args()

    convert_data_graph_construction(args.name,args.edge_ratio)




