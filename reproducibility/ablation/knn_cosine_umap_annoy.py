import scanpy as sc
import bbknn
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex
import time
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.decomposition import PCA


def print_time(f):
    def fi(*args, **kwargs):
        s = time.time()
        res = f(*args, **kwargs)
        print('--> RUN TIME: <%s> : %s' % (f.__name__, time.time() - s))
        return res

    return fi




@print_time
def annoy_construt_graph(datastr_name, features, topk=5, tree_num=10):
    print("over read data \n")
    f = features.shape[1]
    t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
    for index, value in enumerate(features):
        t.add_item(index, value)
    t.build(tree_num)
    u = t
    n = features.shape[0]
    f = open("./data/" + datastr_name + "_annoy.txt", 'w')
    f.write('{} \n'.format(n))
    print("begin to write********** \n")
    for index in range(n):
        nodes = u.get_nns_by_item(index, topk)
        for near_node in nodes:
            f.write('{} {}\n'.format(index, near_node))
    f.close()


@print_time
def cosine_construct_graph(datastr_name, features, topk=5):

    dist = cosine_similarity(features, features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open("./data/" + datastr_name + "_cosine.txt", 'w')
    f.write('{} \n'.format(features.shape[0]))
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


@print_time
def umap_construct_graph(datastr_name, features_tmp, label1, label2):

    index1 = len(label1)
    index2 = len(label2)
    batch1 = ['batch1'] * index1
    batch2 = ['batch2'] * index2
    batch_labels = np.concatenate((batch1, batch2), axis=0)
    pca = PCA(n_components=20)
    features_tmp = pca.fit_transform(features_tmp)

    formatting = AnnData(features_tmp)
    formatting.obs["batch"] = batch_labels

    sc.pp.neighbors(formatting, n_neighbors=15, use_rep='X')
    sc.tl.umap(formatting)
    connectivities=formatting.obsp['connectivities'].toarray()
    save_graph_data(connectivities, datastr_name, "umap")



@print_time
def knn_construct_graph(datastr_name, features):

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features)
    #distances, indices = nbrs.kneighbors(features)
    graph_matrix=nbrs.kneighbors_graph(features).toarray()
    save_graph_data(graph_matrix, datastr_name, "sklearnknn")


def save_graph_data(matrix, data_name, method_name):
    n_count=matrix.shape[0]
    f = open("./data/"+data_name+"_"+method_name+".txt", 'w')
    f.write('{}\n'.format(n_count))
    adj = defaultdict(list)
    for i, row in enumerate(matrix):
        for j, adjacent in enumerate(row):
            if adjacent:
                adj[i].append(j)
                f.write('{} {}\n'.format(i, j))
        if adj[i].__len__ == 0:
            adj[i] = []
    f.close()
    return adj


if __name__=="__main__":

    data_base = "../../data/"

    dataname = [


    'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human',
'Baron_human_Baron_mouse'

    ]

    for index, datastr_name in enumerate(dataname):

        features = np.load(data_base + datastr_name + "_feat.npy")
        batch1_label = data_base + datastr_name + "/Label1.csv"
        batch2_label = data_base + datastr_name + "/Label2.csv"
        label1 = pd.read_csv(batch1_label).values.flatten()
        label2 = pd.read_csv(batch2_label).values.flatten()


        annoy_construt_graph(datastr_name, features)
        cosine_construct_graph(datastr_name, features, topk=5)
        umap_construct_graph(datastr_name, features, label1, label2)
        knn_construct_graph(datastr_name, features)
