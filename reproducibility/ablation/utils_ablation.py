import sys
sys.path.append('../../')

import numpy as np
import BiP
import torch
from sklearn.metrics import f1_score
import gc
import os
import scipy.sparse as sp
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, minmax_scale
import sys
import scanpy as sc
from anndata import AnnData
import anndata
import torch.nn.functional as F
from scipy import sparse as sp
import scipy.sparse

#os.chdir("../../")


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro



def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")


def get_train_test_id(train_id, train_labels):
   trains=[]
   tests=[]
   label_type=np.unique(train_labels)
   #import pdb;pdb.set_trace()
   for label_name in label_type:
       index = np.where(train_labels == label_name)
       same_label_id=train_id[index]
       idx_train, idx_val = train_test_split(same_label_id, test_size=0.2, random_state=42)
       trains.extend(idx_train)
       tests.extend(idx_val)
    
   return trains, tests 


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features




def load_GBP_data(datastr,alpha,rmax,rrz, origin_data=False, data_scale=False, umap=False):
    base="data/"
    batch1_label = base + datastr + "/Label1.csv"
    batch2_label = base + datastr + "/Label2.csv"

    origin=origin_data
    scale=data_scale
    umap=umap

    if not origin:
        features = BiP.ppr(datastr, alpha, rmax, rrz)
        features = torch.FloatTensor(features).T
        print("over ppr")
    else:
        features=np.load("data/"+datastr+"_feat.npy")
        features = torch.FloatTensor(features)



    label1 = pd.read_csv(batch1_label).values.flatten()
    label2 = pd.read_csv(batch2_label).values.flatten()

    labels=np.concatenate((label1, label2), axis=0)
    label_types = np.unique(labels).tolist()
    labels = pd.DataFrame(labels)
    rename = {}

    for line in range(0, len(label_types)):
        key = label_types[line]
        rename[key] = int(line)

    labels = labels.replace(rename)
    labels=labels.values.flatten()
    index1 = len(label1)
    index2=len(label2)

    idx_train = np.arange(index1)
    idx_test= np.arange(index1, index1+index2)

    idx_train, idx_val=get_train_test_id(idx_train, label1)

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("over load data")

    return features,labels,idx_train,idx_val,idx_test, rename


def get_silhouette_score(data, cell_type):
    from sklearn.metrics import silhouette_score

    sil_corr = silhouette_score(data, cell_type)
    print("silhouette: ",np.median(sil_corr))
    return sil_corr
