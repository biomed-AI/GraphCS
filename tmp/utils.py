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


def muticlass_f1_test(output, labels, real_idx_test):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    output_raito = F.softmax(output, dim=1)
    output_raito=output_raito.cpu().detach().numpy()
    row, col=output.shape
    high_score_idx=[]
    high_score_lable = []

    label_types=np.unique(labels)
    for type_value in label_types:
        type_index=np.where(preds==type_value)
        label_ratio=output_raito[type_index][:, type_value]

        if len(type_index[0])<1:
            continue
        type_index=[index_value for index, index_value in enumerate(type_index[0]) if label_ratio[index]>0.7]
        if len(type_index)<1:
            continue

        top_k=int(len(type_index)/10)
        if top_k ==0:
            top_k=len(type_index)

        row_index=type_index
        #row_index=type_index[np.argpartition(label_ratio, top_k)[-top_k:]]

        high_score_idx.extend(row_index)
        high_score_lable.extend([type_value]*len(row_index))

    micro = f1_score(labels, preds, average='micro')
    return micro, high_score_idx, high_score_lable


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


def get_max_pred_ratio(label1, label2):
    num_count=0
    same_type=0
    types=np.unique(label2)
    for type_name in types:
        if type_name not in label1:
            continue
        type_num=len(label2[label2==type_name])
        num_count+=type_num

    max_correct_raito=num_count/len(label2)
    print("max_correct_raito: ",max_correct_raito)

    print("refernce: query")
    totol_type=np.unique(np.concatenate((np.unique(label1), np.unique(label2)),axis=0))
    for type_name in totol_type:
        type_num1 = len(label1[label1 == type_name])
        type_num2 = len(label2[label2 == type_name])
        print("type_name: ", type_name,": ", type_num1, " " ,type_num2)

    print("over get_max_pred_ratio")


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def chcek_graph_error_rate(datastr, labels, train_index, test_index):
    inter_count=0
    inter_error_count=0
    train_intra_count=0
    train_intra_error_count=0
    test_intra_count=0
    test_intra_error_count=0
    error_count=0

    graph_path="data/"+datastr+".txt"
    data=pd.read_table(graph_path).values
    total_count = data.shape[0]
    edge_con=True
    for value in data:
         split_value= value[0].split(' ')
         index1, index2=int(split_value[0]), int(split_value[1])

         if labels[index1]!=labels[index2]:
             error_count+=1
             edge_con=False
         if index1 <train_index and index2 <train_index:
             train_intra_count+=1
             if not edge_con:
                 train_intra_error_count+=1
         elif index1>train_index and index2>train_index:
             test_intra_count+=1
             if not edge_con:
                 test_intra_error_count+=1
         else:
             inter_count+=1
             if not edge_con:
                  inter_error_count+=1
         edge_con = True

    print("graph error: ", error_count/total_count)
    print("train_intra_count ", train_intra_count)
    print("train_intra_error_count ", train_intra_error_count/train_intra_count)
    print("inter_count ", inter_count)
    print("inter_error_count ", inter_error_count/inter_count)
    print("test_intra_count ", test_intra_count)
    if test_intra_count >0:
        print("test_intra_error_count ", test_intra_error_count/test_intra_count)




def load_citation(datastr,alpha,rmax,rrz, origin_data=False, data_scale=False, umap=False):
    base = "data/"  
    batch1_label = base + datastr + "/Label1.csv"
    batch2_label = base + datastr + "/Label2.csv"

    origin=origin_data
    scale=data_scale
    umap=umap

    if not origin:
        features = BiP.ppr(datastr, alpha, rmax, rrz)
        features = torch.FloatTensor(features).T
        print("over ppr")
        if scale:
            features = features.data.cpu().numpy()
            features = minmax_scale(features, feature_range=(0, 1), axis=0)
            features = torch.FloatTensor(features)
    else:
        features=np.load("data/"+datastr+"_feat.npy")

        if scale:
            features = minmax_scale(features, feature_range=(0, 1), axis=0)
        features = torch.FloatTensor(features)

   #####################################rename label#########
    label1 = pd.read_csv(batch1_label).values.flatten()
    label2 = pd.read_csv(batch2_label).values.flatten()
    get_max_pred_ratio(label1, label2)

    labels=np.concatenate((label1, label2), axis=0)
    label_types = np.unique(labels).tolist()
    labels = pd.DataFrame(labels)
    rename = {}

    for line in range(0, len(label_types)):
        key = label_types[line]
        rename[key] = int(line)

    print(rename)
    labels = labels.replace(rename)
    labels=labels.values.flatten()
    index1 = len(label1)
    index2=len(label2)
    print("index1: ", index1)
    print("index2: ", index2)

    idx_train = np.arange(index1)
    idx_test= np.arange(index1, index1+index2)
    idx_train, idx_val=get_train_test_id(idx_train, label1)


    print(features.shape)
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
