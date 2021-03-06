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


def mutilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = 0
    return f1_score(y_true, y_pred, average="micro")


def accuracy_for_unkonw(output, labels, number_unkonw, cut_off_ratio=0):

    total_konw_cells=len(labels)-number_unkonw
    preds = output.max(1)[1].type_as(labels)

    output_raito = F.softmax(output, dim=1).max(1)[0]
    output_raito = output_raito.cpu().detach().numpy()
    index = np.where(np.array(output_raito) > cut_off_ratio)

    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    labels = labels[index]
    preds = preds[index]

    correct=sum(preds==labels)

    print("correct, total_konw_cells: ", correct, total_konw_cells)
    return correct / total_konw_cells


def get_FPR(output, labels, unklonw_cell,FPR=0.05):

    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # get the max probability
    output_raito = F.softmax(output, dim=1).max(1)[0]
    output_raito=output_raito.cpu().detach().numpy()

    finally_result=[]
    for cell in unklonw_cell:
        print("unknown cell: ", cell)
        # get the index of unkonw cells
        type_index = np.where(labels == cell)

        type_output=output_raito[type_index]
        type_output_list = sorted(type_output, reverse=True)
        finally_result.extend(type_output_list)

    number_cells=len(finally_result)
    cut_off_ratio=finally_result[int(number_cells*FPR)]

    print("cut_off_ratio value: ", cut_off_ratio)
    return cut_off_ratio, len(finally_result)


def get_train_test_id(train_id, train_labels):
   trains=[]
   tests=[]
   label_type=np.unique(train_labels)
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


def load_GBP_data(datastr,alpha,rmax,rrz, origin_data=True):
    base = "data/"  
    batch1_label = base + datastr + "/Label1.csv"
    batch2_label = base + datastr + "/Label2.csv"

    origin=origin_data

    features = BiP.ppr(datastr, alpha, rmax, rrz)  #rmax
    features = torch.FloatTensor(features).T

   #####################################rename label#########
    data1=pd.read_csv(batch1_label,header=0,index_col=0)
    label1 = data1.values.flatten()
    name1=data1.index
    data2=pd.read_csv(batch2_label,header=0,index_col=0)
    label2 = data2.values.flatten()
    name2=data2.index

    labels=np.concatenate((label1, label2), axis=0)
    label_types = np.unique(labels).tolist()
    labels = pd.DataFrame(labels)
    names=np.concatenate((name1,name2),axis=0)
    rename = {}

    for line in range(0, len(label_types)):
        key = label_types[line]
        rename[key] = int(line)

    #print(rename)
    labels = labels.replace(rename)
    labels=labels.values.flatten()
    index1 = len(label1)
    index2=len(label2)
    #print("index1: ", index1)
    #print("index2: ", index2)

    idx_train = np.arange(index1)
    idx_test= np.arange(index1, index1+index2)
    idx_train, idx_val=get_train_test_id(idx_train, label1)


    #print(features.shape)
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("over load data")

    return features,labels,names,idx_train,idx_val,idx_test, rename


def get_silhouette_score(data, cell_type):
    from sklearn.metrics import silhouette_score

    sil_corr = silhouette_score(data, cell_type)
    print("silhouette: ",np.median(sil_corr))
    return sil_corr
