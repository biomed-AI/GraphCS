import pandas as pd
import scanpy as sc
import scvi
from scvi.model import SCANVI
import os
import argparse
import tarfile
from scvi.data import setup_anndata, synthetic_iid, transfer_anndata_setup
from scvi.dataloaders import (
    AnnDataLoader,
    DataSplitter,
    SemiSupervisedDataLoader,
    SemiSupervisedDataSplitter,
)
import time
import anndata as ad
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
from scipy.sparse import csr_matrix
from torch.nn import Softplus


 #First, you need to convert RData data to h5ad data using conver_between_scanpy_seurat.R 
 # ,and save the h5ad data into dataset


EPOCHES=15
base='./dataset/'
def proc_data(dataset,key='CellType'):
    adata=dataset.copy()
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    scvi.data.setup_anndata(adata, layer="counts", labels_key=key)
    return adata
def main(dataname,bigdata):
    trainset=sc.read_h5ad(base+'/train/%s.h5ad'%(dataname))
    testset=sc.read_h5ad(base+'/test/%s.h5ad'%(dataname))
    label=testset.obs['CellType'].copy()
    testset.obs['CellType']=trainset.obs['CellType'][0]
    if bigdata:
        config={}
        config['max_epochs']=200
        config['batch_size']=4096
        config['early_stopping']=True
        config['early_stopping_patience']=5
    else:
        config={'max_epochs':EPOCHES}
    while True:
        try:
            adata1=proc_data(trainset)
            adata2=proc_data(testset)
            model=SCANVI(adata1.copy(),'Unk')
            model.train(**config, check_val_every_n_epoch=1)
            break
        except Exception as e:
            print(e)
            if config['max_epochs']>3:
                config['max_epochs']-=1
            else:
                break
            continue
    pred=model.predict(adata2)
    adata2.obs['pred']=pred
    adata2.obs['CellType']=label
    acc=sum(adata2.obs['CellType']==adata2.obs['pred'])/adata2.n_obs
    return acc
def scanvi_test(filename):
    ACC=[]
    for i in filename:
        print('---------------------------------')
        print('loading file %s'%(i))
        bigdata=i=='mouse_brain'
        acc=main(i,bigdata)
        print(acc)
        ACC.append(acc)
        print('---------------------------------')
    return ACC
if __name__=='__main__':
    sim=[]
    for i in range(1,9):
        for j in range(1,6):
            tmp=round(i*0.2,1) if i!=5 else 1
            sim.append("splatter_2000_1000_4_batch.facScale"+str(tmp)+"_de.facScale0.2_10000_"+str(j))
    print(len(sim))
    filename=[
        [
          'pbmcsca_10x_Chromium_CEL-Seq2',
          'pbmcsca_10x_Chromium_Drop-seq',
          'pbmcsca_10x_Chromium_inDrops',
          'pbmcsca_10x_Chromium_Seq_Well',
          'pbmcsca_10x_Chromium_Smart-seq2',
          'mouse_retina',
          'mouse_brain',
        ],
        [
          'Baron_mouse_Baron_human',
          'Baron_mouse_segerstolpe',
          'Baron_human_Baron_mouse',
          'Baron_mouse_combination',
        ],
        sim
    ]
    savebase=['../cross-platforms.csv','../cross-species.csv','../simulate.csv']
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoches', type=int, default=15, help='number of epochs.')
    parser.add_argument('--data_sets', type=list, default=[0,1,2], help='0: cross-platforms  1: cross-species  2: simulated default: 012')
    parser.add_argument('--data_name', type=str, default=None, help='if specified, run the input dataset')
    parser.add_argument('--write', default=True, help='write to csv.')
    
    args = parser.parse_args()
    EPOCHES=args.epoches
    args.write=args.write!='False'
    print(args)
    res={}
    if args.data_name is not None:
        print('---------------------------------')
        print('loading file %s'%(args.data_name))
        acc=main(args.data_name,args.data_name=='mouse_brain')
        print('accuracy of '+args.data_name+' is %.4f'%(acc))
    else:
        args.data_sets=map(int,args.data_sets)
        for i in args.data_sets:
            ACC=scanvi_test(filename[i])
            ACC=np.array(ACC)
            if i==2:
                ACC=np.reshape(ACC,(8,5))
                std=pd.read_csv('../simulate-std.csv',header=0,index_col=0)
                std.loc['scANVI']=np.std(ACC,1)
                std.to_csv('../simulate-std.csv')
                ACC=ACC.mean(1).reshape(-1)
            res[savebase[i][3:-4]]=ACC.round(3)
            if args.write:
                result=pd.read_csv(savebase[i],header=0,index_col=0)
                result.loc['scANVI']=ACC
                result.to_csv(savebase[i])
        print(res)
