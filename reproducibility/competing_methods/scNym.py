import pandas as pd
import scanpy as sc
import os
import tarfile
import time
import anndata as ad
import argparse
import numpy as np
import copy as cp
from scnym.api import scnym_api

 #First, you need to convert RData data to h5ad data using conver_between_scanpy_seurat.R 
  # ,and save the h5ad data into dataset

base='./dataset'
def main(dataname,bigdata=False):
    trainset=sc.read_h5ad(base+'/train/%s.h5ad'%(dataname))
    testset=sc.read_h5ad(base+'/test/%s.h5ad'%(dataname))
    if bigdata:
        config={'n_epochs': 200,'patience': 10,'batch_size': 4096}
    else:
        config='default'
    sc.pp.normalize_total(trainset, target_sum=1e6)
    sc.pp.normalize_total(testset, target_sum=1e6)
    sc.pp.log1p(trainset)
    sc.pp.log1p(testset)
    print('training...')
    while True:
        try:
            scnym_api(
                adata=trainset.copy(),
                task='train',
                groupby='CellType',
                out_path='./model/',
                config=config,
            )
            break
        except Exception as e:
            continue
    print('predicting...')
    scnym_api(
        adata=testset,
        task='predict',
        key_added='pred',
        trained_model='./model/',
        config=config,
    )
    acc=sum(testset.obs['CellType']==testset.obs['pred'])/testset.n_obs
    return acc
def scnym_test(filename):
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
    parser.add_argument('--data_sets', type=list, default=[1,2,0], help='0: cross-platforms  1: cross-species  2: simulated  default: 120')
    parser.add_argument('--data_name', type=str, default=None, help='if specified, run the input dataset')
    parser.add_argument('--write', default=True, help='write to csv.')
    
    args = parser.parse_args()
    args.write=args.write!='False'
    res={}
    if args.data_name is not None:
        print('---------------------------------')
        print('loading file %s'%(args.data_name))
        acc=main(args.data_name,args.data_name=='mouse_brain')
        print('accuracy of '+args.data_name+' is %.4f'%(acc))
    else:
        args.data_sets=[int(i) for i in args.data_sets]
        for i in args.data_sets:
            ACC=scnym_test(filename[i])
            ACC=np.array(ACC)
            if i==2:
                ACC=np.reshape(ACC,(8,5))
                std=pd.read_csv('../simulate-std.csv',header=0,index_col=0)
                std.loc['scNym']=np.std(ACC,1)
                std.to_csv('../simulate-std.csv')
                ACC=ACC.mean(1).reshape(-1)
            res[savebase[i][3:-4]]=ACC.round(3)
            if args.write:
                result=pd.read_csv(savebase[i],header=0,index_col=0)
                result.loc['scNym']=ACC
                result.to_csv(savebase[i])
        print(res)
