import pandas as pd
import numpy as np
acc=[]
base='../../../'
with open('cross-platforms.txt') as f:
    for i in f:
        acc.append(float(i[:-1]))
    acc.append(0)
    csv=pd.read_csv(base+'cross-platforms.csv',header=0,index_col=0)
    csv.loc['scGCN']=acc
    csv.to_csv(base+'cross-platforms.csv')
acc=[]
with open('cross-species.txt') as f:
    for i in f:
        acc.append(float(i[:-1]))
    csv=pd.read_csv(base+'cross-species.csv',header=0,index_col=0)
    csv.loc['scGCN']=acc
    csv.to_csv(base+'cross-species.csv')
acc=[]
with open('simulate.txt') as f:
    for i in f:
        acc.append(float(i[:-1]))
    acc=np.array(acc).reshape(5,8)
    std=acc.std(0)
    mean=acc.mean(0)
    csv=pd.read_csv(base+'simulate.csv',header=0,index_col=0)
    csv.loc['scGCN']=mean
    csv.to_csv(base+'simulate.csv')
    csv=pd.read_csv(base+'simulate-std.csv',header=0,index_col=0)
    csv.loc['scGCN']=std
    csv.to_csv(base+'simulate-std.csv')
