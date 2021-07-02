#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division,print_function
from utils import load_citation,muticlass_f1, accuracy, muticlass_f1_test, get_silhouette_score
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model import GnnBP
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from vat import VATLoss
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import psutil
import torch.distributed as dist
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# In[2]:


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_data', type=str, default=False, help='use the origin data.')
    parser.add_argument('--scale', type=str, default=False, help='use scale normlization.')
    parser.add_argument('--umap', type=str, default=False, help='umap.')
    parser.add_argument('--seed', type=int, default=20159, help='random seed.')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
    parser.add_argument('--layer', type=int, default=2, help='number of layers.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate.')
    parser.add_argument('--patience', type=int, default=20, help='patience')
    parser.add_argument('--data', default='dataset8', help='dateset8')
    parser.add_argument('--dev', type=int, default=3, help='device id')
    parser.add_argument('--alpha', type=float, default=0.05, help='decay factor')
    parser.add_argument('--rmax', type=float, default=1e-5, help='threshold.')
    parser.add_argument('--rrz', type=float, default=0.5, help='r.')
    parser.add_argument('--bias', default='none', help='bias.')
    parser.add_argument('--batch', type=int, default=120, help='batch size')
    parser.add_argument('--gpus', type=list, default=[0,1,2,7], help='parallel gpu ids')
    parser.add_argument('--vat_lr', type=float, default=0.1, help='r.')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.origin_data=='T':
        args.origin_data=True
    else:
        args.origin_data = False

    if args.scale=='T':
        args.scale=True
    else:
        args.scale = False

    if args.umap=='T':
        args.umap=True
    else:
        args.umap = False
    print("--------------------------")
    print(args)
    return args


# In[3]:


def train(model,loader,test_loader,dev_id,optimizer,loss_fn):
    lds=0
    model.train()
    loss_list = []
    time_epoch = 0
    t1 = time.time()
    for (batch_x, batch_y), (test_batch_x, test_batch_y) in zip(loader, test_loader):
    #for batch_x, batch_y in loader:

        batch_x = batch_x.to(dev_id)
        batch_y = batch_y.to(dev_id)

        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)

        if args.vat_lr > 0:
            test_batch_x = test_batch_x.to(dev_id)
            vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
            lds = vat_loss(model, test_batch_x)
        loss_train += lds*args.vat_lr

        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train.item())

    time_epoch += (time.time() - t1)
#     print(time.time() - t1)
    return np.mean(loss_list), time_epoch

def validate(model,features,labels,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features[idx_val])
        micro_val = muticlass_f1(output, labels[idx_val])
        return micro_val.item()

def acc_for_every_type(output, ori_labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = ori_labels.cpu().detach().numpy()

    label_types=np.unique(labels)
    for index, value in enumerate(label_types):
        type_index=np.where(labels==value)[0]
        acc = accuracy(output[type_index], ori_labels[type_index])
        micro = muticlass_f1(output[type_index],ori_labels[type_index])
        print(value, "ratio: ",len(type_index)/len(labels) ,": acc ",acc.item(), " micro: ", micro.item())


# In[4]:


def test(rename,model):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    test_time=time.time()
    with torch.no_grad():
        output = model(features[idx_test])
        #micro_test=muticlass_f1(output, labels[idx_test])
        acc=accuracy(output, labels[idx_test])
        acc_for_every_type(output, labels[idx_test])
        print("test time:", time.time()-test_time)

        rename = {v: k for k, v in rename.items()}
        # import pdb;
        # pdb.set_trace()
        output= output.data.cpu().numpy().argmax(1) #
        test_labels = labels[idx_test].data.cpu().numpy()
        output=output.tolist()
        test_labels = test_labels.tolist()

        output=pd.DataFrame({"pred_label":output})
        test_labels = pd.DataFrame({"true_label":  test_labels})
        output=output.replace(rename).values.flatten()
        test_labels = test_labels.replace(rename).values.flatten()
        pd.DataFrame({"true_label":  test_labels, "pred_label":output}).to_csv("pred.csv", index=False)

        return acc.item(), acc.item()

def test_chunk(rename,model,test_loader,checkpt_file):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    test_time = time.time()
    with torch.no_grad():
        output_list = []
        test_chunk_lable = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.cuda()
            output_list.append(model(batch_x))
            test_chunk_lable.extend(batch_y)

        output = output_list[0]
        for i in range(1, len(output_list)):
            output = torch.cat((output, output_list[i]), 0)

        test_chunk_lable = torch.LongTensor(test_chunk_lable).cuda()
        micro_test = muticlass_f1(output, test_chunk_lable)
        acc = accuracy(output, test_chunk_lable)
        acc_for_every_type(output, test_chunk_lable)
        print("test time:", time.time() - test_time)


        ###################get the pred cell type#################
        # import pdb;
        # pdb.set_trace()
        # rename = {v: k for k, v in rename.items()}
        # output = output.data.cpu().numpy().argmax(1)  #
        # test_labels = test_chunk_lable.data.cpu().numpy()
        # output = output.tolist()
        # test_labels = test_labels.tolist()
        #
        # output = pd.DataFrame({"pred_label": output})
        # test_labels = pd.DataFrame({"true_label": test_labels})
        # output = output.replace(rename).values.flatten()
        # test_labels = test_labels.replace(rename).values.flatten()
        # pd.DataFrame({"true_label": test_labels, "pred_label": output}).to_csv("pred.csv", index=False)

        ####################################

        return micro_test.item(), acc.item()


# In[5]:


def run(proc_id, args, data):
    if proc_id==0:
        print('?')
    #torch.backends.cudnn.benchmark = True
    features,labels,idx_train,idx_val,idx_test,rename,checkpt_file,p=data
    devices=args.gpus
    n_gpus=len(args.gpus)
    dev_id = devices[proc_id]
    
    if proc_id==0:
        print('?')
    if n_gpus > 1:
        dist.init_process_group(backend="nccl",
                                init_method='tcp://127.0.0.1:'+str(10000+proc_id*100+1),
                                world_size=n_gpus,
                                rank=proc_id
                                )
    if proc_id==0:
        print('?')
        
    torch.cuda.set_device(dev_id)
    
    model = GnnBP(nfeat=features.shape[1],
                nlayers=args.layer,
                nhidden=args.hidden,
                nclass=labels.max().item()+1,
                dropout=args.dropout,
                bias = args.bias)
    model = DistributedDataParallel(model.cuda(), device_ids=[dev_id],
                                    #find_unused_parameters=True,
                                    output_device=dev_id)
    
    if proc_id==0:
        print('?')
    torch_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
    sampler = Data.distributed.DistributedSampler(torch_dataset)###
    loader = Data.DataLoader(dataset=torch_dataset,
                            batch_size=args.batch,
                            #pin_memory=True,
                            sampler=sampler,
                            num_workers=3) # num_workers=6

    torch_test_dataset = Data.TensorDataset(features[idx_test], labels[idx_test])
    test_sampler= Data.distributed.DistributedSampler(torch_test_dataset)
    test_loader = Data.DataLoader(dataset=torch_test_dataset,
                            batch_size=args.batch,
                            #pin_memory=True,
                            sampler=test_sampler,
                            num_workers=3)#num_workers=6
    
    if proc_id==0:
        print('?')
    begin_time_train = time.time()
    train_time = 0
    bad_counter = 0
    best = 0
    best_epoch = 0
    val_time_total=0

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        loss_tra,train_ep = train(model,loader,test_loader,dev_id,optimizer,loss_fn)
        begin_val_time = time.time()
        f1_val = validate(model,features,labels,idx_val)
        val_time_total+=time.time()-begin_val_time
        train_time+=train_ep
        if(epoch+1)%1 == 0 and proc_id == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                '| val acc:{:.6f}'.format(f1_val),
                '| cost:{:.3f}'.format(train_time))
        if f1_val > best:
            best = f1_val
            best_epoch = epoch
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
        # if  epoch==100:
        #     break


    #f1_test, acc = test(rename)
    end_time = time.time()
    if n_gpus > 1:
        torch.distributed.barrier()
    if proc_id == 0:
        print("Train cost: {:.4f}s".format(train_time))
        print("val time cost: {:.4f}s".format(val_time_total))
        print("train + val time: {:.4f}s".format(train_time+val_time_total))

        print('Load {}th epoch'.format(best_epoch))
        # print("Test f1:{:.3f}".format(f1_test))
        # print("Test acc:{:.3f}".format(acc))
        print("between train begin and train end time cost: {:.4f}s".format(end_time-begin_time_train))
        print("total time after getting ppr data: {:.4f}s".format(end_time-begin_time))
        print("-------------------------------------------------------------")

        f1_test2, acc2 =test_chunk(rename,model,test_loader,checkpt_file)
        print("Test f2:{:.3f}".format(f1_test2))
        print("Test acc2:{:.3f}".format(acc2))
        print("--------------------------")

        created = p.memory_full_info().uss/1024/1024
        print("total memory:", created, "MB")
        print("process memory:", created-start, "MB")


# In[6]:


if __name__ == '__main__':
  args=set_args()
  features,labels,idx_train,idx_val,idx_test,rename = load_citation(args.data,args.alpha,args.rmax,args.rrz, origin_data=args.origin_data,
                                                         data_scale=args.scale, umap=args.umap)
  ori_labels=labels
  ori_idx_test=idx_test
  ori_features=features
  ori_idx_train=idx_train
  ori_idx_val=idx_val
  
  begin_time = time.time()
  checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
  # memory
  pid = os.getpid()
  p = psutil.Process(pid)
  start = p.memory_full_info().uss/1024/1024
  data=features,labels,idx_train,idx_val,idx_test,rename,checkpt_file,p
  
  
  # In[7]:
  
  procs = []
  for proc_id in range(len(args.gpus)):
      pro = mp.Process(target=run, args=(proc_id, args, data))
      pro.start()
      procs.append(pro)
  for pro in procs:
      pro.join()
  #mp.spawn(run, nprocs=len(args.gpus), args=(args, data))


# In[ ]:




