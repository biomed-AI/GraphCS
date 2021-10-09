from __future__ import division
from __future__ import print_function
from utils import load_GBP_data, muticlass_f1, accuracy, \
    get_silhouette_score,get_FPR, accuracy_for_unkonw
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
import torch.nn.functional as F


import os
import psutil
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# memory
pid = os.getpid()
p = psutil.Process(pid)
start = p.memory_full_info().uss/1024/1024

# Training settings
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
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--gpus', type=list, default=[0], help='parallel gpu ids')
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


features,labels,idx_train,idx_val,idx_test, rename = load_GBP_data(args.data,args.alpha,args.rmax,args.rrz)
ori_labels=labels
ori_idx_test=idx_test
ori_features=features
ori_idx_train=idx_train
ori_idx_val=idx_val

begin_time = time.time()
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'

model = GnnBP(nfeat=features.shape[1],
            nlayers=args.layer,
            nhidden=args.hidden,
            nclass=int(labels.max()) + 1,
            dropout=args.dropout,
            bias = args.bias).cuda()

model = nn.DataParallel(model.cuda(), device_ids=args.gpus)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

loss_fn = nn.CrossEntropyLoss()


features = features
labels = labels

torch_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
loader = Data.DataLoader(dataset=torch_dataset,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=3) # num_workers=6


torch_test_dataset = Data.TensorDataset(features[idx_test], labels[idx_test])
test_loader = Data.DataLoader(dataset=torch_test_dataset,
                        batch_size=args.batch,
                        shuffle=True,
                        num_workers=3)#num_workers=6



def train():
    lds=0
    model.train()
    loss_list = []
    time_epoch = 0
    t1 = time.time()
    for (batch_x, batch_y), (test_batch_x, test_batch_y) in zip(loader, test_loader):
    #for batch_x, batch_y in loader:

        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)

        if args.vat_lr > 0:
            test_batch_x = test_batch_x.cuda()
            vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
            lds = vat_loss(model, test_batch_x)
        loss_train += lds*args.vat_lr

        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train.item())

    time_epoch += (time.time() - t1)
    return np.mean(loss_list), time_epoch



def validate():
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
        print(value, "ratio: ",len(type_index)/len(labels), ": acc ", acc.item(), " micro: ", micro.item())


def test(rename):
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


def test_chunk(rename):
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


        ########################################## get FPR #########################################################



        query_label = set(list(labels[idx_test].numpy()))
        referenece_label=set(list(labels[idx_train].numpy()))
        unklonw_cell=list(query_label-referenece_label)
        print("unkonw_cell", unklonw_cell)

        pred_labels= output.max(1)[1].cpu().detach().numpy()
        #import pdb;pdb.set_trace()
        output_raito = F.softmax(output, dim=1).max(1)[0]
        output_raito = output_raito.cpu().detach().numpy()

        for label in query_label:
            cell_index = np.where(test_chunk_lable.cpu().detach().numpy() == label)[0]
            #print(output_raito[cell_index])
            print("cell_type:", label)
            print(np.mean(output_raito[cell_index]))
            print(set(pred_labels[cell_index]))


        #import pdb;pdb.set_trace()

        query_label = set(list(labels[idx_test].numpy()))
        referenece_label = set(list(labels[idx_train].numpy()))
        unklonw_cell = list(query_label - referenece_label)
        print("unkonw_cell", unklonw_cell)

        cut_off_ratio, number_unkonw=get_FPR(output, test_chunk_lable, unklonw_cell)
        print("number_unkonw, cut_off_ratio: ", number_unkonw, cut_off_ratio)
        #micro_test_cut = muticlass_f1_for_unkonw(output, test_chunk_lable, cut_off_ratio)
        acc_cut=accuracy_for_unkonw(output, test_chunk_lable, number_unkonw, cut_off_ratio)
        # acc_for_every_type(output, test_chunk_lable)
        #print("unkown cells f1 scores for left cells:", unklonw_cell, micro_test_cut)
        print("unkown cells acc scores only for known cells:", unklonw_cell, acc_cut)

        ###################################################################################################

        return acc_cut,acc_cut


begin_time_train = time.time()

train_time = 0
bad_counter = 0
best = 0
best_epoch = 0
val_time_total=0

for epoch in range(args.epochs):
    loss_tra,train_ep = train()
    begin_val_time = time.time()
    f1_val = validate()
    val_time_total+=time.time()-begin_val_time
    train_time+=train_ep
    if(epoch+1)%10 == 0:
        print('Epoch:{:04d}'.format(epoch+1),
            'train',
            'loss:{:.3f}'.format(loss_tra),
            '| val',
            'acc:{:.6f}'.format(f1_val),
            '| cost{:.3f}'.format(train_time))
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

print("Train cost: {:.4f}s".format(train_time))
print("val time cost: {:.4f}s".format(val_time_total))
print("train + val time: {:.4f}s".format(train_time+val_time_total))

print('Load {}th epoch'.format(best_epoch))
# print("Test f1:{:.3f}".format(f1_test))
# print("Test acc:{:.3f}".format(acc))
print("between train begin and train end time cost: {:.4f}s".format(end_time-begin_time_train))
print("total time after getting ppr data: {:.4f}s".format(end_time-begin_time))
print("-------------------------------------------------------------")



f1_test2, acc2 =test_chunk(rename)
print("Test f2:{:.3f}".format(f1_test2))
print("Test acc2:{:.3f}".format(acc2))
print("--------------------------")

created = p.memory_full_info().uss/1024/1024
print("total memory:", created, "MB")
print("process memory:", created-start, "MB")







