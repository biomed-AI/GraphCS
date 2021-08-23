library(CellID)
library(tidyverse) # general purpose library for data handling
library(ggpubr) #library for plotting
setwd('./')
source('func.R')
sim=c()
for(i in 1:8){
    sim=c(sim,paste0("splatter_2000_1000_4_batch.facScale",0.2*i,"_de.facScale0.2_10000_",1:5))
}
filename=list(
    list(
        'pbmcsca_10x_Chromium_CEL-Seq2',
        'pbmcsca_10x_Chromium_Drop-seq',
        'pbmcsca_10x_Chromium_inDrops',
        'pbmcsca_10x_Chromium_Seq_Well',
        'pbmcsca_10x_Chromium_Smart-seq2',
        'mouse_retina'
    ),
    list(
        'Baron_mouse_Baron_human',
        'Baron_mouse_segerstolpe',
        'Baron_human_Baron_mouse',
        'Baron_mouse_combination'
    ),
    sim
)
if (!dir.exists('log')){dir.create('log')}
savepath=list('../cross-platforms.csv','../cross-species.csv','../simulate.csv')
logpath=list("cross-platforms.log",'cross-species.log','simulate.log')
test<-function(filename,savepath,logpath){
    con <- file(paste0("./log/cellid-",logpath))
    sink(con,append = FALSE)
    sink(con,append = FALSE,type ="message")
    ACC1=c()
    ACC2=c()
    for(i in filename){
        print('---------------------------------')
        print(paste0('Loading data ',i,' ...'))
        data0=read_data(i)
        trainset<-data0[[1]]
        testset<-data0[[2]]
        trainlabel<-as.data.frame(data0[[3]])
        testlabel<-as.data.frame(data0[[4]])
        
        print('pre-processing ref-set...')
        train <- CreateSeuratObject(counts = trainset, project = paste0(i,'-train'), min.cells = 5, meta.data = trainlabel)
        test <- CreateSeuratObject(counts = testset, project = paste0(i,'-test'), min.cells = 5, meta.data = testlabel)
        train <- NormalizeData(train)
        train <- ScaleData(train, features = rownames(train))
        train <- RunMCA(train)
        
        print('pre-processing query-set...')
        test=preproc(test)
        
        print('performing prediction...')
        c2c_result=cell2cell(train,test)
        g2c_result=group2cell(train,test)
        c2c_acc=evaluate(c2c_result)
        g2c_acc=evaluate(g2c_result)
        print(paste0('accuracy of cell-to-cell on ',i,' is ',c2c_acc,'%'))
        print(paste0('accuracy of group-to-cell on ',i,' is ',g2c_acc,'%'))
        ACC1=c(ACC1,c2c_acc)
        ACC2=c(ACC2,g2c_acc)
        print('---------------------------------')
    }
    if(savepath=='../cross-platforms.csv'){
        ACC=c(ACC,0)
    }
    if(savepath=='../simulate.csv'){
        ACC1=as.matrix(ACC1)
        dim(ACC1)=c(5,8)
        std=read.csv('../simulate-std.csv',header=TRUE,row.names=1,stringsAsFactors = FALSE)
        std['CelliD(C)',]=apply(ACC1,2,sd)
        write.csv(std,'../simulate-std.csv',row.names=T)
        ACC1=apply(ACC1,2,mean)
        ACC2=as.matrix(ACC2)
        dim(ACC2)=c(5,8)
        std=read.csv('../simulate-std.csv',header=TRUE,row.names=1,stringsAsFactors = FALSE)
        std['CelliD(G)',]=apply(ACC2,2,sd)
        write.csv(std,'../simulate-std.csv',row.names=T)
        ACC2=apply(ACC2,2,mean)
    }
    print(ACC1)
    print(ACC2)
    result=read.csv(savepath,header=TRUE,row.names=1,stringsAsFactors = FALSE)
    result['CelliD(C)',]=ACC1
    result['CelliD(G)',]=ACC2
    write.csv(result,savepath)
    sink()
    sink(type='message')
}
for(i in 1:3){
    print(paste0('processing ',logpath[[i]],' with cellid'))
    test(filename[[i]],savepath[[i]],logpath[[i]])
}
