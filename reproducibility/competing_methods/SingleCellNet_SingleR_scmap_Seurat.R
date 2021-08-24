source('r-model.R')
method=list('SingleCellNet','SingleR','scmap','Seurat-CCA','Seurat-PCA')
func=list(singlecellNet_ck,singleR_ck,scmap_ck,Seurat3_CCA,Seurat3_PCA)
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
test<-function(filename,savepath,logpath,func,method){
    con <- file(paste0('./log/',method,'-',logpath))
    sink(con,append = FALSE)
    sink(con,append = FALSE,type ="message")
    ACC=c()
    for(i in filename){
          print('---------------------------------')
          print(paste0('Loading data ',i,' ...'))
          data0=read_data(i)
          trainset<-data0[[1]]
          testset<-data0[[2]]
          trainlabel<-data0[[3]]
          testlabel<-data0[[4]]
          pred=func(trainset,testset,trainlabel,testlabel)
          if(i=='mouse_retina'){
              write_pred_label(pred[[1]],pred[[2]],method,'mouse_retina')
          }    
          acc=sum(pred[[1]]==pred[[2]])/length(pred[[1]])
          ACC=c(ACC,acc)
          print(acc)
    }
    if(logpath=='cross-platforms.log'){
        ACC=c(ACC,0)
    }
    if(logpath=='simulate.log'){
        ACC=as.matrix(ACC)
        dim(ACC)=c(5,8)
        std=read.csv('../simulate-std.csv',header=TRUE,row.names=1,stringsAsFactors = FALSE)
        std[method,]=apply(ACC,2,sd)
        write.csv(std,'../simulate-std.csv',row.names=T)
        ACC=apply(ACC,2,mean)
    }
    print(paste0('Accuracy of ',method,' : ',paste0(ACC,collapse=' ')))
    result=read.csv(savepath,header=TRUE,row.names=1,stringsAsFactors = FALSE)
    result[method,]=ACC
    write.csv(result,savepath,row.names=T)
    sink()
    sink(type="message")
}
for(i in 1:length(filename)){
    for(j in 1:length(method)){
        print(paste0('processing ',logpath[[i]],' with ',method[[j]]))
        test(filename[[i]],savepath[[i]],logpath[[i]],func[[j]],method[[j]])
    }
}
