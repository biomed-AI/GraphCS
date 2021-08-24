source('r-model.R')
method=list('Seurat-PCA','SingleR','scmap')
func=list(Seurat3_PCA,singleR_ck,scmap_ck)
filename='mouse_brain'
savepath='../cross-platforms.csv'
logpath="cross-platforms-bigdata.log"
if (!dir.exists('log')){dir.create('log')}

test<-function(filename,savepath,logpath,func,method){
    con <- file(paste0('./log/',method,'-',logpath))
    #sink(con,append = FALSE)
    #sink(con,append = FALSE,type ="message")
    print('---------------------------------')
    print(paste0('Loading data ',filename,' ...'))
    data0=read_data(filename)
    trainset<-data0[[1]]
    testset<-data0[[2]]
    trainlabel<-data0[[3]]
    testlabel<-data0[[4]]
    pred=func(trainset,testset,trainlabel,testlabel)
    acc=sum(pred[[1]]==pred[[2]])/length(pred[[1]])
    print(paste0('Accuracy of ',method,' on bigdata: ',acc))
    result=read.csv(savepath,header=TRUE,row.names=1,stringsAsFactors = FALSE)
    result[method,'mouse_brain']=acc
    write.csv(result,savepath,row.names=T)
    #sink()
    #sink(type="message")
}
for(j in 1:length(method)){
    print(paste0('processing ',filename,' with ',method[[j]]))
    test(filename,savepath,logpath,func[[j]],method[[j]])
}
