library(scClassify)
source('func.R')
con <- file("scclassify.log")
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
    con <- file(paste0("./log/scclassify-",logpath))
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
        
        print('pre-processing dataset...')
        trainset <- log1p(trainset)
        testset <- log1p(testset)
        
        print('training dataset...')
        pred <- scClassify(exprsMat_train = trainset,
                              cellTypes_train = trainlabel,
                              exprsMat_test = list(test=testset),
                              cellTypes_test = list(test=testlabel),
                              tree = "HOPACH",
                              algorithm = "WKNN",
                              selectFeatures = c("limma"),
                              similarity = c("pearson"),
                              returnList = FALSE,
                              verbose = FALSE)
        print('predicting results...')
        acc=sum(pred$testRes$test$pearson_WKNN_limma$predRes==testlabel)/length(testlabel)
        ACC=c(ACC,acc)
        print(paste0('accuracy on ',i,' is ',acc,'%'))
        print('---------------------------------')
    }
    if(savepath=='../cross-platforms.csv'){
        ACC=c(ACC,0)
    }
    if(savepath=='../simulate.csv'){
        ACC=as.matrix(ACC)
        dim(ACC)=c(5,8)
        std=read.csv('../simulate-std.csv',header=TRUE,row.names=1,stringsAsFactors = FALSE)
        std['scClassify',]=apply(ACC,2,sd)
        write.csv(std,'../simulate-std.csv',row.names=T)
        ACC=apply(ACC,2,mean)
    }
    print(ACC)
    result=read.csv(savepath,header=TRUE,row.names=1,stringsAsFactors = FALSE)
    result['scClassify',]=ACC
    write.csv(result,savepath,row.names=T)
    sink()
    sink(type='message')
}
for(i in 1:3){
    print(paste0('processing ',logpath[[i]],' with scclassify'))
    test(filename[[i]],savepath[[i]],logpath[[i]])
}
