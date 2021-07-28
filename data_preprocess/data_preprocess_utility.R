# the idea of pre_process code come from scGCN and seurat

normalize_data <- function(count.list){
    norm.list <- vector('list')
    var.features <- vector('list')
    for ( i in 1:length(count.list)){
        norm.list[[i]] <- as.matrix(Seurat:::NormalizeData.default(count.list[[i]]))
        #' select variable features
        hvf.info <- Seurat:::FindVariableFeatures.default(count.list[[i]],selection.method='vst')
       # hvf.info <- hvf.info[which(x = hvf.info[, 1, drop = TRUE] != 0), ]
        #hvf.info <- hvf.info[order(hvf.info$vst.variance.standardized, decreasing = TRUE), , drop = FALSE]
        #var.features[[i]] <- head(rownames(hvf.info), n = 2000)
    }

    return (list(norm.list))}


select_feature <- function(data,label,nf=2000){
    M <- nrow(data); new.label <- label[,1]
    pv1 <- sapply(1:M, function(i){
        mydataframe <- data.frame(y=as.numeric(data[i,]), ig=new.label)
        fit <- aov(y ~ ig, data=mydataframe)
        summary(fit)[[1]][["Pr(>F)"]][1]})
    names(pv1) <- rownames(data)
    pv1.sig <- names(pv1)[order(pv1)[1:nf]]
    egen <- unique(pv1.sig)
    return (egen)
}


pre_process <- function(count_list,label_list){
    sel.features <- select_feature(count_list[[1]],label_list[[1]])
    count_list_new <- list(count_list[[1]][sel.features,],count_list[[2]][sel.features,])
    res1 <- normalize_data(count_list_new)
    norm_list <- res1[[1]];

    return (list(count_list_new,norm_list))
}



save_processed_data <- function(count.list,label.list, data_name="example"){
    print("begin pre_process_data \n")
    res1 <- pre_process(count_list=count.list,label_list=label.list)

    norm.list <- res1[[2]];

    dir.create(paste0('../data/', data_name))
    write.csv(label.list[[1]],file=paste0('../data/', data_name, '/Label1.csv'),quote=F,row.names=F)
    write.csv(label.list[[2]],file=paste0('../data/', data_name, '/Label2.csv'),quote=F,row.names=F)

    #' save processed data to certain path: 'process_data'
    dir.create('../process_data')
    outputdir <- paste0('../process_data/',data_name); dir.create(outputdir)

    N <- length(norm.list)

    for (i in 1:N){
        df = norm.list[[i]]
        if (!dir.exists(paste0(outputdir,'/norm_data'))){dir.create(paste0(outputdir,'/norm_data'))}
        file.name=paste0(outputdir,'/norm_data/norm_data_',i,'.csv')
        write.csv(df,file=file.name,quote=F)
    }

}





