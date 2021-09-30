# the idea of pre_process code come from scGCN and seurat



#' This function returns a set of highly variable features
#' @param count.list list of raw counts of (1) reference data and (2) query data, rows are genes and columns are cells
#' @param var.features variable features of data list
#' @return This function returns a common set of highly variable features
#' @export
#' @examples
#' selectIntegrationFeature(count.list,var.features)
selectIntegrationFeature <- function(count.list,var.features,nfeatures = 2000){
    var.features1 <- unname(unlist(var.features))
    var.features2 <- sort(table(var.features1), decreasing = TRUE)
    for (i in 1:length(count.list)) {
        var.features3 <- var.features2[names(var.features2) %in% rownames(count.list[[i]])]}    
    tie.val <- var.features3[min(nfeatures, length(var.features3))]
    features <- names(var.features3[which(var.features3 > tie.val)])
    if (length(features) > 0) {
        feature.ranks <- sapply(features, function(x) {
            ranks <- sapply(var.features, function(y) {
                if (x %in% y) {
                    return(which(x == y))
                }
                return(NULL)
            })
            median(unlist(ranks))
        })
        features <- names(sort(feature.ranks))
    }
    features.tie <- var.features3[which(var.features3 == tie.val)]
    tie.ranks <- sapply(names(features.tie), function(x) {
        ranks <- sapply(var.features, function(y) {
            if (x %in% y) {return(which(x == y))}
            return(NULL)
        })
        median(unlist(ranks))
    })
    features <- c(features, names(head(sort(tie.ranks), nfeatures - length(features))))
    return(features)
}




normalize_data <- function(count.list){
    norm.list <- vector('list')
    var.features <- vector('list')
    for ( i in 1:length(count.list)){
        norm.list[[i]] <- as.matrix(Seurat:::NormalizeData.default(count.list[[i]]))
        #' select variable features
        hvf.info <- Seurat:::FindVariableFeatures.default(count.list[[i]],selection.method='vst')
         hvf.info <- hvf.info[which(x = hvf.info[, 1, drop = TRUE] != 0), ]
        hvf.info <- hvf.info[order(hvf.info$vst.variance.standardized, decreasing = TRUE), , drop = FALSE]
        var.features[[i]] <- head(rownames(hvf.info), n = 2000)
    }
     
    sel.features <- selectIntegrationFeature(count.list,var.features)
    
    return (list(norm.list,sel.features))}


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
    hvg_features <- res1[[2]]; 

    return (list(count_list_new,norm_list,hvg_features))
}



save_processed_data <- function(count.list,label.list,batch_info, data_name="example", tpm_data=FALSE){
    print("begin pre_process_data \n")
    res1 <- pre_process(count_list=count.list,label_list=label.list)

    count.list<- res1[[1]]
    norm.list <- res1[[2]];
    hvg.features <- res1[[3]]

    dir.create(paste0('../data/', data_name))
    write.csv(label.list[[1]],file=paste0('../data/', data_name, '/Label1.csv'),quote=F,row.names=F)
    write.csv(label.list[[2]],file=paste0('../data/', data_name, '/Label2.csv'),quote=F,row.names=F)
    printer = file(paste0('../data/', data_name, '/batch_info.txt'),"w")
    writeLines(as.character(batch_info),con=printer,sep=' ')
    close(printer)
   
    #' save processed data to certain path: 'process_data'
    dir.create('../process_data')
    outputdir <- paste0('../process_data/',data_name); dir.create(outputdir)
    write.csv(hvg.features,file=paste0(outputdir,'/sel_features.csv'),quote=F,row.names=F)

    N <- length(norm.list)

    for (i in 1:N){
        df = norm.list[[i]]
        if (!dir.exists(paste0(outputdir,'/norm_data'))){dir.create(paste0(outputdir,'/norm_data'))}
        file.name=paste0(outputdir,'/norm_data/norm_data_',i,'.csv')
        write.csv(df,file=file.name,quote=F)
    }

   if (tpm_data){
    for (i in 1:N){
        df = count.list[[i]]
        # count to TPM
       df = apply(df,2,function(x) (x*10^6)/sum(x))

        if (!dir.exists(paste0(outputdir,'/tpm_data'))){dir.create(paste0(outputdir,'/tpm_data'))}
        file.name=paste0(outputdir,'/tpm_data/tpm_data_',i,'.csv')
        write.csv(df,file=file.name,quote=F)

  } }


}




