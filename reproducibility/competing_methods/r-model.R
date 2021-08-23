library(SingleCellExperiment)
library(reticulate)
library(tidyverse)
library(Seurat)
library(scmap)
library(fmsb)
library(singleCellNet)
library(SingleR)
library(magrittr)
library(pryr)
write_pred_label<-function(true_label, pred_label, save_name,file_name){
    data=data.frame(true_label=true_label,pred_label=pred_label)
    dir.create('pred_label_result')
    write.csv(data, paste0("pred_label_result/", save_name,"_",file_name,".csv"), row.names=FALSE, sep = ",")
}
singlecellNet_ck<-function(train_data, query_data, train_Labels, test_Labels){
    # gene x cell
    Train_Labels=data.frame(cellname=colnames(train_data), CellType=as.character(train_Labels))
    Test_Labels=data.frame(cellname=colnames(query_data), CellType=as.character(test_Labels))

    class_info<-scn_train(stTrain = Train_Labels, expTrain = train_data, dLevel = "CellType", colName_samp = "cellname")
    classRes_val_all = scn_predict(cnProc=class_info[['cnProc']], expDat=query_data, nrand = 2)
    stQuery <- assign_cate(classRes = classRes_val_all[!rownames(classRes_val_all)=="rand",1:nrow(Test_Labels)], sampTab = Test_Labels)#cThresh = 0.5

     tibble(
    ori = as.character(test_Labels),
    prd = as.character(stQuery$category),
    method = 'singlecellNet')

}
singleR_ck<-function(ref_data, test_data, ref_labels, test_labels){
    #### format genes x cells
    singler = SingleR(test=test_data, ref=ref_data, labels=ref_labels)
    tibble(
    ori = test_labels,
    prd = as.character(singler$labels),
    method = 'singleR')

}
scmap_ck <- function(expr_train, expr_test,train_label, test_label , gene_num = 500){
    sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(expr_train)), colData = data.frame(cell_type1 = train_label))
    logcounts(sce) <- log2(normcounts(sce) + 1)
    rowData(sce)$feature_symbol <- rownames(sce)
    sce <- sce[!duplicated(rownames(sce)), ]
    sce <- selectFeatures(sce, suppress_plot = TRUE)

    tx_sce <- SingleCellExperiment(assays = list(normcounts = as.matrix(expr_test)))
    logcounts(tx_sce) <- log2(normcounts(tx_sce) + 1)
    rowData(tx_sce)$feature_symbol <- rownames(tx_sce)

    sce <- selectFeatures(sce, suppress_plot = T, n_features = gene_num)# n_features = 500

    sce <- indexCluster(sce)
    scmapCluster_results <- scmapCluster(
        projection = tx_sce,
        threshold = 0,
        index_list = list(
          yan = metadata(sce)$scmap_cluster_index
        )
    )

    tibble(
    ori = as.character(test_label),
    prd = unlist(scmapCluster_results$combined_labs),
    #prob = scmapCluster_results$scmap_cluster_siml[,1],
    method = 'scmap')

}                      
Seurat3_CCA <- function(X_train, X_test,train_label,test_label, gene_num = 2000){
  data.frame(
    celltype = as.character(train_label),
    tech = 'xx'
  ) -> metadata

  data.frame(
    celltype =  as.character(test_label),
    tech = 'yy'
  ) -> metadata1

  ori <- as.character(test_label)
  matr <- cbind(X_train, X_test)
  metadata <- rbind(metadata, metadata1)
  colnames(matr) <- as.character(1:ncol(matr))
  rownames(metadata) <- as.character(1:nrow(metadata))

  ttest <- CreateSeuratObject(counts = matr, meta.data = metadata)
  #browser()
  ttest.list <- SplitObject(object = ttest, split.by = "tech")

     start_time <- Sys.time()
  for (i in 1:length(x = ttest.list)) {
    ttest.list[[i]] <- NormalizeData(object = ttest.list[[i]], verbose = FALSE)
    ttest.list[[i]] <- FindVariableFeatures(object = ttest.list[[i]],
                                            selection.method = "vst", nfeatures = gene_num, verbose = FALSE)
  }
     end_time <- Sys.time()
  Seurat3_cca_Training_Time <- as.numeric(difftime(end_time,start_time,units = 'secs'))
    print("noramlized data time")
     print(Seurat3_cca_Training_Time)

  anchors <- FindTransferAnchors(reference = ttest.list[[1]],
                                 query = ttest.list[[2]],
                                 dims = 1:30,
                                 k.filter = 100,
                                 features = VariableFeatures(ttest.list[[1]]),
                                  reduction = 'cca'
                                  )

  predictions <- TransferData(anchorset = anchors,
                              refdata = ttest.list[[1]]$celltype,
                              dims = 1:30,
      weight.reduction = "cca"
                             )

  tibble(
    ori = ori,
    prd = predictions$predicted.id,
    #prob = predictions$prediction.score.max,
    method = 'Seurat'
  )
}
Seurat3_PCA <- function(X_train, X_test,train_label,test_label, gene_num = 2000){

  data.frame(
    celltype = as.character(train_label),
    tech = 'xx'
  ) -> metadata

  data.frame(
    celltype =  as.character(test_label),
    tech = 'yy'
  ) -> metadata1

  ori <- as.character(test_label)
  matr <- cbind(X_train, X_test)
  metadata <- rbind(metadata, metadata1)
  colnames(matr) <- as.character(1:ncol(matr))
  rownames(metadata) <- as.character(1:nrow(metadata))

  ttest <- CreateSeuratObject(counts = matr, meta.data = metadata)
  
  ttest.list <- SplitObject(object = ttest, split.by = "tech")

  start_time <- Sys.time()
  for (i in 1:length(x = ttest.list)) {
    ttest.list[[i]] <- NormalizeData(object = ttest.list[[i]], verbose = FALSE)
    ttest.list[[i]] <- FindVariableFeatures(object = ttest.list[[i]],
                                            selection.method = "vst", nfeatures = gene_num, verbose = FALSE)
  }
  end_time <- Sys.time()
  Seurat3_pca_Training_Time <- as.numeric(difftime(end_time,start_time,units = 'secs'))
    print("noramlized data time")
  print(Seurat3_pca_Training_Time)

  anchors <- FindTransferAnchors(reference = ttest.list[[1]],
                                 query = ttest.list[[2]],
                                 dims = 1:30,
                                 k.filter = 100,
                                 features = VariableFeatures(ttest.list[[1]])
                                  )

  predictions <- TransferData(anchorset = anchors,
                              refdata = ttest.list[[1]]$celltype,
                              dims = 1:30
                             )

  tibble(
    ori = ori,
    prd = predictions$predicted.id,
    #prob = predictions$prediction.score.max,
    method = 'Seurat'
  )
}
read_data<-function(data_name='dataset2'){

    print(data_name)
    base="../../example_data/"
    data=load(paste0(base,data_name,".RData"))
    data=get(data)

    train_data=data[[1]]@assays$RNA@counts
    test_data=data[[2]]@assays$RNA@counts
    train_label=as.character(data[[1]]@meta.data$CellType)
    test_label=as.character(data[[2]]@meta.data$CellType)

    #   common_gene for seurat
    train_gene= rownames(train_data)
    test_gene= rownames(test_data)
    common_gene=intersect(train_gene, test_gene)
    train_data=train_data[common_gene,]
    test_data = test_data[common_gene,]

    train_row_names=c(paste0("cell_",1:length(train_label)))
    test_row_names=c(paste0("cell_",(1+length(train_label)):(length(test_label)+length(train_label))))

    colnames(train_data)=train_row_names
    colnames(test_data)=test_row_names

    train_label_matrix=as.matrix(as.character(train_label))
    rownames(train_label_matrix)=train_row_names
    colnames(train_label_matrix)="label"

    test_label_matrix=as.matrix(as.character(test_label))
    rownames(test_label_matrix)=test_row_names
    colnames(test_label_matrix)="label"

    results=list(train_data=train_data, test_data=test_data,
    train_label=train_label, test_label=test_label)

    return(results)
}
