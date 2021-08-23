library(SingleCellExperiment)
library(reticulate)
library(tidyverse)
library(Seurat)
library(fmsb)
library(magrittr)
library(pryr)



exmpale_read_from_seurat_object<-function(data_name='dataset2'){

    print(data_name)
    base="../../GraphCS/example_data/"
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
    colnames(train_label_matrix)="label"

    test_label_matrix=as.matrix(as.character(test_label))
    colnames(test_label_matrix)="label"

    results=list(train_data=train_data, test_data=test_data,
    train_label=train_label, test_label=test_label)


    return(results)

}




Seurat3_CCA <- function(filename, X_train, X_test,train_label,test_label,  gene_num = 2000){

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

  for (i in 1:length(x = ttest.list)) {
    ttest.list[[i]] <- NormalizeData(object = ttest.list[[i]], verbose = FALSE)
    ttest.list[[i]] <- FindVariableFeatures(object = ttest.list[[i]],
                                            selection.method = "vst", nfeatures = gene_num, verbose = FALSE)
  }

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


    if (!dir.exists('Seurat3')){dir.create('Seurat3')}
    outputdir="./Seurat3/"
    data_name=filename
    dir.create(paste0(outputdir,data_name))

    embedding_data=anchors@ object.list[[1]]@ reductions$ cca@ cell.embeddings
    cell_type=anchors@ object.list[[1]]@ meta.data$celltype



    file.name=paste0(outputdir,data_name,'/embedding_data.csv')
    write.csv(embedding_data,file=file.name,quote=F)

    file.name=paste0(outputdir,data_name,'/Label.csv')
    write.csv(cell_type,file=file.name,quote=F, row.names = F)


  tibble(
    ori = ori,
    prd = predictions$predicted.id,
    #prob = predictions$prediction.score.max,
    method = 'Seurat'
  )
}


pipe_fun <- function(filename="example"){



    pre_data=exmpale_read_from_seurat_object(filename)


    train_data=pre_data$train_data
    test_data=pre_data$test_data
    train_label=pre_data$train_label
    test_label=pre_data$test_label

    print(dim(train_data))
    print(dim(test_data))
    print("over read data")


     Seurat3_CCA(filename, train_data, test_data, train_label, test_label)
}






filename_list=c(

'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human'

)


for (filename in filename_list){
 res <- pipe_fun(filename=filename)
}
