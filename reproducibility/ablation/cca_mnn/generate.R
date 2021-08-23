#' This functions takes raw counts and labels of reference/query set to generate scGCN training input
#' @param count.list list of reference data and query data; rows are genes and columns are cells
#' @param label.list list of reference label and query label (if any), both are data frames with rownames identical with colnames of data; the first column is cell type
#' @return This function returns files saved in folders "input" & "process_data"
#' @export: all files are saved in current path
#' @examples: load count.list and label.list from folder "example_data"
#' save_processed_data(count.list,label.list)
library(Seurat)
source('data_preprocess_utility.R')
base="../../../example_data/"



####################################################################################################
get_data_from_seurat_obj_with_unique<-function(data_name){

    data_path=paste0(base, data_name,".RData")
    seurat_obj=load(data_path)
    data=get(seurat_obj)

    train_data=as.matrix(data[[1]]@assays$RNA@counts)
    test_data=as.matrix(data[[2]]@assays$RNA@counts)
    train_label=data[[1]]@meta.data$CellType
    test_label=data[[2]]@meta.data$CellType

     train_row_names=c(1:length(train_label))
    test_row_names=c((1+length(train_label)):(length(test_label)+length(train_label)))

    colnames(train_data)=train_row_names
    colnames(test_data)=test_row_names

    train_label=data.frame(train_label, row.names=train_row_names)
    train_label[[1]]=as.character(train_label[[1]])
    test_label=data.frame(test_label, row.names=test_row_names)
    test_label[[1]]=as.character(test_label[[1]])

    count.list=list(train_data, test_data)
    label.list=list(train_label, test_label)

     results=list(count=count.list, label=label.list)
     return (results)
}



####################################################################################################




 data_name_list=c(

'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human',
'Baron_human_Baron_mouse'


)



for (file_name in data_name_list){
    #browser()
    options(future.globals.maxSize = 100000 * 1024^2)
    results=get_data_from_seurat_obj_with_unique(file_name)
    count.list <- results$count
    label.list <- results$label

    train_label=as.character(label.list[[1]]$train_label)
    test_label=as.character(label.list[[2]]$test_label)

    save_processed_data(count.list,train_label, test_label, file_name)

}
