#' This functions takes raw counts and labels of reference/query set to generate scGCN training input
#' @param count.list list of reference data and query data; rows are genes and columns are cells
#' @param label.list list of reference label and query label (if any), both are data frames with rownames identical with colnames of data; the first column is cell type
#' @return This function returns files saved in folders "input" & "process_data"
#' @export: all files are saved in current path
#' @examples: load count.list and label.list from folder "example_data"
#' save_processed_data(count.list,label.list)
library(Seurat)
source('data_preprocess_utility.R')
base="../../../../example_data/"


get_data_from_seurat_obj<-function(data_name){

    data_path=paste0(base, data_name,".RData")
    seurat_obj=load(data_path)
    data=get(seurat_obj)

    train_data=as.matrix(data[[1]]@assays$RNA@counts)
    test_data=as.matrix(data[[2]]@assays$RNA@counts)
    train_label=data[[1]]@meta.data$CellType
    test_label=data[[2]]@meta.data$CellType

    colnames(train_data)=train_label
    colnames(test_data)=test_label

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
sim=c()
for(i in 1:8){
    sim=c(sim,paste0("splatter_2000_1000_4_batch.facScale",0.2*i,"_de.facScale0.2_10000_",1:5))
}
data_name_list=c(
    list(
        'pbmcsca_10x_Chromium_CEL-Seq2',
        'pbmcsca_10x_Chromium_Drop-seq',
        'pbmcsca_10x_Chromium_inDrops',
        'pbmcsca_10x_Chromium_Seq_Well',
        'pbmcsca_10x_Chromium_Smart-seq2',
        'mouse_retina'
    ),
    list(

      'Baron_mouse_combination',
      'Baron_mouse_segerstolpe',
      'Baron_mouse_Baron_human',
       'Baron_human_Baron_mouse'
    ),


    sim 
)
print(length(data_name_list))

for (file_name in data_name_list){
    print(file_name)
    results=get_data_from_seurat_obj(file_name)

    count.list=results$count
    label.list=results$label
    
    save_processed_data(count.list,label.list, file_name)
    print('finish')
}
