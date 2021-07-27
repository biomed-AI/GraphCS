library(Seurat)
source('data_preprocess_utility.R')
base="../example_data/"


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


file_name="example"
results=get_data_from_seurat_obj_with_unique(file_name)
count.list=results$count
label.list=results$label
save_processed_data(count.list,label.list, file_name)


