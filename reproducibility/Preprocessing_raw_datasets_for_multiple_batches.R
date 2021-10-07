library(Seurat)
#library(SeuratDisk)

# Note, all datasets used in this paper have been Pre-processed, if you used the datasets provided by us, you don't need to execute this script.

####################################################################################################
save_two_data_to_seurat_obj<-function(reference_count, query_count, reference_cell_lines, query_cell_lines, save_path='./',
                                    save_query_name="seurat_obj", Batch_1='Batch_1', Batch_2='Batch_2'){

     #######
    # format -> reference_count: genes x cells
    ######

     # save the relationships between samples and lables

      # reference_sample = colnames(reference_count)
      # colnames(reference_count)=1:length(reference_sample)

      reference_sample = colnames(reference_count)
      query_sample = colnames(query_count)
      reference_frame=data.frame(reference_cell_lines, row.names=reference_sample)
      reference_frame[[1]]=as.character(reference_frame[[1]])
      query_frame=data.frame(query_cell_lines, row.names=query_sample)
      query_frame[[1]]=as.character(query_frame[[1]])
      names(reference_frame) <-NULL
      names(query_frame) <-NULL

     #format genes x cells
    # count contains cell and gene names
    # label are data.frame format, row.names=sample names
    save_reference_frame=reference_frame
    save_query_frame=query_frame
    names(save_reference_frame) <-"CellType"
    names(save_query_frame) <-"CellType"
    save_reference_frame$batchlb <- Batch_1
    save_query_frame$batchlb <- Batch_2

    save_reference_frame$batchinfo <- "Batch_1"
    save_query_frame$batchinfo <- "Batch_2"


    # save as common genes

    common_gene=intersect(rownames(reference_count), rownames(query_count))
    reference_count=reference_count[common_gene, ]
    query_count=query_count[common_gene, ]
    print(c("common_gene length:", length(common_gene)))

    expr_mat = cbind(reference_count,query_count)
    metadata = rbind(save_reference_frame, save_query_frame)
    batches <- CreateSeuratObject(expr_mat, meta.data = metadata, project = "Seurat3_benchmark")
    batch_list <- SplitObject(batches, split.by = 'batchinfo')
     #browser()

    save(batch_list,file=paste0(save_path,save_query_name, ".RData"))
    print("over save seurat")
}



run_save_serurat_data_from_seurat_obj<-function(data_name, base){

    save_name=paste0(data_name, "_example")
    data_path=paste0(base, data_name,".RData")
    seurat_obj=load(data_path)
    data=get(seurat_obj)

    train_data=data[[1]]@assays$RNA@counts
    train_cell_type=data[[1]]@meta.data$CellType
    train_batchifo=data[[1]]@meta.data$batchlb


    data[[2]] <- data[[2]][, data[[2]]@meta.data$CellType %in% train_cell_type]
    query.list=data[[2]]@assays$RNA@counts
    query.meta.data.cell_type=data[[2]]@meta.data$CellType
    query.meta.data.batchinfo=data[[2]]@meta.data$batchlb
    # multiple batches in query dataset
    if (length(data) > 2){
        for ( index in 3:length(data))
            {
                print(index)
                data[[index]] <- data[[index]][, data[[index]]@meta.data$CellType %in% train_cell_type]
                query.list=cbind(query.list, data[[index]]@assays$RNA@counts)
                query.meta.data.cell_type=c(query.meta.data.cell_type, data[[index]]@meta.data$CellType)
                query.meta.data.batchinfo=c(query.meta.data.batchinfo, data[[index]]@meta.data$batchlb)
            }
    }


    save_path="./example_data/"
    save_two_data_to_seurat_obj(train_data, query.list, train_cell_type, query.meta.data.cell_type,
    save_path=save_path, save_query_name=save_name, Batch_1=train_batchifo, Batch_2=query.meta.data.batchinfo)
}


# Note, all datasets used in this paper have been Pre-processed, if you used the datasets provided by us, you don't need to execute this script.

# We use this script to remove the cells that are in the query dataset but not in the reference dataset.
# For the format of the original dataset, you can refer to the datasets in folder example_data in GraphCS project.
data_name_list=c(
"Baron_mouse_Mutaro_b2_Wang_b4_Xin_b5_Segerstolpe_b3_Baron_b1_with_mutilply_batches_info",
    "Baron_mouse_segerstolpe_example"
)

# please make dir raw_data in current dir and put the raw paired reference_query data into raw_data
base="./raw_data/"

for (data_name in data_name_list){
    print(data_name)
    run_save_serurat_data_from_seurat_obj(data_name, base)
}

#  We use this script to preprocess the raw data downloaded from the original paper or GEO and feed the preprocessed count data into GraphCS.
# Note, all datasets used in this paper have been Pre-processed, if you used the datasets provided by us, you don't need to execute this script.