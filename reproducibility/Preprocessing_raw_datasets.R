library(Seurat)
#library(SeuratDisk)



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


    # save as common genes
    common_gene=intersect(rownames(reference_count), rownames(query_count))
    reference_count=reference_count[common_gene, ]
    query_count=query_count[common_gene, ]
    print(c("common_gene length:", length(common_gene)))

    expr_mat = cbind(reference_count,query_count)
    metadata = rbind(save_reference_frame, save_query_frame)
    batches <- CreateSeuratObject(expr_mat, meta.data = metadata, project = "Seurat3_benchmark")
    batch_list <- SplitObject(batches, split.by = 'batchlb')

    save(batch_list,file=paste0(save_path,save_query_name, ".RData"))
    print("over save seurat")
}


run_save_serurat_data_from_seurat_obj<-function(data_name){
    base="../example_data/" #  the original raw data dir
    save_name=paste0(data_name, "_example")
    data_path=paste0(base, data_name,".RData")
    seurat_obj=load(data_path)
    data=get(seurat_obj)


     train_data=data[[1]]@assays$RNA@counts
    train_cell_type=data[[1]]@meta.data$CellType

    query_data=data[[2]]@assays$RNA@counts
    query_cell_type=data[[2]]@meta.data$CellType

    data[[2]] <- data[[2]][, data[[2]]@meta.data$CellType %in% train_cell_type]
    query_data=data[[2]]@assays$RNA@counts
    query_cell_type=data[[2]]@meta.data$CellType

    save_path="../example_data/"
    save_two_data_to_seurat_obj(train_data, query_data, train_cell_type, query_cell_type, save_path=save_path, save_query_name=save_name)
}


# We use this script to remove the cells that are in the query dataset but not in the reference dataset.
# For the format of the original dataset, you can refer to the datasets in folder example_data in GraphCS project.
data_name="Baron_mouse_Baron_human_origin" # you can replace the other reference-query datasets.
run_save_serurat_data_from_seurat_obj(data_name)
