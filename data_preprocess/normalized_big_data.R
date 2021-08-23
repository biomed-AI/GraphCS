library(Seurat)
library(Matrix)
library(scrattch.io)

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
     #browser()

    save(batch_list,file=paste0(save_path,save_query_name, ".RData"))
    print("over save seurat")
}


seurat_normlized_data<-function(data_name='dataset2', gene_num=5000){

    print(data_name)
    base="../example_data/"
    data=load(paste0(base,data_name,".RData"))
    data=get(data)

    data_name="mouse_brain"

    start_time <- Sys.time()
    for (i in 1:length(x = data)) {
        data[[i]] <- NormalizeData(object = data[[i]], verbose = FALSE)
        data[[i]] <- FindVariableFeatures(object = data[[i]],
                                                selection.method = "vst", nfeatures = gene_num, verbose = FALSE)
      }
    end_time <- Sys.time()
      Seurat3_cca_Training_Time <- as.numeric(difftime(end_time,start_time,units = 'secs'))
        print("noramlized data time")
         print(Seurat3_cca_Training_Time)


    common_gene=intersect(data[[2]]@assays$RNA@var.features,data[[1]]@assays$RNA@var.features)[1:2000]
    norm_train_data=data[[1]]@assays$RNA@data[common_gene, ]
    norm_test_data= data[[2]]@assays$RNA@data[common_gene, ]
    count_train_data=data[[1]]@assays$RNA@counts[common_gene, ]
    count_test_data= data[[2]]@assays$RNA@counts[common_gene, ]

    train_label=data[[1]]@meta.data$CellType
    test_label=data[[2]]@meta.data$CellType

     save_path="../example_data/"
     save_two_data_to_seurat_obj(count_train_data, count_test_data, train_label, test_label, save_path=save_path, save_query_name="mouse_brain")

    # save to h5 type
     outputdir="../process_data/"
     inputdir="../data/"
     dir.create(paste0(outputdir,data_name))
     dir.create(paste0(outputdir,data_name,"/norm_data"))
     dir.create(paste0(inputdir,data_name))
    # #
     file.name=paste0(outputdir,data_name,'/norm_data/')
     save_data<-as(norm_train_data, "sparseMatrix")
     write_dgCMatrix_h5(save_data, cols_are = "sample_names", paste0(file.name, "reference_data_",1,".h5"), ref_name = data_name, gene_ids = NULL)
     save_data<-as(norm_test_data, "sparseMatrix")
     write_dgCMatrix_h5(save_data, cols_are = "sample_names", paste0(file.name, "reference_data_",2,".h5"), ref_name = data_name, gene_ids = NULL)
    #
     file.name=paste0(inputdir,data_name,'/Label',1,'.csv')
     write.csv(train_label,file=file.name,quote=F, row.names = F)
     file.name=paste0(inputdir,data_name,'/Label',2,'.csv')
     write.csv(test_label,file=file.name,quote=F, row.names = F)

}

seurat_normlized_data("mouse_brain_origin_data")
