library(splatter)
library(Seurat)
library(Matrix)
library(scrattch.io)

####################################################################################################
save_two_data_to_seurat_obj<-function(reference_count, query_count, reference_cell_lines, query_cell_lines, save_path='./',
                                    save_query_name="seurat_obj", Batch_1='Batch_1', Batch_2='Batch_2'){

     # save the relationships between samples and lables
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

    expr_mat = cbind(reference_count,query_count)
    metadata = rbind(save_reference_frame, save_query_frame)
    batches <- CreateSeuratObject(expr_mat, meta.data = metadata, project = "Seurat3_benchmark")
    batch_list <- SplitObject(batches, split.by = 'batchlb')
    save(batch_list,file=paste0(save_path,save_query_name, ".RData"))

}

##################################################################################


generate<-function(de.facScale, batch.facLoc,index)
{
    batch1_num=2000
    batch2_num=1000
    num_cells = c(batch1_num, batch2_num)
 
    group_prob = c(0.25, 0.25, 0.25, 0.25)

    num_batches = length(num_cells)
    num_clusters = length(group_prob)

    save_path="../../example_data/"


    params <- newSplatParams()
    #batch.facLoc=batch.facLoc
    simulated_data <- splatSimulate(batchCells = num_cells, seed=748101+index, nGenes=10000, group.prob = group_prob,
                                method = "groups",dropout.type="experiment", verbose = FALSE,
                                dropout.shape=-1, dropout.mid=0,  batch.facScale= batch.facLoc,
    de.facScale=de.facScale)

    save_name=paste0("splatter_",batch1_num,"_", batch2_num,"_", num_clusters,"_batch.facScale", batch.facLoc,"_de.facScale",de.facScale,"_10000_",index)
    #tpm_dir_path=paste0("../../process_data/",save_name)
    #tpm_data_path=paste0("../../process_data/",save_name,"/tpm_data/")
    #dir.create(tpm_dir_path)
    #dir.create(tpm_data_path)
    #tpm_label_path=paste0("../../data/",save_name,"/")
    #dir.create(tpm_label_path)
    #print("over create dir\n")

    dat = counts(simulated_data)
    print("over counts")
    # labels
    batch = simulated_data@colData$Batch
    cell = simulated_data@colData$Group
    batch = unlist(lapply(batch,function(x) strtoi(substr(x, 6, 100))))
    cell = unlist(lapply(cell,function(x) strtoi(substr(x, 6, 100))))

    # save simulated data
    idx1 = which(batch == 1)
    idx2 = which(batch == 2)

    # save origin count data
    save_two_data_to_seurat_obj(dat[,idx1], dat[,idx2], cell[idx1],
    cell[idx2], save_path=save_path, save_query_name=save_name, Batch_1='Batch_1', Batch_2='Batch_2')

    # count to TPM
#    dat = apply(dat,2,function(x) (x*10^6)/sum(x))
#    #save data
#    
#    print("begin save tpm  data\n")
#    batch_1_data <- rbind(cell_labels = cell[idx1], dat[,idx1])
#    write.table(batch_1_data, file =paste0(tpm_data_path, "tpm_data_1.csv"), sep = ",", quote = F, col.names = F, row.names = T)
#
#    batch_2_data <- rbind(cell_labels = cell[idx2], dat[,idx2])
#    write.table(batch_2_data, file = paste0(tpm_data_path,"tpm_data_2.csv"), sep = ",", quote = F, col.names = F, row.names = T)
#   
#    #save label
#    batch1_labels = data.frame(cell[idx1])
#    colnames(batch1_labels) = c("train_label")
#
#    batch2_labels = data.frame(cell[idx2])
#    colnames(batch2_labels) = c("test_label")
#
#    write.table(batch1_labels,paste0(tpm_label_path, "Label1.csv"), sep = ",", quote = F, col.names = T, row.names = F)
#    write.table(batch2_labels,paste0(tpm_label_path, "Label2.csv"), sep = ",", quote = F, col.names = T, row.names = F)
    print("over splatter")
}

 facScale=0.2
 facLoc_list=c(0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6) # 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6
 index_list=c(1,2,3,4,5)
for(index in index_list){
for (facLoc in facLoc_list){
    generate(facScale, facLoc, index)
}
}
