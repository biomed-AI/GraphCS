library(Seurat)
library(conos)
library(pagoda2)
library(SingleCellExperiment)

base="../../../example_data/"

get_data_from_seurat_obj_with_unique<-function(data_name){

    data_path=paste0(base, data_name,".RData")
    seurat_obj=load(data_path)
    data=get(seurat_obj)


     train_data=data[[1]]@assays$RNA@counts
    test_data=data[[2]]@assays$RNA@counts

    train_label=data[[1]]@meta.data$CellType
    test_label=data[[2]]@meta.data$CellType

    colnames(train_data)=train_label
    colnames(test_data)=test_label

    train_row_names=c(1:length(train_label))
    test_row_names=c((1+length(train_label)):(length(test_label)+length(train_label)))
    colnames(train_data)=train_row_names
    colnames(test_data)=test_row_names

     count.list=list(train_data, test_data)
     results=list(count=count.list)
     return (results)
}

get_conos_graph<-function(data_name, save_data_name){


    results=get_data_from_seurat_obj_with_unique(data_name)

    # formart genes x cells
    panel <- results$count
    print(any(duplicated(unlist(lapply(panel,colnames)))))

    panel.preprocessed <- lapply(panel, basicSeuratProc)

    names(panel.preprocessed)<-c("train", "test")
    con <- Conos$new(panel.preprocessed, n.cores=8)

    con$buildGraph(k=4, k.self=1,space='CCA', ncomps=30, n.odgenes=2000, matching.method='mNN', metric='angular', score.component.variance=TRUE, verbose=TRUE)
   if (!dir.exists('conos_origin_graph')){dir.create('conos_origin_graph')} 
    exchange_dir <- paste0("./conos_origin_graph/", save_data_name, "/")
    dir.create(exchange_dir)

    saveConosForScanPy(con, output.path=exchange_dir, verbose=TRUE)
}


run_pipeline<-function(){

   data_name_list=c(
             
'Baron_mouse_combination',
'Baron_mouse_segerstolpe',
'Baron_mouse_Baron_human',
'Baron_human_Baron_mouse'


     )


    
    for( file_name in data_name_list)
    {
        get_conos_graph(file_name, file_name)
    }
}


run_pipeline()
