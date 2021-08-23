read_data<-function(data_name='dataset2'){

    print(data_name)
    base='../../example_data/'
    data=load(paste0(base,data_name,".RData"))
    data=get(data)
    
    train_data=data[[1]]@assays$RNA@counts
    test_data=data[[2]]@assays$RNA@counts
    train_label=as.character(data[[1]]@meta.data$CellType)
    test_label=as.character(data[[2]]@meta.data$CellType)

    train_gene= rownames(train_data)
    test_gene= rownames(test_data)
    common_gene=intersect(train_gene, test_gene)
    train_data=train_data[common_gene,]
    test_data = test_data[common_gene,]


    train_row_names=c(paste0("cell_",1:length(train_label)))
    test_row_names=c(paste0("cell_",(1+length(train_label)):(length(test_label)+length(train_label))))

    colnames(train_data)=train_row_names
    colnames(test_data)=test_row_names
    train_data = as.matrix(train_data)
    test_data = as.matrix(test_data)

    train_label_matrix=as.matrix(as.character(train_label))
    rownames(train_label_matrix)=train_row_names
    colnames(train_label_matrix)="cell.type"

    test_label_matrix=as.matrix(as.character(test_label))
    rownames(test_label_matrix)=test_row_names
    colnames(test_label_matrix)="cell.type"
    results=list(train_data=train_data, test_data=test_data, train_label=train_label_matrix, test_label=test_label_matrix)
}
preproc<-function(DATA, nmcs=50, dims=1:30){
    DATA <- NormalizeData(DATA)
    DATA <- FindVariableFeatures(DATA)
    DATA <- ScaleData(DATA)
    DATA <- RunMCA(DATA, nmcs = nmcs)
    DATA <- RunPCA(DATA)
    DATA <- RunUMAP(DATA, dims = dims)
    DATA <- RunTSNE(DATA, dims = dims)
    return(DATA)
}
cell2cell<-function(train, test, dims=1:50, feat=200){
    train_cell_gs <- GetCellGeneSet(train, dims = dims, n.features = feat)
    HGT_train_cell_gs <- RunCellHGT(test, pathways = train_cell_gs, dims = dims)
    train_cell_gs_match <- rownames(HGT_train_cell_gs)[apply(HGT_train_cell_gs, 2, which.max)]
    train_cell_gs_prediction <- train$cell.type[train_cell_gs_match]
    train_cell_gs_prediction <- as.character(train_cell_gs_prediction)
    train_cell_gs_prediction_signif <- ifelse(apply(HGT_train_cell_gs, 2, max)>2, train_cell_gs_prediction, "unassigned")
    result=list('c2c_pred'=train_cell_gs_prediction,
            'gt'=test$cell.type)
}
group2cell<-function(train, test, dims=1:50, feat=200){
    train_group_gs <- GetGroupGeneSet(train, dims = dims, n.features = feat, group.by = "cell.type")
    HGT_train_group_gs <- RunCellHGT(test, pathways = train_group_gs, dims = dims)
    train_group_gs_prediction <- rownames(HGT_train_group_gs)[apply(HGT_train_group_gs, 2, which.max)]
    train_group_gs_prediction_signif <- ifelse(apply(HGT_train_group_gs, 2, max)>2, yes = train_group_gs_prediction, "unassigned")
    result=list('g2c_pred'=train_group_gs_prediction,
            'gt'=test$cell.type)
}
evaluate<-function(result){
    p<-as.character(result[[1]])
    g<-as.character(result[[2]])
    acc=sum(p==g)/length(g)
}
