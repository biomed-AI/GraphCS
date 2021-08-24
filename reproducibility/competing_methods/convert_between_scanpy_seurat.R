library(SeuratDisk)
library(Seurat)

if (!dir.exists('./dataset')){
  dir.create('./dataset')
  dir.create('./dataset/train')
  dir.create('./dataset/test')
}

updatedata<-function(name){
	base="../../example_data/"
	save_base="./dataset/"
	
	print(paste0('loading ',name))
	data=get(load(paste0(base, name, ".RData")))
	
	print(paste0('saving ',name))
	train=paste0(save_base,'/train/',name,'.h5Seurat')
	test=paste0(save_base,'/test/',name,'.h5Seurat')
	
	SaveH5Seurat(data[[1]], filename=train)
	SaveH5Seurat(data[[2]], filename=test)
	
	print(paste0('converting ',name))
	Convert( train, dest = "h5ad")
	Convert( test, dest = "h5ad")
}
sim=c()
for(i in 1:8){
    sim=c(sim,paste0("splatter_2000_1000_4_batch.facScale",0.2*i,"_de.facScale0.2_10000_",1:5))
}
filename=c(
	'pbmcsca_10x_Chromium_CEL-Seq2',
	'pbmcsca_10x_Chromium_Drop-seq',
	'pbmcsca_10x_Chromium_inDrops',
	'pbmcsca_10x_Chromium_Seq_Well',
	'pbmcsca_10x_Chromium_Smart-seq2',
	'mouse_retina',
	'Baron_mouse_combination',
	'Baron_mouse_segerstolpe',
	'Baron_mouse_Baron_human',
	'Baron_human_Baron_mouse',
	'mouse_brain',
  sim
)
fail=c()
for(i in filename){
  fit<-try(updatedata(i))
  if('try-error' %in% class(fit)){
    next
  }else{
    fail=c(fail,i)
  }
}
if(length(fail)>0){
  print('The following datasets failed to convert:')
  print(fail)
}
