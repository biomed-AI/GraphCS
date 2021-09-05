# Execute Step
#1 copy the preprocessed cross-species datasets 
mkdir data

cp  -rf ../../data/Baron_mouse_combination data/
cp  ../../data/Baron_mouse_combination_feat.npy data/
cp  ../../data/Baron_mouse_combination.txt data/

cp -rf ../../data/Baron_mouse_segerstolpe data/
cp  ../../data/Baron_mouse_segerstolpe_feat.npy data/
cp  ../../data/Baron_mouse_segerstolpe.txt data/

cp -rf ../../data/Baron_mouse_Baron_human data/
cp  ../../data/Baron_mouse_Baron_human_feat.npy data/
cp  ../../data/Baron_mouse_Baron_human.txt data/

cp -rf ../../data/Baron_human_Baron_mouse data/
cp  ../../data/Baron_human_Baron_mouse_feat.npy data/
cp  ../../data/Baron_human_Baron_mouse.txt data/


#2 get the ablation results
bash run_ablation.sh 


# generate different cell graphs
python knn_cosine_umap_annoy.py

cd cca_mnn
Rscript generate.R 
python read_cca_mnn_graph.py

cd  conos_graph
Rscript run_conos.R 
python get_graph_from_conos.py


#4 run GraphCS on different graphs
bash run_GraphCS_with_graph.sh

#5 plot the results
python read_result.py 
