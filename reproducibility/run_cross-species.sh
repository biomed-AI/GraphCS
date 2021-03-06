#!bin/sh
cd ..
dataset2=(
'Baron_mouse_Baron_human'
'Baron_mouse_segerstolpe'
'Baron_human_Baron_mouse'
'Baron_mouse_combination'
)

for filename in ${dataset2[*]}
do
      echo $filename
      cd data_preprocess
      Rscript data_preprocess.R $filename
      cd ..
      cd graph_construction
      python graph.py --name  $filename
      cd ..
      python -u train.py  --data  $filename --savepath 1

done
