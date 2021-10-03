#!bin/sh
cd ..
dataset1=(
'pbmcsca_10x_Chromium_CEL-Seq2'
'pbmcsca_10x_Chromium_Drop-seq'
'pbmcsca_10x_Chromium_inDrops'
'pbmcsca_10x_Chromium_Seq_Well'
'pbmcsca_10x_Chromium_Smart-seq2'
'mouse_retina'
)

for filename in ${dataset1[*]}
do
      echo $filename
      cd data_preprocess
      Rscript data_preprocess.R $filename
      cd ..      

      cd graph_construction
      python graph.py --name  $filename
      cd ..

      if [ $filename = "mouse_retina" ];then
            python -u train.py  --data  $filename --savepath 0 --batch 4096
      else
            python -u train.py  --data  $filename --savepath 0 --batch 1024
      fi

done

bigdata='mouse_brain'
# save bigdata as h5 format
 cd data_preprocess
 Rscript normalized_big_data.R
 cd ..

cd graph_construction
python graph_for_big_data.py --name  $bigdata
cd ..
python -u train.py  --data $bigdata  --savepath 0 --batch 4096 --patience 5






