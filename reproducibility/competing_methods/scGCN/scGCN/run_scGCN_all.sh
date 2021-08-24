#!bin/sh

# normalized datasets
Rscript data_preprocess.R

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
      python train.py  --data_name  $filename --savepath 0

done
dataset2=(

'Baron_mouse_combination'
'Baron_mouse_segerstolpe'
'Baron_mouse_Baron_human'
'Baron_human_Baron_mouse'
)
for filename in ${dataset2[*]}
do
      echo $filename
      python train.py  --data_name  $filename --savepath 1

done
dataset3=(
"splatter_2000_1000_4_batch.facScale0.2_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale0.4_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale0.6_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale0.8_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale1_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale1.2_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale1.4_de.facScale0.2_10000_1"
"splatter_2000_1000_4_batch.facScale1.6_de.facScale0.2_10000_1"

"splatter_2000_1000_4_batch.facScale0.2_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale0.4_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale0.6_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale0.8_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale1_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale1.2_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale1.4_de.facScale0.2_10000_2"
"splatter_2000_1000_4_batch.facScale1.6_de.facScale0.2_10000_2"

"splatter_2000_1000_4_batch.facScale0.2_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale0.4_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale0.6_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale0.8_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale1_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale1.2_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale1.4_de.facScale0.2_10000_3"
"splatter_2000_1000_4_batch.facScale1.6_de.facScale0.2_10000_3"

"splatter_2000_1000_4_batch.facScale0.2_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale0.4_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale0.6_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale0.8_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale1_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale1.2_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale1.4_de.facScale0.2_10000_4"
"splatter_2000_1000_4_batch.facScale1.6_de.facScale0.2_10000_4"

"splatter_2000_1000_4_batch.facScale0.2_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale0.4_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale0.6_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale0.8_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale1_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale1.2_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale1.4_de.facScale0.2_10000_5"
"splatter_2000_1000_4_batch.facScale1.6_de.facScale0.2_10000_5"
)
for filename in ${dataset3[*]}
do
      echo $filename
      python train.py  --data_name  $filename --savepath 2

done
python txt2csv.py



