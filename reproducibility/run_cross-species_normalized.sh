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
      python -u train.py  --data  $filename --savepath 1

done
