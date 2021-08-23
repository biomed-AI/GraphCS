#!bin/sh

count=0
num=1
txt=".txt"
contact="_"
data_path="data/"

output_name="ablation_cross_species_datasets.out"
save_result="result.out"

datasets=(

'Baron_mouse_combination'
'Baron_mouse_segerstolpe'
'Baron_mouse_Baron_human'
'Baron_human_Baron_mouse'
       )



for filename in ${datasets[*]}
do

    python -u train.py   --data $filename  --alias_name "GraphCS" --result $save_result  >>$output_name

    count=`expr $count + 1`
    if [ $count -eq $num ]
      then
          echo $count
          count=0
      wait
    fi


     python -u train.py   --data $filename  --vat_lr "0"   --alias_name "VAT-" --result $save_result  >>$output_name

    count=`expr $count + 1`
    if [ $count -eq $num ]
      then
          echo $count
          count=0
      wait
    fi


   python -u train.py   --data $filename --origin_data "T"  --vat_lr "0"  --alias_name "GBP-" --result $save_result  >>$output_name

    count=`expr $count + 1`
    if [ $count -eq $num ]
      then
          echo $count
          count=0
      wait
    fi


done

