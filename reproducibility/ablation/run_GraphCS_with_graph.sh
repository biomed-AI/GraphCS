#!bin/sh

count=0
num=1
txt=".txt"
contact="_"
data_path="data/"

output_name="different_graph_full_recored.out"
save_result="result.out"


datasets=(

'Baron_mouse_combination'
'Baron_mouse_segerstolpe'
'Baron_mouse_Baron_human'
'Baron_human_Baron_mouse'
        )


graphnames=(
   "conos"
    "ccamnn"
    "annoy"
   "umap"
    "sklearnknn"
    "cosine"
)


for filename in ${datasets[*]}
do
for graph_name in ${graphnames[*]}
do

   cp $data_path$filename$contact$graph_name$txt  $data_path$filename$txt

    echo "*************************************">>$output_name
   echo "$data_path$filename$contact$graph_name$txt" >>$output_name



   python -u train.py   --data $filename  --alias_name $graph_name --result $save_result   >>$output_name


    count=`expr $count + 1`
    if [ $count -eq $num ]
      then
          echo $count
          count=0
      wait
    fi

   echo "###########################################################################">>$output_name

done
 count=`expr $count + 1`
    if [ $count -eq $num ]
      then
          echo $count
          count=0
      wait
    fi
done


