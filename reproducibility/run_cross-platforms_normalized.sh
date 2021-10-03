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
	  if [ $filename = "mouse_retina" ];then
            python -u train.py  --data  $filename --savepath 0 --batch 4096
        else
           python -u train.py  --data  $filename --savepath 0 --batch 1024
       fi

	  
done

python -u train.py  --data mouse_brain  --savepath 0 --batch 4096 --patience 5






