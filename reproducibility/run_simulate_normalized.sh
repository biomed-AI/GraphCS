#!bin/sh

cd ..


dataset1=(
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



for filename in ${dataset1[*]}
do
      echo $filename

      python -u train.py  --data  $filename --savepath 2

done

