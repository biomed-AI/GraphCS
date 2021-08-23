#!bin/sh


#####################################
# generate simulated data as the following commands,
# which save  simulated data with TPM format for GraphCS and raw simulated data for competing methods, this process is necessary

cd simulated_data
Rscript splatter.R
cd ..

##########################################

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
   
      # generate hvg gene 
      cd data_preprocess
      Rscript data_preprocess.R $filename
      cd ..

      cd graph_construction
      python graph_for_simulated_data.py --name  $filename
      cd ..

      python -u train.py  --data  $filename --savepath 2

done

