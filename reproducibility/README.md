
## Requirements
- Operating system 

  Ubuntu 16.04.7 LTS 
  
  Kernel version4.4.0-189-generic

- CUDA 10.2.89
- Python 3.7.9
- Pytorch 1.7.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)
- BBKNN 1.4.0
- leidenalg 0.8.3
- Scanpy 1.8.1
- scrattch.io 0.1.0
- Seurat 3.1.5


## Overall 

### Directory structure of folder reproducibility
```
.
├── ablation                  # ablation Experiments 
├── competing_methods         # Including all competing methods
├── figures                   # Saving graphs in fig.2c 
├── simulated_data            # Splatter script for generating simulated data 
├── umap_visalization         # Generate the Umap embedding and plot the Umap graph
├── Fig2a.ipynb               # Show the performance of GraphCS on simulated data and plot Fig2a
├── Fig2b.ipynb               # Show the performance of GraphCS on cross-platform data and plot Fig2b
├── Fig2c.ipynb               # Plot the Fig2c
├── Fig2d.ipynb               # Show the performance of GraphCS on cross-species data and plot Fig2d
├── Fig3umap.ipynb            # Plot the Umap graph on cross-species data
├── Fig4ablation.ipynb        # Generate the ablation results and plot the Fig4
├── cross-platforms.csv       # Save the results for all methods on cross-platform data
├── cross-species.csv         # Save the results for all methods on cross-species data
├── simulate.csv              # Save the results for all methods on simulated data
├── simulate-std.csv          # Save the std results for all methods on simulated data
├── run_cross-platforms.sh    # Bash script for GraphCS to run all raw cross-platform data
├── run_cross-species.sh      # Bash script for GraphCS to run all raw cross-species data
├── run_simulate.sh           # Bash script for GraphCS to run all raw simulated on data
├── Preprocessing_raw_datasets.R        # R script for removing the cells that are in query dataset but not in the reference dataset.
├── run_cross-platforms_normalized.sh   # Bash script for GraphCS to run all  preprocessed cross-platforms data
├── run_cross-species_normalized.sh     # Bash script for GraphCS to run all  preprocessed cross-species data
├── run_simulate_normalized.sh          # Bash script for GraphCS to run all  preprocessed simulated data
└── create_csv.py                       # For creating a null CSV files to save the results produced by all methods

```
### Brief introduction
Before reproducing the results, you should download the datasets and put them into the corresponding folder as described in `Datasets`.
For our method GraphCS, you can run on preprocessed datasets or raw datasets.  Other methods shared the raw datasets with GraphCS in folder `example_data` and could be run as described in 
`Reproduce results`.  The results for all methods on simulated, cross-platform, and cross-species will be saved in  `simulate.csv, cross-platforms.csv and cross-species.csv`.
In addition, you can get the graph used in this manuscript by running the jupyter scripts as described in  `Reproduce results`.   
We have shown the results predicted by our method in the jupyter scripts (ipynb files), you can see the results
directly by open them on the current Github page. 


## Datasets
### raw datasets:
You can downloaded raw datasets from
 [website](https://drive.google.com/drive/folders/1nNPUaJ91upcGvg08AZlIaebNCxtyAbAd?usp=sharing), 
and put them into folder `example_data` (In the parent directory of the current directory).  

### preprocessed datasets:
You can download the preprocessed datasets and the graph txt files from
 [website](https://drive.google.com/file/d/1H0XXXAlpzIS9GOMC3S-aP4Dq7JS-ehM1/view?usp=sharing), 
and put them into the folder `data`  (In the parent directory of the current directory). 


# Reproduce results

## GraphCS (our method)
you can get the accuracy of GraphCS on simulated, cross-platform,
 and cross-species datasets by run the following commands in turn.
 
### run on preprocessed  datasets 
`run_cross-platforms_normalized.sh` contains the commands to run all preprocessed 
cross-platform datasets by GraphCS, where cross-platform datasets were normalized by Seurat and
 the corresponding cell graphs constructed by BBKNN. Same for other scripts. 
  
```
 bash run_cross-platforms_normalized.sh 
 bash run_cross-species_normalized.sh
 bash run_simulate_normalized.sh
```

or 

### run on raw datasets 
`run_cross-platforms.sh` contains the commands to run all raw cross-platform datasets by GraphCS,
 where cross-platform datasets needed to be normalized by Seurat and the corresponding cell
  graphs need to be constructed
 by BBKNN. All procedures were coded in  `run_cross-platforms.sh`.  Same for other scripts. 

```
 bash run_cross-platforms.sh 
 bash run_cross-species.sh
 bash run_simulate.sh
```


### ablation experiments for GraphCS

We conduct the ablation experiments in folder `ablation`. You can follow the `readme.txt` 
in folder `ablation`
 or the instructions in Fig4ablation.ipynb to get ablation results.


### plot umap on cross-species datasets
You can get the embeddings for all methods on cross-species in folder `umap_visalization` by running
following commands:

cd `umap_visalization`:
```
# get embedding for GraphCS
python get_GraphCS_embedding.py

# get embedding for raw data
python get_raw_data_embedding.py

# get embedding for Seurat-CCA
Rscript get_seurat_embedding.R

# The embedding of scGCN will be saved in folder scGCN when running the scGCN on cross-species 
datasets. We have added the saving codes into train.py of scGCN to save the embedding in scGCN.
 You can run
scGCN as described in `Reproduce results` to embeddings.  

# plot the Umap graph
python plot_umap.py
```


## Other methods
 All competing methods were saved in folder `competing_methods`. You can get the results 
 for other competing methods by following commands directly:
 
Note: All datasets used in competing methods were saved in folder `example_data`,
 which is also the dataset folder for GraphCS.
 
cd competing_methods

####  Cellid:
```
Rscript cellid.R
```

#### scclassify:
```
Rscript scclassify.R
```

#### SingleCellNet, SingleR, scmap, Seurat-CCA, and Seurat-PCA:
```
Rscript SingleCellNet_SingleR_scmap_Seurat.R 
```

####  scanvi and scNym:

Before running scanvi and scNym, you should run the script `convert_between_scanpy_seurat.R`
to convert the "RData" type of datasets into the "h5ad" type, which is needed by scanvi and scNym.

```
Rscript convert_between_scanpy_seurat.R

python scanvi.py

python scNym.py
```

#### scGCN:
```
 cd competing_methods/scGCN/scGCN
 bash run_scGCN_all.sh 
```