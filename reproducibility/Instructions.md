
# Requirements
- Operating system 

  **Ubuntu 16.04.7 LTS** 
  
  Kernel version4.4.0-189-generic
  
- **BBKNN 1.4.0**
- **Scanpy 1.8.1**
- leidenalg 0.8.3
- CUDA 10.2.89
- Python 3.7.9
- Pytorch 1.7.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)
- scrattch.io 0.1.0
- Seurat 3.1.5


**Note: All results have been obtained under Ubuntu 16.04.7 LTS. We have also noticed that the BBKNN 1.4.0 
used in our method has constructed approximate nearest neighbors based on the Annoy algorithm, 
which slightly changes under different operating environments.**


# Overall 

## Directory structure of folder reproducibility
```
.
├── ablation                  # Ablation Experiments 
├── competing_methods         # Including all competing methods
├── figures                   # Saving diagrams used in fig.2c 
├── simulated_data            # Splatter script for generating simulated data 
├── umap_visalization         # Generate the Umap embedding and plot the Umap graph
├── Fig2a.ipynb               # Show the performance of GraphCS on simulated data and plot Fig2a
├── Fig2b.ipynb               # Show the performance of GraphCS on cross-platform data and plot Fig2b
├── Fig2c.ipynb               # Plot the Fig2c
├── Fig2d.ipynb               # Show the performance of GraphCS on cross-species data and plot Fig2d
├── Fig3umap.ipynb            # Plot the Umap diagram on cross-species data
├── Fig4ablation.ipynb        # Generate ablation results and plot the Fig4
├── cross-platforms.csv       # Predicted results of all methods on cross-platform data
├── cross-species.csv         # Predicted results of all methods on cross-species data
├── simulate.csv              # Predicted results of all methods on simulated data
├── simulate-std.csv          # Save std results for all methods on simulated data
├── run_cross-platforms.sh    # Bash script for running GraphCS on raw cross-platform data
├── run_cross-species.sh      # Bash script for running GraphCS on raw cross-species data
├── run_simulate.sh           # Bash script for running GraphCS on raw simulated on data
├── Preprocessing_raw_datasets.R        # R script for removing the cell types that are in query dataset but not in the reference dataset.
├── run_cross-platforms_normalized.sh   # Bash script for running GraphCS on preprocessed cross-platforms data
├── run_cross-species_normalized.sh     # Bash script for running GraphCS on  preprocessed cross-species data
├── run_simulate_normalized.sh          # Bash script for running GraphCS on preprocessed simulated data
└── create_csv.py                       # For creating null `cross-platforms.csv, cross-species.csv, simulate.csv, and simulate-std.csv` to save results predicted by all methods

```

  
# Download datasets
**Note: If you want to run the bash scripts of GraphCS (such as [run_cross-platforms.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-platforms.sh) or 
[run_cross-platforms_normalized.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-platforms_normalized.sh))  or R scripts of competing methods (such as [cellid.R](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/competing_methods/cellid.R)),  you must download the corresponding datasets included in scripts.  Thus, we suggest downloading all datasets as following links before reproducing results.
 On the other hand, 
you can revise dataset names in scripts to only run the partial datasets that have been downloaded.**
 
 
## Raw datasets:
You can download raw datasets from
 [here](https://drive.google.com/drive/folders/1ST0T90HcxCKuxOTmOvqCI-IyE2IY6YvM?usp=sharingv), 
and place them into folder [example_data](https://github.com/biomed-AI/GraphCS/tree/main/example_data).  

## Preprocessed datasets:
You can download the preprocessed datasets and the graph txt files from
 [here](https://drive.google.com/drive/folders/1ST0T90HcxCKuxOTmOvqCI-IyE2IY6YvM?usp=sharing),
 and place them into the folder [data](https://github.com/biomed-AI/GraphCS/tree/main/data). 


## Other links
The above datasets also can be downloaded from Synapse with ID [syn26147749](https://www.synapse.org/#!Synapse:syn26147749/files/)


# Reproduce results

## GraphCS (our method)
You can obtain the accuracy of GraphCS on simulated, cross-platform,
 and cross-species datasets by running the following commands.
 
**Note: You can choose one of the following three strategies to obtain results of GraphCS on real datasets.**  

### Running GraphCS on preprocessed datasets
[run_cross-platforms_normalized.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-platforms_normalized.sh) contains commands to run all preprocessed 
cross-platform datasets by GraphCS, where cross-platform datasets had been normalized by Seurat and
 the corresponding cell graphs had been constructed by BBKNN. Same for other scripts in this section. 
  
```
 bash run_cross-platforms_normalized.sh 
 bash run_cross-species_normalized.sh
 bash run_simulate_normalized.sh
```

or 

### Running GraphCS on raw datasets
[run_cross-platforms.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-platforms.sh) contains
 commands to run all raw cross-platform datasets by GraphCS,
 where cross-platform datasets are needed to be normalized by Seurat and the corresponding cell
  graphs are  needed to be constructed
 by BBKNN. All these procedures had been coded in  `run_cross-platforms.sh`.  Same for other scripts in this section. 

```
 bash run_cross-platforms.sh 
 bash run_cross-species.sh
 bash run_simulate.sh
```


or 

### Running GraphCS by a single command

You can follow the step as described in `Run the demo` at [README.md](https://github.com/biomed-AI/GraphCS)
to evaluate the performance of GraphCS on simulated, cross-platform, and cross-species datasets.
Note: you can refer to the parameters used in scripts
 ([run_cross-platforms.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-platforms.sh),
  [run_cross-species.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_cross-species.sh), 
  [run_simulate.sh](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/run_simulate.sh)) 
 when running each dataset separately. 





### Ablation experiments for GraphCS

We conduct the ablation experiments in folder `ablation`. You can follow the `ablation_instructions.txt` 
in folder `ablation`
 or the instructions in Fig4ablation.ipynb to obtain results of ablation experiments.


### Plot umap on cross-species datasets
You can get the embeddings for all methods on cross-species in folder `umap_visalization` by running
following commands:


```
cd umap_visalization
# get embedding for GraphCS
python get_GraphCS_embedding.py

# get embedding for raw data
python get_raw_data_embedding.py

# get embedding for Seurat-CCA
Rscript get_seurat_embedding.R

# The embedding of scGCN will be saved in folder scGCN when running the scGCN on cross-species 
datasets. We have added the saving codes into train.py of scGCN to save the embeddings in scGCN.
 You can run scGCN as described in `Reproduce results` to obtain embeddings.  

# plot the Umap graph
python plot_umap.py
```


## Running competing methods
 All competing methods were saved in folder `competing_methods`. You can obtain results 
 of all competing methods by following commands:
 
Note: Competing methods and GraphCS used the same raw datasets
 that were placed in folder [example_data](https://github.com/biomed-AI/GraphCS/tree/main/example_data).

 
```
cd competing_methods
```

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
(in folder `competing_methods`) to convert the ".RData" type of datasets to ".h5ad" type.

```
Rscript convert_between_scanpy_seurat.R

python scanvi.py

python scNym.py
```

#### scGCN:
```
 cd scGCN/scGCN
 bash run_scGCN_all.sh 
```


### Run GraphCS Notebooks :

[Fig2a.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig2a.ipynb)              
[Fig2b.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig2b.ipynb)           
[Fig2c.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig2c.ipynb)               
[Fig2d.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig2d.ipynb)              
[Fig3umap.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig3umap.ipynb)            
[Fig4ablation.ipynb](https://github.com/biomed-AI/GraphCS/blob/main/reproducibility/Fig4ablation.ipynb)        

