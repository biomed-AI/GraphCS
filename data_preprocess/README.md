## Data preprocessing

### Preprocessing simulated datasets

```
# filename is the name of the dataset, 'TRUE' represents that dataset is normalized by TPM

Rscript data_preprocess.R filename  TRUE
```


### Preprocessing real datasets

```
# filename is the name of the dataset, the dataset is normalized by Seurat

Rscript data_preprocess.R filename  
```


### Preprocessing big datasets (> 800000 cells)

To save memory and time costs, we normalized big datasets by Seurat and saved them in h5 format.  

```

# You need to replace the filename if you run your own datasets. The filename  was written in the script normalized_big_data.R. 

Rscript normalized_big_data.R

```







