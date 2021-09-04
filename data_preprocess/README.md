## Data preprocessing

### Preprocessing simulated datasets

```
# filename is the name of the dataset, 'TRUE' represents data is normalized by TPM
Rscript data_preprocess.R filename  TRUE
```


### Preprocessing real datasets

```
# filename is the name of the dataset, data is normalized by Seurat
Rscript data_preprocess.R filename  
```


### Preprocessing big datasets

To save memory and time costs, we saved the normalized big datasets in h5 format. 

```

# Before running the script, you must replace the filename written in normalized_big_data.R. 

Rscript normalized_big_data.R

```







