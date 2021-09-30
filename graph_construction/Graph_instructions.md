## Consturcing cell graph by BBKNN 

### Consturcing cell graph based on simulated datasets

Since simulated data is normalized by TPM and saved in folder `tpm_data`, we construct the cell graph from TPM format data using the following command:

```
# filename is the name of dataset
python graph_for_simulated_data.py --name filename
```


### Consturcing cell graph based on real datasets

This is the recommended format for generating cell graphs from real datasets
```
# filename is the name of dataset
python graph.py --name filename
```


### Consturcing cell graph based on big datasets

# parameter large_data represents that the input is big data mouse_brain

Since the normalized big datasets were saved in h5 format, we read data from the `.h5` format to generate the cell graph. 

```
python graph.py --name filename  --large_data T

```

**Note: edge_ratio is used for trimming the inter-edges between batches. In default, we set edge_ratio=2. We also listed detailed parameters used in this paper in supplementary table S2.  **





