A Robust and Scalable Graph Neural Network for Accurate Single Cell Classification
============


![(Variational) gcn](Fig._1.jpg)


## Requirements
- CUDA 10.1.243
- python 3.6.10
- pytorch 1.4.0
- GCC 5.4.0
- [cnpy](https://github.com/rogersce/cnpy)
- [swig-4.0.1](https://github.com/swig/swig)

## Compilation
```bash
make
```

## Datasets

The `data` folder includes example dataset. 


```
- pre_process raw data: cd data_preprocess; Rscript data_preprocess.R example 
- construct graph: cd graph_construction; python graph.py  --name example
- python -u train.py --data example
```


### or


## Runing the code
```
 if you installed git lfs  and downloaded the processed example data in folder data,
 you can run the following command directly. 
```

```
python -u train.py --data example
```



## Runing your own data

- pre_process raw data: cd data_preprocess; Rscript data_preprocess.R  your_data_name   
- construct graph: cd graph_construction; python graph.py  --name your_data_name
- python -u train.py --data your_data_name 



## Cite
Please cite our paper if you use this code in your own work:

```
@article{zengys,
  title={A Robust and Scalable Graph Neural Networkfor Accurate Single Cell Classification},
  author={Yuansong Zeng, Xiang Zhou, Zixiang Pan, Yutong Lu1, Yuedong Yang},
  journal={xxx},
  year={2021}
}
```

