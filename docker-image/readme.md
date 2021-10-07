**Note: If you want to run GraphCS with docker version, you need to follow the three steps 
as following instructions:** 



# Setp1: Prepare your host computer

We recommend using **Ubuntu operating system with version 21.04** as same as us. On the other hand,
  you can also use other versions of the Ubuntu operating system or other operating systems(such as CentOS) as long as 
you can install the Docker and load our graphcs docker image successfully.  



# Setp2: Install Docker and load grahpcs image

## Install Docker

Installing Docker follows the official document.: https://docs.docker.com/engine/install/


## Load graphcs image
 Please load the graphcs image after installing Docker successfully. 
 
### download

You need to download the graphcs docker image (graphcs.tgz) from [websit](https://www.synapse.org/#!Synapse:syn26147749/files/)



### Import graphcs image into Docker as following commands: 
1. open a terminal named t1

2. tar -xzvf graphcs.tgz

3. `sudo docker load -i graphcs.tar` (load the graphcs docker image)

4. `sudo docker images` (list available docker images in your Docker and graphcs will appear if you load successfully)

5. (Optionally) If the names of REPOSITORY and TAG are graphcs and latest, please ignore this command. However, 
if the names of REPOSITORY and TAG in graphcs image are none, please follow the following 
command to re-name them (graphcs-image-id can be found in step 4 in this section.):

    `sudo docker tag graphcs-image-id graphcs:latest`
    
Note: if you load graphcs iamge successfully, you will get the following information:
![(Variational) load_successful](load_successful.jpg)
    
    
    
### Run Docker with graphcs image

1. Please download the preprocessed_data and raw_data from: [data](https://drive.google.com/drive/folders/1ST0T90HcxCKuxOTmOvqCI-IyE2IY6YvM?usp=sharing)

2. Please Decompress them and organize them as the format of the data folder in [GraphCS](https://github.com/biomed-AI/GraphCS)

![(Variational) data_formart](data_formart.jpg)

3. Run docker with your data path:

- `sudo docker run --shm-size 20g -v absolute-dir-path-to-data:/home/biomed-ai/GraphCS/data
 -v absolute-dir-path-to-example_data:/home/biomed-ai/GraphCS/example_data -it graphcs /bin/bash`

Note:
- absolute-dir-path-to-data is the absolute path of data that downloaded from [data](https://drive.google.com/drive/folders/1ST0T90HcxCKuxOTmOvqCI-IyE2IY6YvM?usp=sharing),
 for exmaple, '/home/data/'.
- absolute-dir-path-to-data is the absolute path of example_data [example_data](https://drive.google.com/drive/folders/1ST0T90HcxCKuxOTmOvqCI-IyE2IY6YvM?usp=sharing),
 for exmaple, '/home/example_data/'.

If you run docker with graphcs image successfully, you will see the following information in terminal t1. 
![(Variational) run](run.jpg)




# step 3: Test datasets with GraphCS
After you load the graphcs image in your Docker successfully following above two steps,
 you can run GraphCS with following two Schemes:


##  Scheme I
python train.py --data dataset_name --batch batch_size


##  Scheme II

Follow the instructions in the README [GraphCS](https://github.com/biomed-AI/GraphCS).

Note: you must execute all commands in terminal t1, which was entering the graphcs running environment. 


or 


##  Scheme III 
You can run GraphCS on all datasets using the script **test.sh**
or on single dataset using script **start.sh**. You must copy them into the running  docker with  graphcs image as following 
commands (These scripts are stored in the directory of the same level as the graphcs image):

1. open a new terminal named t2

2. `sudo docker ps -a` to get the CONTAINER_ID  of graphcs docker 
 ![(Variational) Container_id](Container_id.jpg)

3. Running in terminal t2:
	sudo docker cp test.sh | start.sh CONTAINER_ID:/home/biomed-ai/GraphCS/ 
	
4. Then, you can run  test.sh or start.sh in terminal t1 using following commands:
	`bash    test.sh | start.sh`
	
Note: you can revise the dataset name in scripts test.sh or start.sh as your needed. 
