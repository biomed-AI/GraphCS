#download

you need to download the docker image from: https://www.synapse.org/#!Synapse:syn26147749/files/


#import
1. open a cmd-terminal called t1
2. sudo docker load -i graphcs.tar
3. sudo docker images #confirm whether the image is loaded properly
4. if its name and version are shown <none>, run the following command:
    sudo docker tag its-image-id graphcs:latest
#run
1. sudo docker run --shm-size 20g -v absolute-dir-path-to-datasets:/home/biomed-ai/GraphCS/example_data -it graphcs /bin/bash


#test datasets
1. open a new cmd-terminal called t2
2. sudo docker ps -a
---------------------------------------------------------------------------------------------------------
CONTAINER ID        IMAGE     COMMAND            CREATED          STATUS	PORTS	NAMES
id-you-need	graphcs	"/bin/bash"   	xxx                  UP xxx		xxx
******************************** stopped-container rows you don't need ********************************
---------------------------------------------------------------------------------------------------------

3. modifiy the dataset-name or params or steps in 'test.sh' or 'start.sh'
4. in t2 run:
	sudo docker cp test.sh id-you-need:/home/biomed-ai/GraphCS/ # or start.sh
5. in t1 run:
	bash test.sh
		or
	bash start.sh
