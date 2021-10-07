#download

you need to download the docker image from: https://www.synapse.org/#!Synapse:syn26147749/files/


#import
- open a terminal named t1

- `sudo docker load -i graphcs.tar`

- `sudo docker images` (confirm whether the image is loaded properly)

- if its name and version are shown <none>, run the following command:

    `sudo docker tag its-image-id graphcs:latest`
    
#run
- `sudo docker run --shm-size 20g -v absolute-dir-path-to-datasets:/home/biomed-ai/GraphCS/example_data -it graphcs /bin/bash`


#test datasets
- open a new terminal named t2

- `sudo docker ps -a` to get a CONTAINER_ID  


- modifiy the dataset-name or params or steps in 'test.sh' or 'start.sh'

- Running in t2:
	sudo docker cp test.sh CONTAINER_ID:/home/biomed-ai/GraphCS/ # or start.sh
	
- Running in t1 :
	bash test.sh
		or
	bash start.sh
