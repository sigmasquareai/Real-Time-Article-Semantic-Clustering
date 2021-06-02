#!/bin/sh

########## DOCKER INSTALLATION ############

if ! command -v docker &> /dev/null
then
	curl -fsSL https://get.docker.com -o get-docker.sh
	sh get-docker.sh
else
	echo 'Docker already installed, skipping installation'
fi

####### DOCKER-COMPOSE INSTALLATION #########

if ! command -v docker-compose &> /dev/null
then
	COMPOSE_VERSION=`git ls-remote https://github.com/docker/compose | grep refs/tags | grep -oE "[0-9]+\.[0-9][0-9]+\.[0-9]+$" | sort --version-sort | tail -n 1`
	sudo sh -c "curl -L https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose"
	sudo chmod +x /usr/local/bin/docker-compose
	sudo sh -c "curl -L https://raw.githubusercontent.com/docker/compose/${COMPOSE_VERSION}/contrib/completion/bash/docker-compose > /etc/bash_completion.d/docker-compose"
else
	echo 'Docker Compose already installed, skipping installation'
fi

############# SETUP MILVUS ###############

result=$( sudo docker images | grep 'milvusdb' )

if [ -n "$result" ] && [ "$1" != "buildmilvus" ] 
then
	echo "MilvusDB exists, skipping cache build!"
	echo "Build_Milvus_Cache=\"False\"" > ./ML_Client/.env
else
	MILVUS_ROOT=$PWD/milvus/

	mkdir -p $MILVUS_ROOT/db
	mkdir -p $MILVUS_ROOT/logs
	mkdir -p $MILVUS_ROOT/wal

	echo "Build_Milvus_Cache=\"True\"" > ./ML_Client/.env
fi

########### RUN APP ##############

sudo docker-compose up