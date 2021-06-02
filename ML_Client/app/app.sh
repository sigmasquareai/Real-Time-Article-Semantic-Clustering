#!/bin/sh
fetchstatus() {
  curl \
    --write-out "%{http_code}\n" \
    --output /dev/null \
    -i \
    -X \
    GET \
    "http://mlserver:8585/health"
}

urlstatus=$(fetchstatus)          		# initialize to actual value before we sleep even once
until [ "$urlstatus" = 200 ]; do  		# until our result is success...
  sleep 3                         		# wait a second...
  urlstatus=$(fetchstatus)        		# then poll again.
  echo 'ML Server not up!'
done

echo '*ML Server is up*'

### Test ML Server ###
python3 bert_server_test.py --n 2

### Build Milvus Cache ###
if [ "$Build_Milvus_Cache" == "True" ]
then
	echo 'Setting up Milvus'
	python3 milvus_setup.py
	echo 'Building Milvus Cache'
	python3 create_database_embeddings.py

	export Build_Milvus_Cache="False"
	echo $Build_Milvus_Cache
else
	echo 'Milvus Cache already built'
fi

### RUN APP ###
python3 -u main.py