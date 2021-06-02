from milvus import Milvus, IndexType, MetricType, Status
from config_parser import GetGlobalConfig
import random
import numpy as np

GlobalConfig = GetGlobalConfig() #Get config

# Milvus server IP address and port.
# You may need to change _HOST and _PORT accordingly.
_HOST = GlobalConfig['milvus_host']
_PORT = GlobalConfig['milvus_port']  # default value
# _PORT = '19121'  # default http value

# Vector parameters
_DIM = GlobalConfig['dim']  # dimension of vector

_INDEX_FILE_SIZE = GlobalConfig['index_size']  # max file size of stored index

# Create collection demo_collection if it dosen't exist.
collection_name = GlobalConfig['collection_name']

def main():
    # Specify server addr when create milvus client instance
    # milvus client instance maintain a connection pool, param
    # `pool_size` specify the max connection num.
    milvus = Milvus(_HOST, _PORT)

    status, ok = milvus.has_collection(collection_name)

    if ok:
        print('Collection already exists!')
    else:
        param = {
            'collection_name': collection_name,
            'dimension': int(_DIM),
            'index_file_size': int(_INDEX_FILE_SIZE),  # optional
            'metric_type': MetricType.IP  # optional
        }

        milvus.create_collection(param)

        #Insert dummy vectors to init DB
        status = milvus.create_partition(collection_name, 'dummy')
        vectors = [[random.random() for _ in range(int(_DIM))] for _ in range(10)]
        _status, inserted_vector_ids = milvus.insert(collection_name, \
                                                            records=vectors, \
                                                            ids = list(range(10)), \
                                                            partition_tag='dummy')

        # Show collections in Milvus server
        _, collections = milvus.list_collections()

        if collection_name in collections:
            print('Collection created successfully!')

        milvus.close()

if __name__ == '__main__':
    main()