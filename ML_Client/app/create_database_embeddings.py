from milvus import Milvus, IndexType, MetricType, Status
from config_parser import GetGlobalConfig
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import requests
import pandas as pd
import numpy as np
import json

GlobalConfig = GetGlobalConfig() #Get config

def create_db_connection():
    '''
    Function that creates connection with PostgreSQL database
    '''
    _CONNECTION_STRING = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
                                        GlobalConfig['user'],
                                        GlobalConfig['password'],
                                        GlobalConfig['host'],
                                        GlobalConfig['port'],
                                        GlobalConfig['database']
                                        )
    _ENGINE=create_engine(_CONNECTION_STRING, echo=False, poolclass=NullPool)
    return _ENGINE.connect()

def _make_embeddings(headlines):
    '''
    Function to get embeddings from MLServer
    '''
    #Define headers for post request
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
    }
    #Converting payload to json serialized
    payload = json.dumps( {'headlines': headlines} )
    #Requesting to server
    response = requests.post(GlobalConfig['mlserver_url'], headers=headers, data=payload)
    #Check if successful
    if response.status_code == 200:
        _embeddings = json.loads( response.text )['embeddings']
        _embeddings = np.array(_embeddings).astype('float32')
        return _embeddings
    else:
        print(response.status_code)
        return -1

def create_milvus_partitions(uniqueDates, milvusClient):
    '''
    Create Milvus partitions if not exists
    '''
    returnList = []

    for _date in uniqueDates:
        partitionName = str(_date).replace('-','')
        
        try:
            _exists = milvusClient.has_partition(GlobalConfig['collection_name'], partitionName)[1]
        except Exception as e:
            print(f'Milvus Read Error: {str(e)}')
        
        if not _exists:
            
            try:
                status = milvusClient.create_partition(GlobalConfig['collection_name'], partitionName)
            except Exception as e:
                print(f'Milvus Create Partition Error: {str(e)}')

            if not status.OK():
                print(f'{_date} Partition was not created')
            else:
                print(f'successfully created partition {partitionName}')
                returnList.append( _date )
        else:
            print(f'Partition already exists {partitionName}')
    return returnList[::-1]

def calculate_and_save_embeddings(articlesDF, milvusClient):
    '''
    Function to calculate embeddings and save in milvus db
    '''
    uniqueDates = pd.to_datetime(articlesDF['published_at'], utc=True).dt.date.unique().tolist()
    assert uniqueDates, '7 days old dates not found' #Assert unique dates

    partitionDates = create_milvus_partitions(uniqueDates, milvusClient) #Create partitions

    print('='*50)
    print(f"Inserting embeddings to {len(partitionDates)} partitions {','.join([str(w) for w in partitionDates])}")
    print('='*50)

    for _date in partitionDates:
        partitionName = str(_date).replace('-','')

        partitionDF = articlesDF[ articlesDF['published_at'].astype(str).str.contains(str(_date)) ]

        print('\n'*3)
        print('='*50)
        print(f'Processing {partitionDF.shape[0]} embeddings for partition {partitionName}')
        print('='*50)

        #Calculate embeddings
        _partition_embeddings = _make_embeddings(partitionDF.headline.tolist())

        #Insert embeddings data in Milvus DB
        _status, inserted_vector_ids = milvusClient.insert(GlobalConfig['collection_name'], \
                                                            records=_partition_embeddings, \
                                                            ids=partitionDF.id.tolist(), \
                                                            partition_tag=partitionName)
        if _status.OK():
            print(f'DB insertion successfull for partition {partitionName}')
        else:
            print(f'DB insertion was not successful for partition {partitionName}: {_status}')

def main():
    
    dbCONN = create_db_connection() #Create Database Connection
    milvusClient = Milvus(GlobalConfig['milvus_host'], GlobalConfig['milvus_port']) #Create Milvus db connection
    
    print('Reading database articles...')
    
    try:
        articlesDF = pd.read_sql_query( GlobalConfig['getall'].format(pd.Timestamp.now(tz='UTC')), con=dbCONN ) #Read all articles from Database
    except Exception as e:
        print(f'Database Read Error: {str(e)}')
    
    if articlesDF.empty: #Check if there are new artciles in database
            print('No new articles found!')
    else:
        articlesDF['headline'].replace('', np.nan, inplace=True) #Preprocess the dataframe by dropping NULL article headlines
        articlesDF.dropna(subset=['headline'], inplace=True)
        print('Total Database Length', articlesDF.shape[0])

        calculate_and_save_embeddings(articlesDF, milvusClient) #Calculate and save embeddings

        status, stats = milvusClient.get_collection_stats(GlobalConfig['collection_name'])

        print(stats['row_count'])
        if stats['row_count'] == articlesDF.shape[0]+10:
            print('\n'*3)
            print('='*50)
            print(f'DATABASE UPDATED SUCCESSFULLY')
            print('='*50)

        milvusClient.close()

if __name__ == '__main__':
    main()
