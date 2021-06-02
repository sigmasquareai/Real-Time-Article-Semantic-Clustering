from milvus import Milvus, IndexType, MetricType, Status
from config_parser import GetGlobalConfig
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import requests
import json
import logging
import pandas as pd
import numpy as np
import time

GlobalConfig = GetGlobalConfig() #Get config
RELATED_DB_NAME = 'related_articles'
_THRESHOLD_SIMILARITY = 0.56

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
    _ENGINE=create_engine(_CONNECTION_STRING, echo=False, poolclass=NullPool, pool_pre_ping=True)
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

def get_query_partitions(qryDate, milvusClient):
    '''
    Function to get query partitions to search into
    '''
    qryPartitions = []
    partitionList = [
                        str( (qryDate - pd.Timedelta(days=1)).date() ).replace('-',''),\
                        str( (qryDate - pd.Timedelta(days=2)).date() ).replace('-',''),\
                        str(qryDate.date()).replace('-','')
                    ] #Convert dates to strings
    #Check if partitions exists
    for qryPartition in partitionList:
        _exists = milvusClient.has_partition(GlobalConfig['collection_name'], qryPartition)[1]
        if _exists:
            qryPartitions.append( qryPartition )

    #Exit if no partition exists
    if not qryPartitions:
        return -1
    else:
        return qryPartitions    

def _find_similar_in_milvus(vec, qryPartitions, milvusClient):
    '''
    Function to find similar articles in Milvus DB
    '''
    search_param = {'nprobe': 16}
    try:
        #Search for Top 100 matches
        status, _similarResults = milvusClient.search(GlobalConfig['collection_name'], query_records=vec,\
                                            top_k=100, partition_tags=qryPartitions, params=search_param)
    except Exception as e:
        print(f'Milvus Search Error: {str(e)}')
    
    if not status.OK():
        print(f'No Similar Found: {status}')
    else:
        try:
            similarIdx, similarScore = zip(*[_item for _item in list(zip(_similarResults.id_array[0],\
                                _similarResults.distance_array[0])) if _item[1] > _THRESHOLD_SIMILARITY])
            return [similarIdx, similarScore]
        except Exception as e:
            print('Threshold returned zero')
            return -1
            

def _write_to_db(similarIdx, similarScore, queryID, dbCONN):
    '''
    Function to write related articles results in DB
    '''
    resultDF = pd.DataFrame( {'sim_score':similarScore,\
                                'article':queryID.tolist(),\
                                'related_article':similarIdx } ) #collecting results
    try:
        resultDF.to_sql(RELATED_DB_NAME, con=dbCONN, index=False, if_exists='append') #Writing results to DB
        return True
    except Exception as e:
        print(f'{queryID} Write Error: {str(e)}')
        return False    

def _update_milvus(_qry_embeddings, articlesDF, milvusClient):
    '''
    Function to update embeddings in Milvus DB
    '''
    
    def _insert_into_partition(_qry_embeddings, partitionName, partitionDF, _CALC=False):
        if _CALC:
            _qry_embeddings = _make_embeddings(partitionDF.headline.tolist())

        _RETRY = -3
        
        while _RETRY <= -1:
            #Insert embeddings data in Milvus DB
            try:
                _status, inserted_vector_ids = milvusClient.insert(GlobalConfig['collection_name'], \
                                                                records=_qry_embeddings, \
                                                                ids=partitionDF.id.tolist(), \
                                                                partition_tag=partitionName)
            except Exception as e:
                print(f'_insert_into_partition: DB insertion was not successful: {str(e)}')
            
            if _status.OK():
                print('***Milvus Updated***')
                _RETRY += 3
            
            elif f'partition {partitionName} not found' in _status.message:
                #Create partition if not exists, Retry upto 3 times
                try:
                    status = milvusClient.create_partition(GlobalConfig['collection_name'], partitionName)
                    print(f'{partitionName} partition was created') if status.OK() else \
                                                print(f'Partition {partitionName} was not created, Retrying... ')
                except Exception as e:
                    print(f'Milvus Create Partition Error: {str(e)}')
                _RETRY += 1

    uniqueDates = pd.to_datetime(articlesDF['published_at'], utc=True).dt.date.unique().tolist()

    if len(uniqueDates) == 1:
        partitionName = str(uniqueDates[0]).replace('-','')
        print(f'>>> Updating milvus: with unique date')
        _insert_into_partition(_qry_embeddings, partitionName, articlesDF)
    else:
        for _date in uniqueDates:
            partitionName = str(_date).replace('-','')
            partitionDF = articlesDF[ articlesDF['published_at'].astype(str).str.contains(str(_date)) ]
            print(f'>>> Updating milvus: with multiple date')
            _insert_into_partition(_qry_embeddings, partitionName, partitionDF, _CALC=True)

def find_similar_and_write_to_db(articlesDF, milvusClient, dbCONN):
    '''
    Find similar articles for query articles
    '''
    #Caclulate query articles' embeddings 
    _qry_embeddings = _make_embeddings(articlesDF.headline.tolist())
    if isinstance(_qry_embeddings, int):
        print(f'Processing Error: MLServer request could not process ')
        return -1

    for idx, vec in enumerate(_qry_embeddings): #Iterate over query embeddings
        try:
            print('+'*50)
            vec =  np.expand_dims(vec, 0) #Expand dims and convert to numpy
            
            _startTime = time.time() #Calculating total execution time
            
            qryDate = articlesDF.published_at.iloc[idx] #Published date of query articles
            qryID = articlesDF.id.iloc[idx] #ID of query article
            
            qryPartitions = get_query_partitions(pd.Timestamp(qryDate), milvusClient) #Get list of partitions to search into
            print(f'Finding similar articles for ID {qryID} in {qryPartitions} partitions')
            
            if isinstance(qryPartitions, int): #Check if any search partition 
                print(f'{qryID}: No Similar Articles - no partitions')
                continue
            
            else:
                similarResults = _find_similar_in_milvus(vec, qryPartitions, milvusClient)
                
                if isinstance(similarResults, int): #Check if any search partition 
                    print(f'{qryID}: No Similar Articles Found')
                    continue
                
                else:
                    similarIdx, similarScore = similarResults
                    _write_status = _write_to_db(similarIdx, similarScore, qryID, dbCONN)
                    if _write_status:
                        print(f'Processed {len(similarIdx)} similar artices for ID: {qryID}')

        except Exception as e:
            print(f'{qryID} Processing Error: {str(e)}')
    
    #Update Milvus cache
    _update_milvus(_qry_embeddings, articlesDF, milvusClient)

def main():

    dbCONN = create_db_connection() #Create Database Connection
    milvusClient = Milvus(GlobalConfig['milvus_host'], GlobalConfig['milvus_port']) #Create Milvus db connection

    while True:

        print('\n'*3)
        print('='*50)
        print("Reading articles from DB")
        
        #Read all articles from Database
        try:
            articlesDF = pd.read_sql_query( GlobalConfig['getlatest'], con=dbCONN )
        except Exception as e:
            print(f'Database Read Error: {str(e)}')
            time.sleep(10)
            dbCONN = create_db_connection()
            continue
        
        #Check if there are new artciles in database
        if articlesDF.empty:
            print('='*50)
            print('No new articles found!')
            print('='*50)
            time.sleep(30)
            continue
        
        else:
            #Preprocess the dataframe by dropping NULL article headlines
            articlesDF['headline'].replace('', np.nan, inplace=True)
            articlesDF.dropna(subset=['headline'], inplace=True)
            
            print('\n'*3)
            print('='*50)
            print(f"Found {articlesDF.shape[0]} new articles")
            print('='*50)
            print('\n'*1)
            

            startTime = time.time() #Calculating total execution time

            #Finding similar articles
            find_similar_and_write_to_db(articlesDF, milvusClient, dbCONN) 
            
            try:
                IDs_to_update = articlesDF.id.tolist()
                if len(IDs_to_update) == 1:
                    dbCONN.execute(GlobalConfig['updatestatus'].format( '('+str(IDs_to_update[0])+')' ))
                else:
                    dbCONN.execute(GlobalConfig['updatestatus'].format( tuple(IDs_to_update) )) #Update status of processed articles
                print(f'ID Status updated')
            
            except Exception as e:
                print(f"Database status update error: {str(e)}")
                dbCONN = create_db_connection()

            print( f"Total execution time for {articlesDF.shape[0]} articles is {time.time()-startTime} seconds" )
            print('='*50)
            print(pd.Timestamp.now(tz='utc'))
            print('='*50)
            print('\n'*3)

            time.sleep(30)

if __name__ == '__main__':
    main()
