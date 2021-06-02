from config_parser import GetGlobalConfig
import numpy as np
import requests
import argparse
import random
import time
import json

DATASET_FILEPATH = 'bert_test_dataset.csv' #Filepath to dataset
GlobalConfig = GetGlobalConfig() #Get config

def argPraser():
    '''
    Argument Parser
    '''
    parser = argparse.ArgumentParser(description="BERT model server deployment and throughput test")
    parser.add_argument("--n", default=1, type=int, help='Number of random tests')
    args = parser.parse_args()
    return args

def get_test_dataset():
    '''
    Function to generate list of sentences from The Office tv show script
    '''
    _FILE = open(DATASET_FILEPATH, 'r').readlines()[1:]
    SENTENCES_DATASET = list(map(lambda x: x.strip('\n'), _FILE)) 
    return SENTENCES_DATASET

def get_sent_batch(SENTENCES_DATASET):
    '''
    Function that randomly generates a batch of sentences
    '''
    _BATCH_SIZE = random.choice( range(512) )
    DATA_BATCH = random.sample(SENTENCES_DATASET, _BATCH_SIZE)
    return DATA_BATCH

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

def calculate_embeddings(DATA_BATCH, idx):
    '''
    Function to Run Tests
    '''
    print('*'*100)
    print('='*35, 'RUNNING TEST ', str(idx+1), '='*35)
    print('*'*100)

    start = time.time()
    _embeddings = _make_embeddings(DATA_BATCH)
    exec_time = time.time() - start

    print('Batch Size: ', len(DATA_BATCH), ' Execution Time: ', exec_time, ' seconds')
    print('\n'*3)

def main():

    args = argPraser() #Parse user arguments

    SENTENCES_DATASET = get_test_dataset() #Read and preprocess test dataset

    #Run number of tests
    for idx in range(args.n):
        DATA_BATCH = get_sent_batch(SENTENCES_DATASET) #Get random batch
        calculate_embeddings(DATA_BATCH, idx) #Run model and calculate embeddings

if __name__ == '__main__':
    main()