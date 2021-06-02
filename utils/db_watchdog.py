from config_parser import GetGlobalConfig
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import subprocess
import pandas as pd
import time

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
    _ENGINE=create_engine(_CONNECTION_STRING, echo=False, poolclass=NullPool, pool_pre_ping=True)
    return _ENGINE.connect()

def main():

    dbCONN = create_db_connection() #Create Database Connection

    while True:
        
        try:

            raDF = pd.read_sql_query( "SELECT id,created_at FROM related_articles order by created_at desc LIMIT 5", con=dbCONN ) #Read all articles from Database
            timeNow = pd.Timestamp.now(tz='UTC')

            timeDiff = (timeNow - raDF.created_at.iloc[0])
            timeDiff = int( timeDiff.total_seconds() // 60 )
            print(timeNow, raDF.created_at.iloc[0], timeDiff)

            if timeDiff > 30:
                
                print( '{}: No new related articles in DB, sending email!'.format(timeNow) )
                subprocess.run( ["bash dbwatchdog_helper.sh"], shell=True )

                time.sleep(300)

            else:
                print( '{}: DB is updating successfully'.format(timeNow) )

                time.sleep(60)

        except Exception as e:
            print( '{}: Watchdog Error: {}'.format(timeNow, str(e)) )


if __name__ == '__main__':
    main()