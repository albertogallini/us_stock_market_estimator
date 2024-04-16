import requests, csv, json, urllib
import pandas as pd
import time
from fake_useragent import UserAgent
from datetime import datetime



BASE_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"



class FearAndGreedIndex(object):

    def __init__(self, start:str ='2020-12-30',  end:str = '2023-12-23'):
        self.start_date  = start
        self.end_date = end
        self.fng_data = pd.DataFrame()

    def fetch(self):
        ua = UserAgent()
        headers = {
        'User-Agent': ua.random,
        }

        print("Fetching :" + (BASE_URL + self.start_date))
        r = requests.get(BASE_URL + self.start_date, headers = headers)
        data = r.json()

        
        self.fng_data = pd.read_csv('market_price_fetcher/fear-greed.csv', usecols=['Date', 'Fear Greed'])
        self.fng_data['Date'] = pd.to_datetime(self.fng_data['Date'], format='%Y-%m-%d')  

        self.fng_data.set_index('Date', inplace=True)
        missing_dates = []
        all_dates = (pd.date_range(self.fng_data.index[0], self.end_date, freq='D'))
        for date in all_dates:
            if date not in self.fng_data.index:
                missing_dates.append(date)
                #print(date)
                self.fng_data.loc[date] = [0]
        self.fng_data.sort_index(inplace=True)


        for data in ((data['fear_and_greed_historical']['data'])):
            x = int(data['x'])
            x = datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d')
            y = int(data['y'])
            self.fng_data.at[x, 'Fear Greed'] = y
        
        self.fng_data['Fear Greed'].replace(to_replace=0, method='ffill', inplace=True)
        self.fng_data['Fear Greed'].replace(to_replace=0, method='bfill', inplace=True)
    
        return self.fng_data
        


import unittest
import logging 
from  price_estimator.const_and_utils import FOLDER_MARKET_DATA,FILE_NAME_FNG
   
class TestFearAndGreedIndex(unittest.TestCase):
    
    def test_fear_and_greed_index(self):
     
        print("Start...")
    
        logger = logging.getLogger('index_sector.logger')
        file_handler = logging.FileHandler('index_sector.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
        logger.info('Fetching USD index sectors ')
        fngi = FearAndGreedIndex(start='2020-09-30', end='2024-05-01')
        df = fngi.fetch()
        df.to_csv(FOLDER_MARKET_DATA + FILE_NAME_FNG)
        print(df)
        
        self.assertEqual(df.empty,False)
        

if __name__ == "__main__":
    unittest.main()