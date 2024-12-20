'''
Created on Nov 17, 2023

@author: albertogallini




'''

import yfinance as yf
import bs4 as bs
from datapackage import Package

import requests
import datetime
import time
import pandas as pd
import logging

from price_estimator.const_and_utils import *




class  AlphaVantage: 
    '''
     Temporary class. It might be helpful in the future. 
    '''
    def __init__(self):
        self.url = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=US&apikey=demo'
        self.headers = {"Authorization": "UKCFL4OVIFQII426"}
    
    def fetch(self):
        r = requests.get(self.url, headers=self.headers)
        return r.json()
    
    
    
class  NSYETickers: 
    from datapackage import Package
   
    NYSE_TICKER_URL = 'https://datahub.io/core/nyse-other-listings/datapackage.json'
    
    def __init__(self):
        self.url = self.NYSE_TICKER_URL
    
    def fetch(self):
        try:
            package = Package(self.url)
        
            nyse_tickers = list()
            # print list of all resources:
            print(package.resource_names)
            
            # print processed tabular data (if exists any)
            for resource in package.resources:
                if resource.descriptor['datahub']['type'] == 'derived/csv':
                    for t in  resource.read('nyse-listed_csv'):
                        nyse_tickers.append(t[0])
            return nyse_tickers
        except:
            print("Datahub.io is down. Using static list")
            return list(pd.read_csv(FOLDER_MARKET_DATA+"static_tickers_list.csv")['Ticker'].values)

    
class NasdaqTickers():
    
    
    NASDAQ_TICKER_URL = 'https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt'
    
    def __init__(self):
        self.url = self.NASDAQ_TICKER_URL
        
    def fetch(self):
        res = requests.get(self.url)
        lines = res.text.split("\n")
        data_split = [item.split("|") for item in lines]
        tickers = list()
        for line in data_split[1:-2]:
            tickers.append(line[0])
        return tickers
    
    

class SP500Tickers():
    
    SPX_TICKER_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    def __init__(self):
        self.url = self.SPX_TICKER_URL
        
    def fetch(self):
        res = requests.get(self.url)
        soup = bs.BeautifulSoup(res.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = [row.findAll('td')[0].text for row in table.findAll('tr')[1:]]
        return [s.replace('\n', '') for s in tickers]

    
    
    
class YahooPriceFetcher():
    
    def __init__(self, ticker):
        self.ticker = ticker

    def get_price(self, p:str, s:str , e:str ) -> pd.DataFrame:
        # Get data on this ticker
        tickerData = yf.Ticker(self.ticker)
        # Get the historical prices for this ticker
        return tickerData.history(period=p, start=s, end=e)
        
        
        
class BatchFetcher():
    
    def __init__(self, period, start, end, ):
        self.period = period
        self.start  = start
        self.end    = end
        
        
    def fetch(self, ticker:str, logger: logging.Logger):
        logger.info("Fetching prices for " + ticker)
        ypf = YahooPriceFetcher(ticker)
        tickerDf = ypf.get_price(self.period, self.start,self.end)
        tickerDf.to_csv(FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + ticker+".csv")

 
 
class BatchFetcherAppend():
    
    def __init__(self, period, start, end, ):
        self.period = period
        self.start  = start
        self.end    = end
        
        
    def fetch(self, ticker:str, logger: logging.Logger):
        logger.info("Fetching prices for " + ticker)
        ypf = YahooPriceFetcher(ticker)
        tickerDf = ypf.get_price(self.period, self.start,self.end)
        if (not tickerDf.empty):
            # Load existing data
            try:
                df_existing = pd.read_csv(FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + ticker+".csv")
            except:
                df_existing = pd.DataFrame()
                
            if (not df_existing.empty):
                logger.info(FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + ticker+".csv")
                df_existing.set_index('Date', inplace=True)
                
                
                # Append new data
                df_combined = pd.concat([df_existing, tickerDf])
                df_combined.reset_index(inplace=True)
                # Drop duplicates
                df_combined.drop_duplicates(subset='Date',inplace=True)
                df_combined.set_index('Date', inplace=True)
                logger.debug(df_combined.tail(30))
                # Write back to csv
                df_combined.to_csv(FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER +ticker+".csv")
      
  
class CustomTickerFetcher(object):
    '''
    this is to fetch a custom ticker prices
    '''

    def __init__(self, ticker: str, logger: logging.Logger):
        self.ticker = ticker
        self.logger = logger
        
    def fetch(self, start_date: str, end_date: str): 
        self.logger.info("Custom Fetching" +   self.ticker)    
        bf = BatchFetcher (period = '1d', start=start_date, end=end_date)
        bf.fetch(ticker = self.ticker, logger = self.logger)
        self.logger.info("Fetching prices done.")  
      
       
'''
  free functions
''' 
        
              
def fulldump(start_date: str, end_date: str, limit = -1):
    print("Start. Limit: " + str(limit))
    logger= logging.getLogger('market_price_fetcher.logger')
    file_handler = logging.FileHandler('market_price_fetcher.log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    index = 0
    logger.info('Fetching NYSE ticker from ')
    nyset =  NSYETickers()
    nyse_tickers = nyset.fetch()
    for t in nyse_tickers:
        try:
            bf = BatchFetcher (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
            index +=1
            if limit > 0 and index > limit:
                print("Reached Limit: " + str(limit))
                break
        except Exception as e:
            print(" full dump ticker {} : {} ".format(t, e))
            continue
        
    logger.info('Fetching NASDAQ ticker from ')
    nasdaqt =  NasdaqTickers()
    nasdaq_tickers = nasdaqt.fetch()
    for t in nasdaq_tickers:
        try:
            bf = BatchFetcher (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
            index +=1
            if limit > 0 and index > limit:
                print("Reached Limit: " + str(limit))
                break
        except Exception as e:
            print(" full dump ticker {} : {} ".format(t, e))
            continue
   
    logger.info('Fetching SPX ticker from ')
    spxt =  SP500Tickers()
    spx_tickers = spxt.fetch()
    for t in spx_tickers:
        try:
            bf = BatchFetcher (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
            index +=1
            if limit > 0 and index > limit:
                print("Reached Limit: " + str(limit))
                break
        except Exception as e:
            print(" full dump ticker {} : {} ".format(t, e))
            continue
      
    
    logger.info("Fetching prices done.")
    
    
    
def update(start_date: str, end_date: str , logger:logging.Logger):


    logger.info('Updating NYSE ticker from ')
    nyset =  NSYETickers()
    nyse_tickers = nyset.fetch()
    for t in nyse_tickers:
        try:
            bf = BatchFetcherAppend (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
        except Exception as e:
            print(" update ticker {} : {} ".format(t, e))
            continue
    
    logger.info('Updating NASDAQ ticker from ')
    nasdaqt =  NasdaqTickers()
    nasdaq_tickers = nasdaqt.fetch()
    for t in nasdaq_tickers:
        try:
            bf = BatchFetcherAppend (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
        except Exception as e:
            print(" update ticker {} : {} ".format(t, e))
            continue
   
    logger.info('Updating SPX ticker from ')
    spxt =  SP500Tickers()
    spx_tickers = spxt.fetch()
    for t in spx_tickers:
        try:
            bf = BatchFetcherAppend (period = '1d', start=start_date, end=end_date)
            bf.fetch(ticker = t, logger = logger)
        except Exception as e:
            print(" update ticker {} : {} ".format(t, e))
            continue
        
    
    
    logger.info("Updating prices done.")
    
    
import unittest
import logging  
import pandas as pd

from  price_estimator.const_and_utils import FOLDER_MARKET_DATA,FILE_NAME_RATES
   
class TestPriceFetecher(unittest.TestCase):
    
    def test_ticker_update(self):
     
        print("Start...")
        logger= logging.getLogger('market_price_fetcher.logger')
        file_handler = logging.FileHandler('market_price_fetcher_test.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        nyset =  NSYETickers()
        nyse_tickers = nyset.fetch()
        
        for ticker in ["TDG", "TDI"]:
            bf = BatchFetcherAppend (period = '1d', start='2023-12-22', end='2023-12-25')
            bf.fetch(ticker = ticker, logger = logger)

            df = pd.read_csv(FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + ticker+".csv")
            self.assertEqual(df.empty,False)
        

if __name__ == "__main__":
    unittest.main()