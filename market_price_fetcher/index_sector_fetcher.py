'''
Created on Dec 10, 2023

@author: albertogallini

Sector    Index Ticker
Consumer Discretionary    XLY
Consumer Staples    XLP
Energy    XLE
Financials    XLF
Health Care    XLV
Industrials    XLI
Information Technology    XLK
Materials    XLB
Real Estate    XRE
Telecommunication Services    XTL
Utilities    XLU

'''

from market_price_fetcher import YahooPriceFetcher
from price_estimator.const_and_utils import *
import pandas as pd


class IndexSectorFetcher(object):
    
    '''
    classdocs
    '''
    __tickers = pd.DataFrame(columns=["Sector","Ticker"])

    def __init__(self, period, start, end, ):
        self.period = period
        self.start  = start
        self.end    = end

        self.__tickers["Sector"] = ['Consumer Discretionary',
                                    'Consumer Staples',
                                    'Energy',
                                    'Financials',
                                    'Health Care',
                                    'Industrials',
                                    'Information Technology',
                                    'Materials',
                                    'Telecommunication Services',
                                    'Utilities']
        
        self.__tickers["Ticker"] = ['XLY',
                                    'XLP',
                                    'XLE',
                                    'XLF',
                                    'XLV',
                                    'XLI',
                                    'XLK',
                                    'XLB',
                                    'XTL',
                                    'XLU']
        
        
        
        
        self.__subsector_data = {
            'Sector': ['Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
                        'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Staples',
                         'Consumer Staples', 'Consumer Staples', 'Energy', 'Energy', 'Energy', 'Energy', 'Energy', 'Financials', 'Financials', 'Financials', 
                         'Financials', 'Health Care', 'Health Care', 'Health Care', 'Health Care', 'Industrials', 'Industrials', 'Industrials', 'Industrials',
                          'Industrials', 'Information Technology', 'Information Technology', 'Materials', 'Materials', 'Materials', 'Real Estate', 
                          'Real Estate', 'Real Estate', 'Real Estate', 'Real Estate', 'Telecommunication Services', 'Telecommunication Services', 'Utilities', 'Utilities', 'Utilities'],
                                 
            'Sub-Sector': ['Apparel Retail', 'Auto Components', 'Food and Staples Retail', 'Home Furnishings', 'Homebuilding', 
                           'Leisure Products', 'Media', 'Restaurants', 'Food', 'Household Products', 'Personal Products', 'Alternative Energy',
                           'Exploration and Production', 'Integrated Oil & Gas', 'Oil & Gas Refining & Marketing', 'Oil Field Services', 'Banks', 'Insurance', 'Multiline Financial Services',
                           'Regional Banks', 'Biotechnology', 'Healthcare Equipment & Services', 'Pharmaceuticals', 'Specialty Retail', 'Construction Materials', 'Construction Services',
                           'Diversified Industrials', 'Machinery', 'Transportation', 'Software & Services', 'Technology Hardware & Equipment',
                        'Construction Materials', 'Metals & Mining', 'Paper & Forest Products', 'REITs', 'Diversified REITs', 'Industrial REITs', 'Office REITs',
                        'Residential REITs', 'Cable & Satellite TV', 'Telecom Services', 'Electric Power', 'Gas Utilities', 'Multi-Utilities'],
                                 
            'Ticker': ['IYK', 'ITA', 'RTH', 'XRT', 'XHB', 'XLY', 'IBD', 'XRT', 'VDC', 'IYC', 'IYC', 'VDE', 'XOP', 'XLE', 'VCR', 'XES', 'KBE', 'IHF',
                        'IYF', 'KRE', 'IBB', 'XLV', 'XPH', 'XLP', 'XLB', 'USD', 'IYJ', 'XLI', 'IYT', 'IGV', 'IYW', 'XLB', 'XME', 'IP', 'IYR', 'IYR', 'IYT', 
                        'ITB', 'IYR', 'IVV', 'IYZ', 'XLU', 'XLU', 'IDU']
        }
        
        self.__subsector_tickers = pd.DataFrame( self.__subsector_data)

        
        
    def get_tickers(self):
        return list(self.__tickers["Ticker"])
    
    def get_subsectors_tickers(self):
        return list(self.__subsector_tickers["Ticker"])
    
    def fetch(self):
        for ticker in self.__tickers["Ticker"]:
            ypf = YahooPriceFetcher(ticker)
            tickerDf = ypf.get_price(self.period, self.start,self.end)
            tickerDf.to_csv(FOLDER_MARKET_DATA+PREFIX_INDEX_SECTOR+ticker+".csv")
           
        for ticker in self.__subsector_tickers["Ticker"]:
            ypf = YahooPriceFetcher(ticker)
            tickerDf = ypf.get_price(self.period, self.start,self.end)
            tickerDf.to_csv(FOLDER_MARKET_DATA+PREFIX_INDEX_SUB_SECTOR+ticker+".csv")
        
            
    
                            
                        
        
        
  
import unittest
import logging  
   
class TestIndexPrice(unittest.TestCase):
    
    def test_index_price(self):
     
        print("Start...")
    
        logger = logging.getLogger('index_sector.logger')
        file_handler = logging.FileHandler('index_sector.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
        logger.info('Fetching USD index sectors ')
        isf = IndexSectorFetcher(period = '1d', start='2020-1-1', end='2024-3-13')
        isf.fetch()
        
        for ticker in isf.get_tickers():
            df = pd.read_csv(FOLDER_MARKET_DATA+PREFIX_INDEX_SECTOR+ticker+".csv")
            self.assertEqual(df.empty,False)
        
        for ticker in isf.get_subsectors_tickers():
            df = pd.read_csv(FOLDER_MARKET_DATA+PREFIX_INDEX_SUB_SECTOR+ticker+".csv")
            self.assertEqual(df.empty,False)
        

if __name__ == "__main__":
    unittest.main()
    