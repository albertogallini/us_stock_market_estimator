'''
Created on Dec 10, 2023

@author: albertogallini
'''
import logging
from  market_price_fetcher import fulldump,update,CustomTickerFetcher
from  index_sector_fetcher import IndexSectorFetcher
from  rates_fetcher import IRFetcherTreasury
from  fear_and_greed_index import FearAndGreedIndex
        
if __name__ == '__main__':
    
    print("Start...")
    
    logger = logging.getLogger('market_price_fetcher.logger')
    file_handler = logging.FileHandler('market_price_fetcher.log')
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    
    
    try:
        import sys
        from datetime import datetime,date
    
        if len(sys.argv) > 2:
            dump_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
            dumpt_date_str = sys.argv[2]
            today = date.today()
            today_str = today.strftime("%Y-%m-%d")
            print("Executing:" + sys.argv[1] + " --> " + dumpt_date_str +" -- " + today_str)
             
             
            logger.info('Fetching USD index sectors ')
            print('Fetching USD index sectors ')
            isf = IndexSectorFetcher(period = '1d', start='2020-1-1', end=today_str)
            isf.fetch()
                
           
            logger.info('Fetching USD stock price ')
            print('Fetching USD stock price ')
            if sys.argv[1] == "full":
                fulldump('2020-1-1',dumpt_date_str)
            elif sys.argv[1] == "update":
                update(dumpt_date_str,today_str,logger)
            else:
                raise Exception("invalid input")
    
        ctf = CustomTickerFetcher(ticker="IONQ", logger=logger)
        ctf.fetch(start_date='2020-1-1', end_date=today_str)
        ctf = CustomTickerFetcher(ticker="NVDA", logger=logger)
        ctf.fetch(start_date='2020-1-1', end_date=today_str)
        
        # this is a very bad way to remove duplicated as the update sometimes fails and I do not understand why.
        from  utils import remove_duplicate_dates
        import const_and_utils
        import os
        
        files = os.listdir(const_and_utils.FOLDER_MARKET_DATA)
        for f in files:
            if f.endswith(".csv") and not f.endswith("rates.csv"): 
                try:
                    remove_duplicate_dates(const_and_utils.FOLDER_MARKET_DATA+f)
                except Exception as e:
                    print(" remove_duplicate_dates - Caught an exception: ", e)
                    continue
                
                
        logger.info('Fetching USD rates ')
        print('Fetching USD rates ')  
        irf = IRFetcherTreasury()
        usd_rates = irf.get_rates()
        logger.debug(usd_rates)
        usd_rates.to_csv('/Volumes/data/usd_rates.csv')

        fngi = FearAndGreedIndex(start='2020-09-30', end=today_str)
        fngdf = fngi.fetch()
        fngdf.to_csv('/Volumes/data/fear_and_greed.csv')
        
        
       
        
    except Exception as e:
         print("Caught an exception: ", e)
    
   
    