'''
Created on Dec 6, 2023

@author: albertogallini



App ID
0eCbUYKZ
Client ID (Consumer Key)
dj0yJmk9TVdtbm1OYW5LRDBvJmQ9WVdrOU1HVkRZbFZaUzFvbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PWE5
Client Secret (Consumer Secret)
9c7791b64fd3263193e26623533f32052e2c4ea4

https://fredaccount.stlouisfed.org/apikey
Your registered API key is: 4bd336592d9ce029bb7684b0f892ee42 Documentation is available on the St. Louis Fed web services website.


'''

import requests

class IRFetcherFredgraph():
    
    
    '''
    import requests

    # Define the API endpoint
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Define the parameters
    params = {
        "series_id": "FEDFUNDS",  # The id of the series to fetch
        "api_key": "4bd336592d9ce029bb7684b0f892ee42",  # Replace with your API key
        "file_type": "json",  # We want the data in JSON format
        "frequency": "m",  # We want daily data
    }
    
    # Send the request
    response = requests.get(url, params=params)
    
    # Make sure the request was successful
    #assert response.status_code == 200
    
    # Parse the response
    data = response.json()
    data
    
    # The observations are now stored in data['observations']
    '''
    

    __url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FEDFUNDS&scale=left&cosd=1954-07-01&coed=2023-11-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2023-11-01&line_index=1&transformation=lin&vintage_date=2023-12-01&revision_date=2023-12-01&nd=1954-07-01"
    
    def __init__(self):
        self.__response = requests.get(self.__url)
        # Make sure the request was successful
        assert self.__response.status_code == 200
        # Write the contents of the response to a file
        with open(FILE_NAME_RATES, 'wb') as f:
            f.write(self.__response.content)
            
            
    def get_rates(self):
        return self.__response.content
    
    
    
class IRFetcherTreasury:

    __url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value="
    
    def __init__(self):
        import pandas as pd
        
        self.__df = pd.DataFrame()
        for year in ['2020','2021','2022','2023']:
            tables = pd.read_html(self.__url+year)
            df = tables[0] [['Date','1 Mo','2 Mo','3 Mo','4 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr']]
            df['Date'] = pd.to_datetime(df['Date'])
            if (self.__df.empty):
                self.__df = df
            else:
                self.__df = pd.concat([self.__df,df]).sort_values(['Date'])
        
    def get_rates(self):
        return self.__df
    
        
     
  
import unittest
import logging  
import pandas as pd

from  price_estimator.const_and_utils import FOLDER_MARKET_DATA,FILE_NAME_RATES
   
class TestIndexPrice(unittest.TestCase):
    
    def test_index_price(self):
     
        print("Start...")
        
        logger = logging.getLogger('ratres_fetcher.logger')
        file_handler = logging.FileHandler('rates_fetcher.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
        
        logger.info('Fetching USD Rates ')
        irf = IRFetcherTreasury()
        
        usd_rates = irf.get_rates()
        logger.debug(usd_rates)
        
        usd_rates.to_csv(FOLDER_MARKET_DATA + FILE_NAME_RATES)
        df = pd.read_csv(FOLDER_MARKET_DATA + FILE_NAME_RATES)
        self.assertEqual(df.empty,False)
        

if __name__ == "__main__":
    unittest.main()
    
    
    
    
        