import requests

from  price_estimator.const_and_utils import FOLDER_MARKET_DATA,FILE_NAME_INFLATION

class InflationFetcher():

    __url10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T10YIE&scale=left&cosd=2019-01-22&coed=YYYYMMDD&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=YYYYMMDD&revision_date=YYYYMMDD&nd=2020-01-01"
    
    __url5 = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=T5YIE&scale=left&cosd=2020-01-01&coed=YYYYMMDD&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-01-01&line_index=1&transformation=lin&vintage_date=YYYYMMDD&revision_date=YYYYMMDD&nd=2020-01-01"
    
    __urlcpi ="https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=CORESTICKM159SFRBATL&scale=left&cosd=2020-01-01&coed=YYYYMMDD&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=YYYYMMDD&revision_date=YYYYMMDD&nd=2020-01-01"
    
    def __init__(self, datestr: str = None, url: str = 'cpi'):
       import numpy as np
       import io

       self.__url = self.__url10
       field = 'T10YIE'
       self.inflation_field = 'inflation'
       
       if url == '10Y' :
            self.__url = self.__url10
            field = 'T10YIE'
       if url == '5Y' :
            self.__url = self.__url5
            field = 'T5YIE'
       if url == 'cpi' :
            self.__url = self.__urlcpi
            field = 'CORESTICKM159SFRBATL'
            
       if datestr ==None:
            from datetime import date 
            today = date.today()
            today_str = today.strftime("%Y-%m-%d")
            self.__url = self.__url.replace("YYYYMMDD",today_str)
       else:
            self.__url = self.__url.replace("YYYYMMDD",datestr)

       self.__url =  self.__url.rstrip()
       response = requests.get(self.__url)
       csv_file = io.StringIO(response.content.decode('utf-8'))
       self.__dfi = pd.read_csv(csv_file)

       self.__dfi[field] = self.__dfi[field].apply(lambda x : "" if x=="." else x )
       self.__dfi[self.inflation_field] = self.__dfi[field].apply(lambda x : float(x) if "" != x else np.nan)
       self.__dfi[self.inflation_field].replace(to_replace=np.nan, method='ffill', inplace=True)
       self.__dfi.rename(columns={'DATE': 'Date'}, inplace=True)

       if url == 'cpi':
              self.__dfi['Date_Index'] = pd.to_datetime( self.__dfi['Date'])
              self.__dfi =  self.__dfi.set_index('Date_Index')
              self.__dfi = self.__dfi.resample('D').interpolate(method='quadratic')
              self.__dfi['Date']  = [d.strftime("%Y-%m-%d") for d in  self.__dfi.index]
            
            
    def get_inflation(self):
        return self.__dfi[['Date',self.inflation_field]]


     
     
  
import unittest
import logging  
import pandas as pd


   
class TestIndexPrice(unittest.TestCase):
    
    def test_index_price(self):
     
        print("Start...")
        
        logger = logging.getLogger('inflation_fetcher.logger')
        file_handler = logging.FileHandler('inflation_fetcher.log')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        
        
        logger.info('Fetching USD 10Y Inflation ')
        irf = InflationFetcher()
        
        infl10y_rates = irf.get_inflation()
        logger.debug(infl10y_rates)
        
        infl10y_rates.to_csv(FOLDER_MARKET_DATA + FILE_NAME_INFLATION)
        df = pd.read_csv(FOLDER_MARKET_DATA + FILE_NAME_INFLATION)
        self.assertEqual(df.empty,False)
        

if __name__ == "__main__":
    unittest.main()
    
    
    
    
        