'''
Created on Dec 1, 2023

@author: albertogallini
'''


import const_and_utils

def remove_duplicate_dates(input_file_name):
    import pandas as pd
    try:
        df = pd.read_csv(input_file_name)
    except:
        df = pd.DataFrame()
        
    if (not df.empty):
        print("processing: " + input_file_name)
        #print(df.tail(10))
        df.drop_duplicates(subset='Date',inplace=True)
        #print("saving: " + input_file_name)
        #print(df.tail(10))
        df[['Date', 'Open', 'High','Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']].to_csv(input_file_name)
    
    
if __name__ == '__main__':
    import os
    #remove_duplicate_dates(PriceEstimator.const_and_utils.FOLDER_MARKET_DATA+"price_fetcher_A.csv")
    
    files = os.listdir(const_and_utils.FOLDER_MARKET_DATA)
    for f in files:
        if f.endswith(".csv") and not f.endswith("rates.csv"): 
            try:
                remove_duplicate_dates(const_and_utils.FOLDER_MARKET_DATA+f)
            except Exception as e:
                print(" remove_duplicate_dates - Caught an exception: ", e)
                continue
            
   