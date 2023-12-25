

PREFIX_PRICE_FETCHER            = "price_fetcher_"
PREFIX_INDEX_SECTOR             = "index_sector_"
PREFIX_INDEX_SUB_SECTOR         = "index_sub_sector_"

FILE_NAME_RATES                 = "usd_rates.csv"
FILE_NAME_FNG                   = "fear_and_greed.csv"

FOLDER_MARKET_DATA              = "/Volumes/data/"
FOLDER_REPORD_PDF               = "/Volumes/reports/"


def init_config(cfg_file: str = "price_estimator/config.json" ):
    import json
    with open(cfg_file) as f:
        config  = json.load(f)
        
        FOLDER_MARKET_DATA              = config["FOLDER_MARKET_DATA"]
        FOLDER_REPORD_PDF               = config["FOLDER_REPORD_PDF"]

        print("Loaded configuration:")
        print(config)
        print(FOLDER_MARKET_DATA)
        print(FOLDER_REPORD_PDF)
       

    
def get_ticker(input_file : str):
    import re
    pattern = r"_([^_]*)\."
    ticker = re.findall(pattern, input_file)
    return ticker


def append_today_date( input_folder_name: str) -> str:
    from datetime import date
    
    today = date.today()
    today_str = today.strftime("%d-%m-%Y")
    output_folder_name = input_folder_name+ "/"+today_str+"/"
    return output_folder_name
        
 
import pandas as pd    

def fill_value(column_name: str , df: pd.DataFrame):
    df[column_name] = df[column_name].replace(0., pd.NaT)
    df[column_name] = df[column_name].fillna(method='ffill')
    return df                                         