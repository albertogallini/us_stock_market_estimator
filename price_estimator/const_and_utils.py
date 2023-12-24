

PREFIX_PRICE_FETCHER            = "price_fetcher_"

FOLDER_PREFIX_DISTRIBUTION_CSV  = "/Volumes/reports/distributions__"
FOLDER_MARKET_DATA              = "/Volumes/data/"
FOLDER_REPORD_PDF               = "/Volumes/reports/"


def init_config(cfg_file: str):
    import json
    with open(cfg_file) as f:
        config  = json.load(f)
        
        FOLDER_PREFIX_DISTRIBUTION_CSV  = config["FOLDER_PREFIX_DISTRIBUTION_CSV"]
        FOLDER_MARKET_DATA              = config["FOLDER_MARKET_DATA"]
        FOLDER_REPORD_PDF               = config["FOLDER_REPORD_PDF"]

        print("Loaded configuration:")
        print(config)
        print(FOLDER_PREFIX_DISTRIBUTION_CSV)
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