from  const_and_utils import *

from  sequential_model_1stock_multifactors import SequentialModel1StockMultiFactor

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import os



def generate_pdf_report(fprices : dict):
    from matplotlib.backends.backend_pdf import PdfPages
    from const_and_utils import FOLDER_REPORD_PDF
    from datetime import timedelta,datetime
    pdf_pages = PdfPages(FOLDER_REPORD_PDF+'price_trend_estimator.pdf')
    assets = list(fprices.items())
    num_charts_per_page = 4
    num_pages = len(fprices.keys()) // num_charts_per_page + 1
    for page in range(num_pages): # pages
        fig, ax = plt.subplots(2, num_charts_per_page // 2)
        for canvas in range(0, num_charts_per_page): # chart per page

            prediction_index = (page*num_charts_per_page)+(canvas)
            if prediction_index < len(fprices.keys()):
                ticker =  assets[prediction_index][0]   # assets:  fprices is a dict [ticker] -> (forecasted price,model)
                fp     =  assets[prediction_index][1][0]   
                model  =  assets[prediction_index][1][1]  

                fp_series = list([ p[0] for p in model.predictions])
                fp_series.append(fp)
                days      = list(model.test_time)
                days.append( (datetime.strptime(model.time[len(model.time)-1], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d") ) 
                                
                ax[canvas % 2, canvas // 2].grid(which='both', linestyle='--', alpha=0.5)
                ax[canvas % 2, canvas // 2].plot(days, fp_series, alpha = 0.7)
                ax[canvas % 2, canvas // 2].plot( list(model.test_time), list(model.y_test), alpha = 0.7)

                ax[canvas % 2, canvas // 2].set_title("'{} estimated prices ".format(ticker), fontsize=3)
                x_labels =  ax[canvas % 2, canvas // 2].get_xticks()
                x_labels = days
                ax[canvas % 2, canvas // 2].set_xticklabels(x_labels,fontsize=3, rotation=90)
                y_labels =  ax[canvas % 2, canvas // 2].get_yticks()
                y_labels = [ "{:.2f}".format(y) for y in y_labels]
                ax[canvas % 2, canvas // 2].set_yticklabels(y_labels,fontsize=3)
            else:
                continue
        pdf_pages.savefig(fig)
        plt.close(fig)
                
    # Close the pdf
    pdf_pages.close()



def load_models_and_compute_estimate(ticker_list : list , input_data_dir: str,  calibration_folder : str = None ) -> dict:
    
    input_data_file = [(input_data_dir + PREFIX_PRICE_FETCHER + t +".csv") for t in ticker_list]
    ticker_index = 0
    fprices = dict()
    for ticker_index in range (0, len(ticker_list)): 
        
        try:
            
            '''
             as the model is precalibrated the traning percentage here is only used to determine 
             the lenght of the prediction time-frame. It won't affect the model performance. 
            '''
            sm1s = SequentialModel1StockMultiFactor(input_data_price_csv      = input_data_file[ticker_index],
                                                     input_data_rates_csv     = FOLDER_MARKET_DATA + FILE_NAME_RATES,
                                                     input_fear_and_greed_csv = FOLDER_MARKET_DATA + FILE_NAME_FNG,
                                                     training_percentage      = 0.97)  
            
            sm1s.load_model(path=FOLDER_MODEL_STORAGE,scenario_id="")

        except Exception as e:
            print("Caught an exception: ", e)
            print("Error in loading pre-trained model for " + input_data_file[ticker_index] )
            continue

        sm1s.compute_predictions(denormalize =True)
        price, _ = sm1s.get_forecasted_price(denormalize =True)
        fprices[ticker_list[ticker_index]] = ((price[0]),sm1s)
    
    return fprices



import sys

def main():
    import json
    init_config("config.json")
    print(os.environ['PYTHONPATH'])

    tlist = []
    # Load the data from the JSON file
    with open('tickers_red.json', 'r') as f:
        data = json.load(f)
    for item in data:
        stocks = item['stocks']
        tlist += stocks
   
    fprices = load_models_and_compute_estimate(ticker_list=tlist, input_data_dir=FOLDER_MARKET_DATA, calibration_folder=FOLDER_MODEL_STORAGE)
    for t in fprices.items():
        print("ticker : {}  next price : {}".format(t[0], t[1][0]))
    generate_pdf_report(fprices)
        

if __name__ == '__main__':  
    main()