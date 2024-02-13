'''
Created on Nov 17, 2023

@author: albertogallini
'''

from  const_and_utils import *
from  sequential_model_1stock import SequentialModel1Stock
from  sequential_model_1stock_and_rates import SequentialModel1StockAndRates
from  sequential_model_1stock_multifactors import SequentialModel1StockMultiFactor
from  sequential_model_3stock_multifactors import SequentialModel3StockMultiFactor
from  transformer_model_1stock_multifactor import TransformerModel1StockMultiFactor

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt



import logging
import os



def produce_distribution_pdf_from_file(input_data_dir: str):
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import norm,lognorm

    
    input_data_dir = append_today_date(input_data_dir)

    
    pdf_pages = PdfPages(input_data_dir+'report_estimated_prices.pdf')
    files = os.listdir(input_data_dir)
    files = [file for file in files if 'est_prices_' in file]
    
    num_chart = len(files)
    
    if num_chart % 2 == 1:
        num_chart +=1
    num_charts_per_page = 4
    
    num_pages = num_chart // num_charts_per_page + 1
    ticker_index = 0
    for n in range (0, num_pages):
        if ticker_index >= len(files):
                break
        fig, ax = plt.subplots(2,num_charts_per_page // 2)
        
        for p in range(0,num_charts_per_page):
            if ticker_index >= len(files):
                break
            print("Evaluate:" + input_data_dir+str(files[ticker_index]))
            
            fprices = list(pd.read_csv(input_data_dir+files[ticker_index])["prices"].values)
            p_1 = fprices[-1]
            fprices =fprices[:-1]
            
            mu, std = norm.fit(fprices)
            
            ax[p % 2, p // 2].hist(fprices, bins=len(fprices), density=True, alpha=0.6)
        
            # Plot the PDF.
            x = np.linspace(mu - std, mu + std, 100)
            xx = np.linspace(mu - 3*std, mu + 3*std, 100)
            norm_p = norm.pdf(x, mu, std)
            norm_pp = norm.pdf(xx, mu, std)
            ax[p % 2, p // 2].plot(x, norm_p, 'k', linewidth=2)
            ax[p % 2, p // 2].plot(xx, norm_pp, 'k', linewidth=1, linestyle='dashed')
            
            s, loc, scale = lognorm.fit(fprices, floc=0)
            lognorm_distr = lognorm.pdf(x, s, loc, scale)
            ax[p % 2, p // 2].plot(x, lognorm_distr, label='log-normal distribution')
            log_norm_mean = scale * np.exp(s**2 / 2)
            
        
            ticker = get_ticker(files[ticker_index])
            
            title = str(ticker[0]) + " - Fit results: mu = %.2f [%.2f,%.2f],  std = %.2f, lognorm  = %.2f, prev price: %.2f"  % (mu, mu - std, mu + std, std, log_norm_mean, p_1)
            if (mu + std < p_1):
                ax[p % 2, p // 2].set_title(title, fontsize=3, color="red")
            elif (mu - std > p_1):
                ax[p % 2, p // 2].set_title(title, fontsize=3, color="green")
            elif (mu > p_1):
                ax[p % 2, p // 2].set_title(title, fontsize=3, color="orange")
            else:
                ax[p % 2, p // 2].set_title(title, fontsize=3, color="lightcoral")

            
            x_labels = ax[p % 2, p // 2].get_xticks().tolist()
            ax[p % 2, p // 2].set_xticklabels(x_labels,fontsize=3)
            y_labels = ax[p % 2, p // 2].get_yticks().tolist()
            ax[p % 2, p // 2].set_yticklabels(y_labels,fontsize=3)
            
            ticker_index += 1
           
        # Save the figure to the pdf
        pdf_pages.savefig(fig)
        # Close the figure
        plt.close(fig)
                
    # Close the pdf
    pdf_pages.close()


def get_market_sentiment_analysis_score():
    from sentiment_model import SentimentModel
    from scrapers import get_yahoo_finance_news_rss,get_news_text

    print("Sampling news for sentiment analysis socre...")
    top_news = get_yahoo_finance_news_rss()
    sm = SentimentModel()
    acc_score = 0
    counter   = 0
    for n in top_news:
        news_text = get_news_text(n[1])
        if(news_text is not None):
            score = sm.get_sentiment_score(news_text)
            print(f"------------ news {n[0]:s} , score = {score:.6f}")
            acc_score += score
            counter += 1
    score = (acc_score/counter)
    print("Accumulated score {:.5f}".format((score)))
    return score


    
def produce_estimate_price_distributions(ticker_list : list , input_data_dir: str, scenarios: int = 10, file_name_id: str ="", save_calibration : bool = False,  calibration_folder : str = None ):
    print(os.environ['PYTHONPATH'])
    num_chart = len(ticker_list)
    files = [(input_data_dir + PREFIX_PRICE_FETCHER + t +".csv") for t in ticker_list]
    output_data_dir = append_today_date(FOLDER_REPORD_PDF)

    ticker_index = 0
    for ticker_index in range (0, len(ticker_list)): 
        fprices = list()
        try:
            sm1s = SequentialModel1StockMultiFactor(input_data_price_csv = files[ticker_index],
                                                     input_data_rates_csv = FOLDER_MARKET_DATA + FILE_NAME_RATES,
                                                     input_fear_and_greed_csv = FOLDER_MARKET_DATA + FILE_NAME_FNG,
                                                     training_percentage=0.97) 
        except Exception as e:
            print("Caught an exception: ", e)
            print("Error in generating model for " + files[ticker_index] )
            continue
        
        for i in range (0,scenarios):
            
            try:
                if( calibration_folder == None):
                    sm1s.calibrate_model()
                    if (save_calibration):
                        sm1s.save_model(path = output_data_dir, scenario_id = str(i))
                else:
                    sm1s.load_model(path = calibration_folder,scenario_id = str(i) )
                    
                price, p_1 = sm1s.get_forecasted_price()
                fprices.append(price[0])
               
            except Exception as e:
                print("Caught an exception: ", e)
                print("Error in calibrating model for " + files[ticker_index] )
                break

        if(len(fprices) > 0):           
            fprices.append(p_1)
        df = pd.DataFrame(fprices,columns=["prices"])
        df.to_csv(output_data_dir + PREFIX_ESTIMATED_PRICE + ticker_list[ticker_index] + ".csv")

    


         
def run_dask(scenarios: int = 20):
    from dask.distributed import Client
    import shutil 
    import json 
    import ast 
    print("Running in dask mode ... ")
    client = Client('192.168.2.132:32000')
    output_dir = append_today_date(FOLDER_REPORD_PDF)
    # check if the folder exists
    if os.path.exists(output_dir):
        # delete the folder
        shutil.rmtree(output_dir)
        
    try:
        os.mkdir(output_dir)
    except OSError:
        print(f"Creation of the directory {output_dir} failed")
        return
    
    params = list()
    # Load the data from the JSON file
    with open('tickers.json', 'r') as f:
        data = json.load(f)
    for item in data:
        stocks, id = item['stocks'], item['id']
        params.append((stocks, FOLDER_MARKET_DATA, scenarios, id, False, None))
        
    if len(params) > 0:
        tasks = [client.submit(produce_estimate_price_distributions, ticker, input_data_dir, scenarios, ids, save_calibration, calibration_folder)  
                 for ticker, input_data_dir, scenarios, ids, save_calibration, calibration_folder in params]
        client.gather(tasks)
    else:
        print("No valid ticker found.")
    
   

'''
 ######   MAIN   #####
'''
import sys

def main():
    init_config("config.json")
    print(os.environ['PYTHONPATH'])
    scenarios = 50
    run_dask(scenarios)
    produce_distribution_pdf_from_file(FOLDER_REPORD_PDF)
        

if __name__ == '__main__':  
    main()
