'''
Created on Dec 19, 2023

@author: albertogallini
'''
import const_and_utils
from const_and_utils import *

from  sequential_model_1stock              import SequentialModel1Stock
from  sequential_model_1stock_and_rates    import SequentialModel1StockAndRates
from  sequential_model_1stock_multifactors import SequentialModel1StockMultiFactor
from  sequential_model_3stock_multifactors import SequentialModel3StockMultiFactor
from  transformer_model_1stock_multifactor import TransformerModel1StockMultiFactor

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt



import logging
import os


def show_prediction_vs_actual_mult(predictions : list, ticker:str, sm1s:SequentialModel1Stock):
        from matplotlib.ticker import FixedLocator
        import matplotlib.colors as mcolors
        
        # Assuming y_test and predictions are of the same length
        # Plot the actual prices and the predicted prices
        # plt.switch_backend('QtAgg')  # For QtAgg backend
        fig, axs = plt.subplots(2)
        axs[0].plot(sm1s.test_time, sm1s.y_test, color='blue', label='Actual prices')
         
        for p in predictions:
            if (len(p[0]) > 1 ):
                # Transpose the data to get a list of columns
                pt = p.T   
                # For each column, shift the elements down by the column index and set the first element(s) to 0
                for i, col in enumerate(pt):
                    na   = col
                    time = sm1s.test_time
                    for j in range (0,i):
                         na   = np.append(na,np.nan)
                         time = np.append(time," ---- + " + str(j))

                    na = np.roll(na, i)
                    color = mcolors.to_rgb('green')
                    color = [color[0], color[1], color[2], (len(pt)-i+1)/(len(pt)+1)]  # RGBA color
                    if(i > 0):
                        axs[0].plot(time, na, color=color, linewidth=1, linestyle="dashed")
                    else:
                        axs[0].plot(time, na, color="red")
            else:
                axs[0].plot(sm1s.test_time, p, color='red', alpha = 0.5)
        
        axs[1].scatter(sm1s.y_test, sm1s.y_test, label='actual values', color='blue', s=2)
        
        
        for p in predictions:
            if (len(p[0]) > 1 ):
                # Transpose the data to get a list of columns
                pt = p.T   
                # For each column, shift the elements down by the column index and set the first element(s) to 0
                for i, col in enumerate(pt):
                    pt[i] = np.roll(col, i)
                    pt[i][:i] = np.nan
                for i, col in enumerate(pt):
                    axs[1].scatter(sm1s.y_test, pt[i], color='red', s=2)
            else:
                axs[1].scatter(sm1s.y_test, p, color='red', s=2)
            
        axs[0].set_title('Actual vs Predicted Prices ' + str(ticker) )     
        axs[0].set_xticklabels(sm1s.test_time,fontsize=5, rotation = 90)
    
        
        plt.show()
        

def back_test(args: tuple):
    input_file = args[0]
    calibration_folder = args[1]
    scenario_id = str(args[2])
    
    print(input_file)
    print(calibration_folder)
    try:
        sm1s = TransformerModel1StockMultiFactor(input_data_price_csv=input_file,
                                                input_data_rates_csv=FOLDER_MARKET_DATA+"/usd_rates.csv", 
                                                input_fear_and_greed_csv= FOLDER_MARKET_DATA+"/fear_and_greed.csv",
                                                training_percentage=0.90) 
        if(calibration_folder == None):
            print("Calibrating the model ...")
            sm1s.calibrate_model()
        else:
            print("Loading the model ...")
            sm1s.load_model(calibration_folder, scenario_id)
            
        if (sm1s.model == None):
            return None, None
        
        sm1s.plot_model()

    except Exception as e:
            print("Caught an exception: ", e)
            print("Error in calibrating model for " + input_file )
        
    sm1s.compute_predictions(denormalize = True)
    return sm1s.predictions, sm1s




'''
 ######   MAIN   #####
'''
import sys

def main():
    if len(sys.argv) > 2:
        try:

            init_config()
            from multiprocessing import Pool
            file_name = FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + sys.argv[1] +".csv"
            print("Processing " + file_name)
            with Pool(processes = 5) as pool: 
                params = [(file_name, None, i) for i in range(int(sys.argv[2]))]
                print(params)
                results = pool.map(back_test, params)
                predictions, sm1s = zip(*results)
                show_prediction_vs_actual_mult(predictions,get_ticker(file_name),sm1s[0])

        except Exception as e:
            print("Caught an exception: ", e)
            print("Error : " + file_name )
        
       
    else:
        print("Missing input ticker")
                
if __name__ == '__main__':
    main()