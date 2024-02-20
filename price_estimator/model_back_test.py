'''
Created on Dec 19, 2023

@author: albertogallini
'''

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


def show_prediction_vs_actual_pdf(predictions : list, prediction_quality:list, scenarios: int, models:list):
    from matplotlib.backends.backend_pdf import PdfPages
    from const_and_utils import FOLDER_REPORD_PDF
    pdf_pages = PdfPages(FOLDER_REPORD_PDF+'back_tester_batch.pdf')

    num_charts_per_page = 4
    num_pages = (len(predictions) // scenarios) // num_charts_per_page + 1
    for page in range(num_pages): # pages
        fig, ax = plt.subplots(2, num_charts_per_page // 2)
        for canvas in range(0, num_charts_per_page): # chart per page
            
            prediction_index = (page*num_charts_per_page*scenarios)+(canvas*scenarios)
            if prediction_index < len(prediction_quality):
                best_prediction  = (prediction_index,prediction_quality[prediction_index])

            for s in range(scenarios): # scenario per chart 
                p_index  = (page*num_charts_per_page*scenarios)+(canvas*scenarios)+s   
                if p_index < len(predictions):
                    p = predictions[p_index]
                    pp = []
                   
                    for pr in p:
                        pp.append(pr[0])
                    df = pd.DataFrame({ 'Distance':abs(models[p_index].y_test- pp)/pp})
                    thresholds = np.arange(0.0,0.2,0.0001)
                    percentages = []
                    for threshold in thresholds:
                        below_threshold_mask = df['Distance'] < threshold
                        below_threshold = df[below_threshold_mask]
                        percentages.append(below_threshold.shape[0]/ df.shape[0])
                    ax[canvas % 2, canvas // 2].grid(which='both', linestyle='--')
                    ax[canvas % 2, canvas // 2].plot(thresholds,percentages, alpha = 0.7)
                    ax[canvas % 2, canvas // 2].fill_between(thresholds, 0, percentages)
                    ax[canvas % 2, canvas // 2].set_title("'{} est. error < %' dist. density. Price vol = {:.2f} , Actual lookback = {}".format(models[p_index].ticker,models[p_index].df['Close'].std(),models[p_index].lookback), fontsize=3)
                    x_labels =  ax[canvas % 2, canvas // 2].get_xticks().tolist()
                    x_labels = [ "{:.2f}".format(x) for x in x_labels]
                    ax[canvas % 2, canvas // 2].set_xticklabels(x_labels,fontsize=3)
                    y_labels =  ax[canvas % 2, canvas // 2].get_yticks().tolist()
                    y_labels = [ "{:.2f}".format(y) for y in y_labels]
                    ax[canvas % 2, canvas // 2].set_yticklabels(y_labels,fontsize=3)

                    if best_prediction[1] < prediction_quality[prediction_index]:
                        best_prediction = (prediction_index,prediction_quality[prediction_index]) 
                    prediction_index +=1
                else:
                    continue
            models[best_prediction[0]].save_model(path = FOLDER_MODEL_STORAGE, scenario_id="_")

        pdf_pages.savefig(fig)
        plt.close(fig)
                
    # Close the pdf
    pdf_pages.close()


def show_prediction_vs_actual_mult(predictions : list, prediction_quality:list, ticker:str, models:list):
        from matplotlib.ticker import FixedLocator
        import matplotlib.colors as mcolors
        
        # Assuming y_test and predictions are of the same length
        # Plot the actual prices and the predicted prices
        # plt.switch_backend('QtAgg')  # For QtAgg backend
        fig, axs = plt.subplots(3)
        a_model = models[0]
        if(a_model.y_test.ndim > 1):
            axs[0].plot(a_model.test_time, a_model.y_test[:,0], color='blue', label='Actual prices')
        else:
            axs[0].plot(a_model.test_time, a_model.y_test, color='blue', label='Actual prices')

        
        for p in predictions:
            if (len(p[0]) > 1 ):
                # Transpose the data to get a list of columns
                pt = p.T   
                # For each column, shift the elements down by the column index and set the first element(s) to 0
                for i, col in enumerate(pt):
                    na   = col
                    time = a_model.test_time
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
                axs[0].grid(which='both', linestyle='--')
                axs[0].plot(a_model.test_time, p, alpha = 0.7)
        
        axs[1].scatter(a_model.y_test, a_model.y_test, label='actual values', color='blue', s=2)
        
        prediction_index = 0
        best_prediction  = (0,prediction_quality[0]) 
        for p in predictions:
            
            if (len(p[0]) > 1 ):
                # Transpose the data to get a list of columns
                pt = p.T   
                # For each column, shift the elements down by the column index and set the first element(s) to 0
                for i, col in enumerate(pt):
                    pt[i] = np.roll(col, i)
                    pt[i][:i] = np.nan
                for i, col in enumerate(pt):
                    axs[1].scatter(a_model.y_test.T[0], pt[i], color='red', s=2)
            else:
                axs[1].scatter(a_model.y_test, p, color='red', s=2)
                
                pp = []
                for pr in p:
                    pp.append(pr[0])
                df = pd.DataFrame({ 'Distance':abs(a_model.y_test- pp)/pp})
                thresholds = np.arange(0.0,0.2,0.0001)
                percentages = []
                for threshold in thresholds:
                    below_threshold_mask = df['Distance'] < threshold
                    below_threshold = df[below_threshold_mask]
                    percentages.append(below_threshold.shape[0]/ df.shape[0])
                axs[2].grid(which='both', linestyle='--')
                axs[2].plot(thresholds,percentages, alpha = 0.7)
                axs[2].fill_between(thresholds, 0, percentages)
                axs[2].set_title("'est. error < %' dist. density. Price vol = {:.2f} , Actual lookback = {}".format(a_model.df['Close'].std(),a_model.lookback))
                if best_prediction[1] < prediction_quality[prediction_index]:
                    best_prediction = (prediction_index,prediction_quality[prediction_index]) 
                prediction_index +=1

        models[best_prediction[0]].save_model(path = FOLDER_MODEL_STORAGE, scenario_id="_")

        axs[0].set_title('Actual vs Predicted Prices ' + str(ticker) )     
        axs[0].set_xticklabels(a_model.test_time,fontsize=5, rotation = 90)
    
        
        plt.show()
        

def back_test(args: tuple):
    input_file = args[0]
    calibration_folder = args[1]
    scenario_id = str(args[2])
    class_type  = args[3]
    training_percentage = float(args[4])
    
    print(input_file)
    print(calibration_folder)
    try:
        prediction_model = create_instance_of_class(class_type,  
                                          input_data_price_csv=input_file,
                                          input_data_rates_csv=FOLDER_MARKET_DATA+"/usd_rates.csv", 
                                          input_fear_and_greed_csv= FOLDER_MARKET_DATA+"/fear_and_greed.csv",
                                          training_percentage=training_percentage) 
        if(calibration_folder == None):
            print("Calibrating the model ...")
            prediction_model.calibrate_model()
        else:
            print("Loading the model ...")
            prediction_model.load_model(calibration_folder, scenario_id)
            
        if (prediction_model.model == None):
            return None, None
        
        prediction_model.plot_model()
        prediction_model.compute_predictions(denormalize = True)

        # Evaluate probability density
        pp = []
        prediction_quality= []
        for pr in prediction_model.predictions:
            pp.append(pr[0])
            if len(pp) != len(prediction_model.y_test) : 
                continue
            df = pd.DataFrame({ 'Distance':abs(prediction_model.y_test- pp)/pp})
            thresholds = np.arange(0.0,0.2,0.0001)
            percentages = []
            for threshold in thresholds:
                below_threshold_mask = df['Distance'] < threshold
                below_threshold = df[below_threshold_mask]
                percentages.append(below_threshold.shape[0]/ df.shape[0])    
            integral_value =  sum(a * b for a, b in zip(thresholds, percentages))
            prediction_quality.append(integral_value)
        print("scenario {} probability density value = {}".format(scenario_id,integral_value))


        return prediction_model.predictions, prediction_quality, prediction_model

    except Exception as e:
            print("Caught an exception: ", e)
            print("Error in calibrating model for " + input_file )
        
    



'''
 ######   MAIN   #####
'''
import sys

def main():
    
    init_config()
    model_class = get_model_class(None)


    if len(sys.argv) > 2:
        # try:
            from multiprocessing import Pool
            subject = sys.argv[1]

            if len(sys.argv) > 3:
                model_class = get_model_class(sys.argv[3])
            
            if len(sys.argv) > 4:
                training_percentage = float(sys.argv[4])

            if subject.endswith(".json"):
                print("Processing " + subject  + "\n")
                import json
                with open(subject, 'r') as f:
                    data = json.load(f)
                    params = []
                    for item in data:
                        stocks = item['stocks']
                        for ticker in stocks:
                            file_name = FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + ticker +".csv" 
                            params = list(params + [ (file_name, None, i, model_class, training_percentage) for i in range(int(sys.argv[2])) ] )

                    with Pool(processes = 3) as pool:
                        print(params)
                        results = pool.map(back_test, params)
                        results = [r  for r in results if r != None]
                        predictions,prediction_quality,models = zip(*results) 

                    show_prediction_vs_actual_pdf(predictions,prediction_quality, int(sys.argv[2]), models)

            else:
                file_name = FOLDER_MARKET_DATA + PREFIX_PRICE_FETCHER + subject +".csv"
                print("Processing " + file_name  + "\n")

                with Pool(processes = 3) as pool: 
                    params = [(file_name, None, i, model_class, training_percentage) for i in range(int(sys.argv[2]))]
                    print(params)
                    results = pool.map(back_test, params)
                    predictions, prediction_quality, models= zip(*results)

                show_prediction_vs_actual_mult(predictions,prediction_quality,get_ticker(file_name),models)
            
            

        # except Exception as e:
        #     print("Caught an exception: ", e)
        #     print("Error : " + sys.argv[1] )

    else:
        print("Missing input ticker")
                
if __name__ == '__main__':
    main()