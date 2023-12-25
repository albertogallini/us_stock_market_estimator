'''
Created on Nov 28, 2023

@author: albertogallini
'''

import tensorflow as tf
from   tensorflow.keras import layers,regularizers
from   tensorflow import keras


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import fnmatch


import logging
import os

from  const_and_utils import *
from  sequential_model_1stock_multifactors import SequentialModel1StockMultiFactor
from  sequential_model_1stock import evaluate_ticker_distribution






class SequentialModel3StockMultiFactor(SequentialModel1StockMultiFactor):
    
    __OUTPUT_SERIES_NUMBER = 3
    
    def __init__(self, input_data_price_csv : str, 
                 input_data_rates_csv : str ,
                 input_fear_and_greed_csv : str,
                 lookback : int  = 14, 
                 epochs : int = 6,
                 training_percentage : float = 0.90,
                 logger: logging.Logger = None ) :
        
        super().__init__(input_data_price_csv,
                       input_data_rates_csv,
                       input_fear_and_greed_csv,
                       lookback,
                       epochs,
                       training_percentage,
                       logger)
        
        

    
    def calibrate_model (self):
       
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        datal = self.df['Close'].values
        inputs = np.array([self.data[i:i+self.lookback] for i in range(len(self.data)-self.lookback-self.__OUTPUT_SERIES_NUMBER)])
        labels = np.array([datal[i:i+self.__OUTPUT_SERIES_NUMBER] for i in range(self.lookback, len(datal)-self.__OUTPUT_SERIES_NUMBER)])
        
        print(inputs.shape)
        print(labels.shape)
        factor_num = len(self.calibration_factors_list)
        print(factor_num)
        
        # Split data into training and test sets
        train_size = int(len(inputs) * self.training_percentage)
        x_train, self.x_test = inputs[:train_size], inputs[train_size:]
        y_train, self.y_test = labels[:train_size], labels[train_size:]
        self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback:]
        
        
        self.logger.debug('input data shape x-train,x-test,time')
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        self.logger.debug(self.train_time.shape)
   
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        
        self.logger.info('Defining model ...')
        self.logger.info('RNN model ...')
        self.model = tf.keras.Sequential()
        #self.model.add(tf.keras.layers.Reshape((-1, 1)))
        
        self.model.add(tf.keras.layers.LSTM(self.lookback * factor_num  , activation='tanh',input_shape=(self.lookback, factor_num ))) #, recurrent_activation='sigmoid'))
        self.model.add(tf.keras.layers.Dense(units = factor_num, 
                                             kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                             bias_regularizer=regularizers.L2(1e-4),
                                             activity_regularizer=regularizers.L2(1e-5)))
        self.model.add(tf.keras.layers.Dense(self.__OUTPUT_SERIES_NUMBER))
           
        self.logger.info('Compile model...')
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        # Train model
        self.logger.info('Training model ...')
        self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test),callbacks=[earlystop])
        
        self.logger.info('calibration complete.')
        
        
        
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        next_biz_day_input = np.array([self.data[len(self.data) - self.lookback:len(self.data)+1]])
        print("Input: " + str(next_biz_day_input.shape))
        next_biz_day_price = self.model.predict(next_biz_day_input)
        if (denormalize):
            return ((next_biz_day_price * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price,self.data[len(self.data)-1]
        
        
   
                        
        
        
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='val_loss',  # Quantity to be monitored.
                          min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=6,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        
        
   
   

def evaluate_ticker(input_file:str, calibrate: bool, scenario_id: int, model_date: str):
    print("Calibrate SequentialModel1StockMultiFactor Model ...")
    sm3s = SequentialModel3StockMultiFactor(input_data_price_csv = input_file,
                                            input_data_rates_csv = FOLDER_MARKET_DATA+"/usd_rates.csv",
                                            input_fear_and_greed_csv = FOLDER_MARKET_DATA+"/fear_and_greed.csv") 
    
    if calibrate:
        sm3s.calibrate_model()
    else:
        sm3s.load_model(path=FOLDER_REPORD_PDF+model_date+"/", scenario_id=str(scenario_id))
        
    if (sm3s.model == None):
        return None, None
    ticker = get_ticker(input_file)
    return ticker, sm3s 

        
def evaluate_ticker_distribution(input_file:str, scenarios: int = 10, calibrate : bool = True, model_date: str = "15-12-2023" ):
    from scipy.stats import norm,lognorm

    
    fprices_ml = list()
    for i in range (0,scenarios):
        ticker, sm3s = evaluate_ticker(input_file=input_file,calibrate=calibrate, scenario_id=i, model_date=model_date)
        p_list,p_1 = sm3s.get_forecasted_price()
        for i in p_list:
            fprices_ml.append(p_list[0])
            
            
    fig, axs = plt.subplots(sm3s._SequentialModel3StockMultiFactor__OUTPUT_SERIES_NUMBER)
    for i in range (0,len(fprices_ml[0])):
        fprices = [a[i] for a in fprices_ml]
        mu, std = norm.fit(fprices)
        
        
        
        axs[i].hist(fprices, bins=scenarios, density=True, alpha=0.6, color='g')
    
        # Plot the PDF.
        x = np.linspace(mu - std, mu + std, 100)
        xx = np.linspace(mu - 3*std, mu + 3*std, 100)
        p = norm.pdf(x, mu, std)
        pp = norm.pdf(xx, mu, std)
        axs[i].plot(x, p, 'k', linewidth=2)
        axs[i].plot(xx, pp, 'k', linewidth=1, linestyle='dashed')
        
        pattern = r"_([^_]*)\."
        ticker = re.findall(pattern, input_file)
        
        title = str(ticker[0]) + " - Fit results: mu = %.2f [%.2f,%.2f],  std = %.2f, prev price: %2f"  % (mu, mu - std, mu + std, std, p_1)
        if (mu < p_1):
            axs[i].set_title(title, color="red")
        if (mu > p_1):
            axs[i].set_title(title, color="green")
        if (mu == p_1):
            axs[i].set_title(title, color="black")
            
            
        s, loc, scale = lognorm.fit(fprices, floc=0)
        mean, var  = lognorm.stats(s, loc, scale, moments='mv')
        std = np.sqrt(var)
        pdf = lognorm.pdf(x, s, loc, scale)
        axs[i].plot(x, pdf, label='log-normal distribution')
        axs[i].axvline(mean, color='r', linestyle='--', label='Mean')
        axs[i].axvline(mean - std, color='b', linestyle='--', label='Mean - STD')
        axs[i].axvline(mean + std, color='b', linestyle='--', label='Mean + STD')
       
    plt.tight_layout()   
    plt.show()

    # Print the 3 sigma
    print("3 sigma: ", 3*std)    
    
    
    
def check_data_correlation(input_file:str):
    sm1s = SequentialModel3StockMultiFactor(input_data_price_csv = input_file,
                                            input_data_rates_csv = FOLDER_MARKET_DATA+"/usd_rates.csv",
                                            input_fear_and_greed_csv = FOLDER_MARKET_DATA+"/fear_and_greed.csv") 
    
    print (sm1s.df_not_normalized[['Close']].describe())

    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='3 Mo', color = "green")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='4 Mo', color = "blue")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='6 Mo', color = "green")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='1 Yr', color = "red")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='2 Yr', color = "red")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='5 Yr', color = "red")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='7 Yr', color = "red")
    sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='10 Yr', color = "red")
    
    for t in ['IYK', 'ITA', 'RTH', 'XRT', 'XHB', 'XLY', 'IBD', 'XRT', 'VDC', 'IYC', 'IYC', 'VDE', 'XOP', 'XLE', 'VCR', 'XES', 'KBE', 'IHF',
                        'IYF', 'KRE', 'IBB', 'XLV', 'XPH', 'XLP', 'XLB', 'USD', 'IYJ', 'XLI', 'IYT', 'IGV', 'IYW', 'XLB', 'XME', 'IP', 'IYR', 'IYR', 'IYT', 
                        'ITB', 'IYR', 'IVV', 'IYZ', 'XLU', 'XLU', 'IDU'] :
        
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y="['"+t+"']", color = "red")
        
    plt.show()
        
        
def main():
    init_config()
    print("Running in one ticker mode")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #check_data_correlation("/Users/albertogallini/projects/Market Price Fetcher/market_price_fetcher/data/price_fetcher_MSFT.csv")
    evaluate_ticker_distribution(FOLDER_MARKET_DATA+"price_fetcher_PYPL.csv", 30, calibrate = True, model_date= "18-12-2023_portfolio_calibration")
   
    
if __name__ == '__main__':  
    main()
      