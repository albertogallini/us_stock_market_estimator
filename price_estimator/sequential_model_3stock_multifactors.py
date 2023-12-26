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
        
    def get_num_forecasted_prices(self):
        return self._SequentialModel3StockMultiFactor__OUTPUT_SERIES_NUMBER
        
   
                        
        
        
        
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
    
    

from sequential_model_1stock_multifactors import check_data_correlation,evaluate_ticker_distribution        
        
def main():
    init_config()
    print("Running in one ticker mode")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    check_data_correlation(FOLDER_MARKET_DATA+"price_fetcher_PYPL.csv")
    evaluate_ticker_distribution(SequentialModel3StockMultiFactor,FOLDER_MARKET_DATA+"price_fetcher_PYPL.csv", 2, calibrate = True, model_date= "18-12-2023_portfolio_calibration")

    
if __name__ == '__main__':  
    main()
      