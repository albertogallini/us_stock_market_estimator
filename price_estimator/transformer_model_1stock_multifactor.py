'''
Created on Nov 28, 2023

@author: albertogallini
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import MultiHeadAttention


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




class TransformerModel1StockMultiFactor(SequentialModel1StockMultiFactor):

    
    def __init__(self, input_data_price_csv : str, 
                 input_data_rates_csv : str ,
                 input_fear_and_greed_csv : str,
                 lookback : int  = 14, 
                 epochs : int = 12,
                 training_percentage : float = 0.90,
                 use_lstm : bool = False,
                 logger: logging.Logger = None ) :
        
        super().__init__(input_data_price_csv,
                       input_data_rates_csv,
                       input_fear_and_greed_csv,
                       lookback,
                       epochs,
                       training_percentage,
                       use_lstm,
                       logger)
        
        
    
    def calibrate_model (self):
       
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        datal = self.df['Close'].values
        inputs = np.array([self.data[i:i+self.lookback] for i in range(len(self.data)-self.lookback-1)])
        labels = np.array([datal[i:i+1] for i in range(self.lookback, len(datal)-1)])
        
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
        self.logger.info('Transformers model ...')

        # Define the inputs
        inputs = [Input(shape=(self.x_test.shape[1], 1)) for _ in range(self.x_test.shape[2])]
        # Concatenate the inputs
        x = tf.keras.layers.Concatenate(axis=-1)(inputs)
        # Apply the Transformer (MultiHeadAttention) layer
        x = MultiHeadAttention(num_heads=factor_num, key_dim=factor_num)(x, x)
        # Add a global average pooling layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        # Add a dense layer for the output
        output = Dense(1)(x)
        # Define the model
        self.model = Model(inputs=inputs, outputs=output)
        # Compile the model
        self. model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        # Train model
        self.logger.info('Training model ...')
        x_train_list = [x_train[:, :, i:i+1]     for i in range(x_train.shape[2])]
        
        x_test_list =  [self.x_test[:, :, i:i+1] for i in range(self.x_test.shape[2])]
        self.model.fit(x_train_list, y_train, epochs=self.epochs, callbacks=[earlystop])
        self.logger.info('calibration complete.')
        
        
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        next_biz_day_input = np.array([self.data[len(self.data) - self.lookback:len(self.data)+1]])
        next_biz_day_input_list = [next_biz_day_input[:, :, i:i+1]     for i in range(next_biz_day_input.shape[2])]
        #print("Input: " + str(next_biz_day_input_list))
        next_biz_day_price = self.model.predict(next_biz_day_input_list)
        if (denormalize):
            return ((next_biz_day_price[0] * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price[0],self.data[len(self.data)-1]
        
        
    def compute_predictions(self, denormalize : bool = False):
        if (self.model):
            self.logger.info('back-testing model ...') 
            
            inputs = np.array([self.data[i - self.lookback:i] for i in range(self.lookback, len(self.data))])
            labels =  self.df['Close'][self.lookback:]
            train_size = int(len(inputs) * self.training_percentage)
            x_train, self.x_test = inputs[:train_size], inputs[train_size:]
            y_train, self.y_test = labels[:train_size], labels[train_size:]
            self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback:]
            # Generate prediction
            x_test_list =  [self.x_test[:, :, i:i+1] for i in range(self.x_test.shape[2])]
            self.predictions = self.model.predict(x_test_list)
            self.logger.debug("Predictions Shape")
            self.logger.debug(self.predictions.shape)
            
            if (denormalize):
                self.logger.debug("price_denormalization_factor: "+str(self.price_denormalization_factor) )
                self.logger.debug("price_denormalization_sum: "   +str(self.price_denormalization_sum)    )
                self.y_test      = (self.y_test      * self.price_denormalization_factor) + self.price_denormalization_sum 
                self.predictions = (self.predictions * self.price_denormalization_factor) + self.price_denormalization_sum
        
    def get_num_forecasted_prices(self):
        return 1
        
   
                        
        
        
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='loss',  # Quantity to be monitored.
                          min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=4,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        
        
   
   

def evaluate_ticker(input_file:str, calibrate: bool, scenario_id: int, model_date: str):
    print("Calibrate SequentialModel1StockMultiFactor Model ...")
    sm3s = TransformerModel1StockMultiFactor(input_data_price_csv = input_file,
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
    check_data_correlation(FOLDER_MARKET_DATA+"price_fetcher_GOOGL.csv")
    evaluate_ticker_distribution(TransformerModel1StockMultiFactor,FOLDER_MARKET_DATA+"price_fetcher_PYPL.csv", 20, calibrate = True, model_date= "18-12-2023_portfolio_calibration")

    
if __name__ == '__main__':  
    main()
      