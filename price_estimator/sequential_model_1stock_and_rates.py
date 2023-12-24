'''
Created on Nov 28, 2023

@author: albertogallini
'''

import tensorflow as tf
from   tensorflow.keras import layers
from   tensorflow import keras

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re


import logging
import os

from  const_and_utils import get_ticker,fill_value
from  sequential_model_1stock import SequentialModel1Stock


class SequentialModel1StockAndRates(SequentialModel1Stock):
    
    def __init__(self, input_data_price_csv : str,input_data_rates_csv : str , lookback : int  = 14, epochs : int = 4, training_percentage : float = 0.90, logger: logging.Logger = None ) :
        
        if (logger == None):
            self.logger= logging.getLogger(input_data_price_csv+'_calibrate_model_sequential_model.logger')
            file_handler = logging.FileHandler(input_data_price_csv+'_calibrate_model_sequential_model.log')
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
            
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
        
        self.df = pd.read_csv(input_data_price_csv)
        self.df_rates  = pd.read_csv(input_data_rates_csv)
        self.df['Date'] = self.df['Date'].apply(lambda l :  l.split(" ")[0])
        
       
        self.df = self.df.merge(self.df_rates, on='Date', how='left')
        #print(self.df.tail(30))
        self.lookback = lookback  
        self.model  = None
        self.y_test = None
        self.x_test = None
        self.data   = None
        self.predictions = None
        self.time   = None
        self.train_time   = None
        self.test_time   = None
        self.price_denormalization_factor = 1
        self.price_denormalization_sum    = 0
        self.epochs = epochs
        self.training_percentage = training_percentage
        
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        self.logger.info('Normalizing ...')
        
        self.price_denormalization_factor = ( self.df['Close'].max() -  self.df['Close'].min()) 
        self.price_denormalization_sum    = self.df['Close'].min()
        # Assuming you have a DataFrame `df` with 'price' and 'volume' columns
        # Normalize your data (very important for neural networks)
        self.logger.info('Preparing data ...')
        if(len(self.df['Close'].values) != len(self.df['Volume'].values)):
            self.df = pd.DataFrame()
            return
        
        self.df = fill_value('Close',self.df)
        self.df = fill_value('Volume',self.df)
        self.df = fill_value('3 Mo',self.df)
        self.df = fill_value('10 Yr',self.df)
        
        
        
        self.df['Close']  = ( self.df['Close'] -  self.df['Close'].min())   / ( self.df['Close'].max() -  self.df['Close'].min())
        self.df['Volume'] = ( self.df['Volume'] -  self.df['Volume'].min()) / ( self.df['Volume'].max() -  self.df['Volume'].min())
        self.df['3 Mo'] = ( self.df['3 Mo'] -  self.df['3 Mo'].min()) / ( self.df['3 Mo'].max() -  self.df['3 Mo'].min())
        self.df['10 Yr'] = ( self.df['10 Yr'] -  self.df['10 Yr'].min()) / ( self.df['10 Yr'].max() -  self.df['10 Yr'].min())
        
        self.time = self.df['Date']
        # Convert DataFrame to numpy array
        self.data =  self.df[['Close', 'Volume','3 Mo','10 Yr']].values
    
        
    
    def calibrate_model (self, mode: str = "Rnn"):
       
        
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        
        # Define lookback period and split inputs and labels
        
        inputs = np.array([self.data[i - self.lookback:i] for i in range(self.lookback, len(self.data))])
        labels =  self.df['Close'][self.lookback:]
        
        
        #print((inputs[-1] * self.price_denormalization_factor) + self.price_denormalization_sum )
        #print((labels * self.price_denormalization_factor) + self.price_denormalization_sum )
        
        # Split data into training and test sets
        train_size = int(len(inputs) * self.training_percentage)
        x_train, self.x_test = inputs[:train_size], inputs[train_size:]
        y_train, self.y_test = labels[:train_size], labels[train_size:]
        self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback:]
        
        self.logger.debug('input data shape x-train,x-test,time')
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        self.logger.debug(self.train_time.shape)
       
        self.logger.debug('Reshaping data ...')
        # Reshape inputs for dense layer
        print(x_train.shape)
      
        #x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        #self.x_test = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2]))
        
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        
        self.logger.info('Defining model ...')
        
        # Define model
        if mode == "Rnn":
            self.logger.info('RNN model ...')
            self.model = tf.keras.Sequential()
            #self.model.add(tf.keras.layers.Reshape((-1, 1)))
            self.model.add(tf.keras.layers.SimpleRNN(self.lookback*4*8, activation='tanh',input_shape=(self.lookback, 4))) #, recurrent_activation='sigmoid'))
            '''
            self.model.add(tf.keras.layers.Dense(self.lookback * 32))
            self.model.add(tf.keras.layers.Reshape((-1, 1)))  # Reshape the output of the Dense layer
            self.model.add(tf.keras.layers.SimpleRNN(self.lookback * 16, activation='relu'))
            self.model.add(tf.keras.layers.Dense(self.lookback * 8))
            self.model.add(tf.keras.layers.Reshape((-1, 1)))
            self.model.add(tf.keras.layers.LSTM(self.lookback // 2, activation='relu'))
            self.model.add(tf.keras.layers.Reshape((-1, 1)))
            self.model.add(tf.keras.layers.SimpleRNN(self.lookback * 4, activation='relu'))
            self.model.add(tf.keras.layers.Dense(self.lookback * 24, activation='relu', input_shape=(x_train.shape[1], 1)))
            self.model.add(tf.keras.layers.Dropout(0.1))
            '''
            self.model.add(tf.keras.layers.Dense(1))
            
        else:
            self.logger.info('Simple dense model ...')
            self.model = tf.keras.Sequential([
                layers.Dense(2048, activation='relu', input_shape=(x_train.shape[1],)),
                layers.Dense(1)
            ])
            
        
       
        # Compile model
        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        # Train model
        self.logger.info('Training model ...')
        self.model.fit(x_train, y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test),callbacks=[earlystop])
        
        self.logger.info('calibration complete.')
    
    
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        next_biz_day_input = np.array([self.data[len(self.data) - self.lookback:len(self.data)+1] for i in range(self.lookback, len(self.data))])
        print(next_biz_day_input.shape)
        next_biz_day_price = self.model.predict(next_biz_day_input)
        if (denormalize):
            return ((next_biz_day_price[0] * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price[0],self.data[len(self.data)-1][0]

        
        
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='val_loss',  # Quantity to be monitored.
                          min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=6,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        