'''
Created on Nov 28, 2023

@author: albertogallini
'''

import tensorflow as tf
from   tensorflow.keras import layers,regularizers
from   tensorflow import keras
from   tensorflow.keras.metrics import F1Score


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



@tf.keras.utils.register_keras_serializable(package='Custom')
class SM3SDistanceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='Mean Absolute Error Metric', **kwargs):
        super(SM3SDistanceMetric, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = (tf.abs(tf.pow(y_true,2) - tf.pow(y_pred,2)))
        self.total.assign_add(tf.reduce_max(error))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)



class SequentialModel3StockMultiFactor(SequentialModel1StockMultiFactor):
    
    __OUTPUT_SERIES_NUMBER = 3

    
    def __init__(self, input_data_price_csv : str, 
                 input_data_rates_csv : str ,
                 input_fear_and_greed_csv : str,
                 lookback : int  = 14, 
                 epochs : int = 32,
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
         
        datal            = self.df['Close'].values
        labels           = np.array([datal[i:i+self.__OUTPUT_SERIES_NUMBER] for i in range(self.lookback, len(datal)-self.__OUTPUT_SERIES_NUMBER)])      
        inputs           = np.array([self.data[i:i+self.lookback]           for i in range(0, len(self.data)-self.lookback-self.__OUTPUT_SERIES_NUMBER)])
        sentiment_inputs = np.array([self.sentiment_data[i - 1:i]           for i in range(self.lookback, len(self.sentiment_data)-self.__OUTPUT_SERIES_NUMBER)])
        
        factor_num            = len(self.calibration_factors_list)
        sentiment_factor_num  = len(self.sentiment_factors_list)

        
        
        # Split data into training and test sets  and set labels (y)
        train_size = int(len(inputs) * self.training_percentage)
        x_sentiment_train, self.x_sentiment_test = sentiment_inputs[:train_size], sentiment_inputs[train_size:]
        x_train, self.x_test = inputs[:train_size], inputs[train_size:]
        y_train, self.y_test = labels[:train_size], labels[train_size:]
        self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback:]
        
        
        self.logger.debug('input data shape x-train,x-test,time')
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        self.logger.debug(self.train_time.shape)
        self.logger.debug(x_train.shape)

        print(x_train.shape)
        print(x_sentiment_train.shape)
        print(y_train.shape)
        print(train_size)
        print(self.x_test.shape)
        print(self.x_sentiment_test.shape)
        
        self.logger.info('Defining model ...')
        self.logger.info('Recurrent model ...')

    
        recurrent_model_inputs =  tf.keras.Input(shape=(self.lookback, factor_num ))
        # Create the first stage of the model : prices, volumes, rates 
        recurrent_nn = tf.keras.layers.SimpleRNN(self.lookback * factor_num,
                                                activation='tanh',
                                                input_shape=(self.lookback, factor_num ))(recurrent_model_inputs)
        if(self.use_lstm):
            recurrent_nn = tf.keras.layers.LSTM(self.lookback * factor_num,
                                                activation='tanh',
                                                input_shape=(self.lookback, factor_num ))(recurrent_model_inputs)
            
        dense_rnn = tf.keras.layers.Dense(units = self.lookback * factor_num, 
                                          activation ='linear',
                                          kernel_regularizer   = regularizers.L1L2(l1 = 1e-4*(-self.price_vol+1), l2 = 1e-4*(-self.price_vol+1)),
                                          bias_regularizer     = regularizers.L2(1e-4),
                                          activity_regularizer = regularizers.L2(1e-5))(recurrent_nn)
        
        dense_rnn_1 = tf.keras.layers.Dense(units = self.__OUTPUT_SERIES_NUMBER, 
                                            activation ='linear',
                                            kernel_regularizer   = regularizers.L1L2(l1 = 1e-4*(-self.price_vol+1), l2 = 1e-4*(-self.price_vol+1)),
                                            bias_regularizer     = regularizers.L2(1e-4),
                                            activity_regularizer =regularizers.L2(1e-5))(dense_rnn)
        
        #dense_rnn_1_reshaped = tf.keras.layers.Reshape((1,self.__OUTPUT_SERIES_NUMBER ))(dense_rnn_1)
        
        
        # Create the second stage of the model : sentiment analysis and sectr correlation
        s_input = tf.keras.Input(shape=(sentiment_factor_num))
    
        print("Sentiment mode input shape{}".format(dense_rnn_1.shape))
        print("Sentiment mode input shape{}".format(s_input.shape))

        # Combine the outputs of the LSTM model and the score input
        sentiment_model_inputs  = tf.keras.layers.Concatenate(axis=1)([dense_rnn_1, s_input])        

        print("Sentiment mode input shape{}".format(sentiment_model_inputs.shape))
        
        sentiment_dense = tf.keras.layers.Dense( self.__OUTPUT_SERIES_NUMBER * sentiment_factor_num,
                                                activation = 'linear', 
                                                kernel_regularizer=regularizers.L1L2(l1=1e-4*(-self.price_vol+1), l2=1e-4*(-self.price_vol+1)),
                                                bias_regularizer=regularizers.L2(1e-4),
                                                activity_regularizer=regularizers.L2(1e-5))(sentiment_model_inputs)
        
        sentiment_predictions = tf.keras.layers.Dense(self.__OUTPUT_SERIES_NUMBER, 
                                                      activation='linear',
                                                      kernel_regularizer=regularizers.L1L2(l1=1e-4*(-self.price_vol+1), l2=1e-4*(-self.price_vol+1)),
                                                      bias_regularizer=regularizers.L2(1e-4),
                                                      activity_regularizer=regularizers.L2(1e-5))(sentiment_model_inputs)
    
        self.model = tf.keras.Model(inputs=[recurrent_model_inputs,s_input], outputs=dense_rnn_1)
        

           
           
        self.logger.info('Compile model...')
        self.model.compile(optimizer='adam', loss='mae', metrics=[SM3SDistanceMetric()])
        
        # Train model
        self.logger.info('Training model ...')


        x_sentiment_train     = np.reshape(x_sentiment_train, (x_sentiment_train.shape[0], x_sentiment_train.shape[2]))
        self.x_sentiment_test = np.reshape(self.x_sentiment_test, (self.x_sentiment_test.shape[0], self.x_sentiment_test.shape[2]))
        print(x_train.shape)
        print(x_sentiment_train.shape)

        self.model.fit([x_train,x_sentiment_train], y_train,
                        epochs=self.epochs,
                        validation_data=([self.x_test,self.x_sentiment_test], self.y_test),
                        callbacks=[earlystop])
        
        self.logger.info('calibration complete.')
    
        

    def get_num_forecasted_prices(self):
        return self._SequentialModel3StockMultiFactor__OUTPUT_SERIES_NUMBER
    
    
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        sentiment_data =  np.array([self.sentiment_data[len(self.sentiment_data) - 1:len(self.sentiment_data)+1]])
        sentiment_data = np.reshape(sentiment_data, (sentiment_data.shape[0], sentiment_data.shape[2]))
        next_biz_day_input = [ np.array([self.data[len(self.data) - self.lookback:len(self.data)+1]]),sentiment_data]
        
        print("Get forecasted price: {} , {}".format( str(next_biz_day_input[0].shape), str(next_biz_day_input[1].shape) ) ) 
        next_biz_day_price = self.model.predict(next_biz_day_input)[0]
        print(next_biz_day_price[0])
        if (denormalize):
            return ((next_biz_day_price * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price,self.data[len(self.data)-1][0]
        
    
    def compute_predictions(self, denormalize : bool = False):
        if (self.model):
            print('back-testing model ...')
            self.logger.info('back-testing model ...') 
        
            datal            = self.df['Close'].values
            labels           = np.array([datal[i:i+self.__OUTPUT_SERIES_NUMBER] for i in range(self.lookback, len(datal)-self.__OUTPUT_SERIES_NUMBER)])      
            inputs           = np.array([self.data[i:i+self.lookback]           for i in range(0, len(self.data)-self.lookback-self.__OUTPUT_SERIES_NUMBER)])
            sentiment_inputs = np.array([self.sentiment_data[i - 1:i]           for i in range(self.lookback, len(self.sentiment_data)-self.__OUTPUT_SERIES_NUMBER)])
        

            train_size = int(len(inputs) * self.training_percentage)

            x_train, self.x_test = inputs[:train_size], inputs[train_size:]
            y_train, self.y_test = labels[:train_size], labels[train_size:]
            x_sentiment_train, self.x_sentiment_test = sentiment_inputs[:train_size], sentiment_inputs[train_size:]

            x_sentiment_train     = np.reshape(x_sentiment_train, (x_sentiment_train.shape[0], x_sentiment_train.shape[2]))
            self.x_sentiment_test = np.reshape(self.x_sentiment_test, (self.x_sentiment_test.shape[0], self.x_sentiment_test.shape[2]))

            self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback: -self.__OUTPUT_SERIES_NUMBER]
            # Generate prediction
            self.predictions = np.reshape(self.model.predict([self.x_test, self.x_sentiment_test]), (len(self.y_test), self.__OUTPUT_SERIES_NUMBER))

            self.logger.debug("Predictions Shape")
            self.logger.debug(self.predictions.shape)
            
            if (denormalize):
                self.logger.debug("price_denormalization_factor: "+str(self.price_denormalization_factor) )
                self.logger.debug("price_denormalization_sum: "   +str(self.price_denormalization_sum)    )
                self.y_test      = (self.y_test      * self.price_denormalization_factor) + self.price_denormalization_sum 
                self.predictions = (self.predictions * self.price_denormalization_factor) + self.price_denormalization_sum
        
              
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='loss',  # Quantity to be monitored.
                          min_delta=0.00001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=6,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        


from sequential_model_1stock_multifactors import check_data_correlation,evaluate_ticker_distribution        
        
def main():
    init_config()
    print("Running in one ticker mode")
    input_file = PREFIX_PRICE_FETCHER + "PYPL" + ".csv"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    check_data_correlation(FOLDER_MARKET_DATA+input_file)
    evaluate_ticker_distribution(SequentialModel3StockMultiFactor,
                                 FOLDER_MARKET_DATA+input_file,
                                3, 
                                calibrate = True)

    
if __name__ == '__main__':  
    main()
      