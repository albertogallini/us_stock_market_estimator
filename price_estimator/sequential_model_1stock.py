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

from  const_and_utils import *



class SequentialModel1Stock:
    
    def __init__(self, input_data_csv : str , lookback : int  = 14, epochs : int = 4, training_percentage : float = 0.90, logger: logging.Logger = None ) :
        
        if (logger == None):
            self.logger= logging.getLogger(input_data_csv+'_calibrate_model_sequential_model.logger')
            file_handler = logging.FileHandler(input_data_csv+'_calibrate_model_sequential_model.log')
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
            
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
        
        self.df = df = pd.read_csv(input_data_csv)
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
        
        self.df['Close']  = ( self.df['Close'] -  self.df['Close'].min())   / ( self.df['Close'].max() -  self.df['Close'].min())
        self.df['Volume'] = ( self.df['Volume'] -  self.df['Volume'].min()) / ( self.df['Volume'].max() -  self.df['Volume'].min())
        
        self.time = self.df['Date'].apply(lambda l :  l.split(" ")[0])
        # Convert DataFrame to numpy array
        self.data =  self.df[['Close', 'Volume']].values
        
        
    
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
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2]))
        
        self.logger.debug(x_train.shape)
        self.logger.debug(self.x_test.shape)
        
        self.logger.info('Defining model ...')
        
        # Define model
        if mode == "Rnn":
            self.logger.info('RNN model ...')
            self.model = tf.keras.Sequential()
            self.model.add(tf.keras.layers.Reshape((-1, 1)))
            self.model.add(tf.keras.layers.SimpleRNN(self.lookback*16, activation='tanh')) #, recurrent_activation='sigmoid'))
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
            self.predictions = self.model.predict(self.x_test)
            self.logger.debug("Predictions Shape")
            self.logger.debug(self.predictions.shape)
            
            if (denormalize):
                self.logger.debug("price_denormalization_factor: "+str(self.price_denormalization_factor) )
                self.logger.debug("price_denormalization_sum: "   +str(self.price_denormalization_sum)    )
                self.y_test      = (self.y_test      * self.price_denormalization_factor) + self.price_denormalization_sum 
                self.predictions = (self.predictions * self.price_denormalization_factor) + self.price_denormalization_sum
                
        
        
    def show_prediction_vs_actual(self, ticker:str):
        from matplotlib.ticker import FixedLocator
        
        self.compute_predictions(denormalize = True)

        # Assuming y_test and predictions are of the same length
        # Plot the actual prices and the predicted prices
        plt.switch_backend('QtAgg')  # For QtAgg backend
        plt.figure(figsize=(14, 5))
        plt.plot(self.test_time, self.y_test, color='blue', label='Actual prices')
        plt.plot(self.test_time, self.predictions, color='red', label='Predicted prices')
        plt.title('Actual vs Predicted Prices ' + str(ticker) )
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.xticks(rotation=90)  # rotates labels by 45 degrees
        plt.legend()
        plt.show()
        
            
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        next_biz_day_input = np.array([self.data[len(self.data) - self.lookback:len(self.data)+1] for i in range(self.lookback, len(self.data))])
        # Reshape inputs for dense layer
        next_biz_day_input = next_biz_day_input.reshape((next_biz_day_input.shape[0], next_biz_day_input.shape[1] * next_biz_day_input.shape[2]))
       
        next_biz_day_price = self.model.predict(next_biz_day_input)
        if (denormalize):
            return ((next_biz_day_price[0] * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price[0],self.data[len(self.data)-1][0]
        
        
    def save_model(self, path: str, scenario_id: str):
        print("Save Model")
        print(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".model")
        print(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".rfactors")
        self.model.save(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".model")
        
    def load_model(self, path: str, scenario_id: str):
        print("Load Model")
        print(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".model")
        print(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".rfactors")
        self.model = tf.keras.models.load_model(path+PREFIX_MODEL_TRAINING+self.ticker[0]+"__"+scenario_id+".model")
        self.lookback = self.model.input_shape[0][1] # this must pe inherited as the lookback is in funtion of the price volatitly and can change.


    
    def plot_model(self, file_name:str = 'model.png'):
        import pydot
        dot = pydot.Dot()
        dot.set_type('digraph')

        # Traverse through the layers and add them to the dot object
        for i, layer in enumerate(self.model.layers):
            label = f"{layer.name}\ninput: {layer.input_shape}\noutput: {layer.output_shape}"
            node = pydot.Node(str(i), label=label)
            dot.add_node(node)
            if i > 0:
                edge = pydot.Edge(str(i-1), str(i))
                dot.add_edge(edge)

        # Save the dot object as a PNG image
        dot.write_png(file_name)
        
        
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='val_loss',  # Quantity to be monitored.
                          min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=6,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        


def evaluate_ticker(input_file:str):
    sm1s = SequentialModel1Stock(input_data_csv=input_file) 
    sm1s.calibrate_model()
    if (sm1s.model == None):
        return None, None
    ticker = get_ticker(input_file)
    return ticker, sm1s


        
def evaluate_ticker_distribution(input_file:str, scenarios: int = 10 ):
    from scipy.stats import norm

    fprices = list()
    for i in range (0,scenarios):
       ticker, sm1s = evaluate_ticker(input_file)
       p,p_1 = sm1s.get_forecasted_price()
       fprices.append(p[0])
       
    mu, std = norm.fit(fprices)
    
    plt.hist(fprices, bins=scenarios, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(mu - std, mu + std, 100)
    xx = np.linspace(mu - 3*std, mu + 3*std, 100)
    p = norm.pdf(x, mu, std)
    pp = norm.pdf(xx, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.plot(xx, pp, 'k', linewidth=1, linestyle='dashed')
    
    pattern = r"_([^_]*)\."
    ticker = re.findall(pattern, input_file)
    
    title = str(ticker[0]) + " - Fit results: mu = %.2f [%.2f,%.2f],  std = %.2f, prev price: %2f"  % (mu, mu - std, mu + std, std, p_1)
    if (mu < p_1):
        plt.title(title, color="red")
    if (mu > p_1):
        plt.title(title, color="green")
    if (mu == p_1):
        plt.title(title, color="black")
   
    plt.show()

    # Print the 3 sigma
    print("3 sigma: ", 3*std)


    
    
        
def main():

    print("Running in oneticker mode")
    init_config()
    evaluate_ticker_distribution(FOLDER_MARKET_DATA+"/price_fetcher_IONS.csv", 10)
   
    
if __name__ == '__main__':  
    main()

            