'''
Created on Nov 28, 2023

@author: albertogallini
'''

import tensorflow as tf
from   tensorflow.keras import regularizers
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
from  sequential_model_1stock_and_rates import SequentialModel1StockAndRates
from  sequential_model_1stock import evaluate_ticker_distribution



class SequentialModel1StockMultiFactor(SequentialModel1StockAndRates):
    
    def __init__(self,
                input_data_price_csv : str,
                input_data_rates_csv : str ,
                input_fear_and_greed_csv : str,
                lookback : int  = 14,
                epochs : int = 32,
                training_percentage : float = 0.90,
                use_lstm : bool = False,
                logger: logging.Logger = None ) :
        
        if (logger == None):
            self.logger= logging.getLogger(input_data_price_csv+'_calibrate_model_sequential_model.logger')
            file_handler = logging.FileHandler(input_data_price_csv+'_calibrate_model_sequential_model.log')
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
            
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
        self.ticker = get_ticker(input_data_price_csv)
        self.use_lstm = use_lstm
        
            
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
        data_dir_path = FOLDER_MARKET_DATA

        self.df = pd.read_csv(input_data_price_csv)
        self.df['Date'] = self.df['Date'].apply(lambda l :  l.split(" ")[0])
        
        self.df_rates           = pd.read_csv(input_data_rates_csv)
        self.df_fear_and_greed  = pd.read_csv(input_fear_and_greed_csv)
        self.sentiment_score    = pd.read_csv("news_scores.csv")[["Date","Scores"]].groupby('Date')['Scores'].mean().to_frame().reset_index()
        
        self.df = self.df.merge(self.df_rates, on='Date', how='left')
        self.df = self.df.merge(self.df_fear_and_greed, on='Date', how='left')
        self.df = self.df.merge(self.sentiment_score, on='Date', how='left')

        index_sub_sector_files = fnmatch.filter(os.listdir(data_dir_path), 'index_sub_sector_*')
        self.index_sub_sector_price = dict()
        for f in  index_sub_sector_files :
            self.index_sub_sector_price[str(get_ticker(f))] = pd.read_csv(data_dir_path+f)
        for k in self.index_sub_sector_price.keys():
            self.index_sub_sector_price[k].rename(columns={'Close': k}, inplace=True)
            self.index_sub_sector_price[k]['Date'] = self.index_sub_sector_price[k]['Date'].apply(lambda l :  l.split(" ")[0])
            self.df = self.df.merge(self.index_sub_sector_price[k][['Date',k]], on='Date', how='left')

       
        # Assuming you have a DataFrame `df` with 'price' and 'volume' columns
        # Normalize your data (very important for neural networks)
        self.logger.info('Preparing data ...')
        
        if(len(self.df['Close'].values) != len(self.df['Volume'].values)):
            self.df = pd.DataFrame()
            return
        
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        self.df = fill_value('Close',self.df)
        self.df = fill_value('Volume',self.df)
        self.df = fill_value('3 Mo',self.df)
        self.df = fill_value('10 Yr',self.df)
        self.df = fill_value('5 Yr',self.df)
        self.df = fill_value('1 Yr',self.df)
        self.df = fill_value('Fear Greed',self.df)

         # clean up Scores data :
        self.df['Scores_pxy'] = (self.df['Fear Greed'].pct_change() + 1)  * 0.5 # some recaling is needed to makes inferred valuea comparable to the actual ones.
        self.df['Scores_pxy'] = self.df['Scores_pxy'].apply(lambda x :  x if abs(x) < 1 else 0.5)
        self.df['Scores'].fillna(self.df['Scores_pxy'], inplace=True)
        self.df['Scores'] = self.df['Scores']
   
        self.price_denormalization_factor = ( self.df['Close'].max() -  self.df['Close'].min() ) 
        self.price_denormalization_sum    =   self.df['Close'].min()
        self.df_not_normalized = self.df.copy()
        

        ###### max min normalization ######
        self.logger.info('Normalizing ...')
        
        self.df['Close']  = ( self.df['Close']  -  self.df['Close'].min())   / ( self.df['Close'].max()  -  self.df['Close'].min())
        self.df['Volume'] = ( self.df['Volume'] -  self.df['Volume'].min())  / ( self.df['Volume'].max() -  self.df['Volume'].min())
        
        self.df['3 Mo']   = (( self.df['3 Mo']  -  self.df['3 Mo'].min())    / ( self.df['3 Mo'].max()   -  self.df['3 Mo'].min()))    * 0.2
        self.df['1 Yr']   = (( self.df['1 Yr']  -  self.df['1 Yr'].min())    / ( self.df['1 Yr'].max()   -  self.df['1 Yr'].min()))    * 0.6
        self.df['5 Yr']   = (( self.df['5 Yr']  -  self.df['5 Yr'].min())    / ( self.df['5 Yr'].max()   -  self.df['5 Yr'].min()))    * 0.6
        self.df['10 Yr']  = (( self.df['10 Yr'] -  self.df['10 Yr'].min())   / ( self.df['10 Yr'].max()  -  self.df['10 Yr'].min()))   * 1


        self.df['Fear Greed'] = (( self.df['Fear Greed'] -  self.df['Fear Greed'].min())   / ( self.df['Fear Greed'].max()  -  self.df['Fear Greed'].min()))  * 1
        

        for k in self.index_sub_sector_price.keys():
            self.df[k] = (( self.df[k] -  self.df[k].min())  / ( self.df[k].max() -  self.df[k].min())) * 0.1
        ##################################    
            
        self.df[["Scores","Fear Greed","Scores_pxy"]].to_csv("debug.csv")


        # inpout factor list:
        self.calibration_factors_list = ['Close', 'Volume','3 Mo','1 Yr','5 Yr','10 Yr'] + list(self.index_sub_sector_price.keys())
        self.sentiment_factors_list   = ['Fear Greed','Scores'] 

        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        # time data:
        self.time = self.df['Date']
        # Convert DataFrame to numpy array
        self.data           =  self.df[ self.calibration_factors_list ].values
        self.sentiment_data =  self.df[ self.sentiment_factors_list ].values

        #adjust lookback in funtion of the price volatiltiy: longer lookback means less vol:
        self.price_vol     = self.df['Close'].std() * 2.5
        print("Adjsuting lookback {} --> 1 - 2.5 x norm. price stddev {:.5f}, adjusted lookback: {}".format(self.lookback,(-self.price_vol+1), int((-self.price_vol+1)*self.lookback)))
        self.lookback = int((-self.price_vol+1)*self.lookback)

        

    
    def calibrate_model (self):
       
        if (self.df.empty):
            self.logger.info('No Valid data')
            return
        
        inputs           = np.array([self.data[i - self.lookback:i]           for i in range(self.lookback, len(self.data))])
        sentiment_inputs = np.array([self.sentiment_data[i - 1:i] for i in range(self.lookback, len(self.sentiment_data))])
        labels           =  self.df['Close'][self.lookback:]
        factor_num       = len(self.calibration_factors_list)
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
            
        dense_rnn = tf.keras.layers.Dense(units = self.lookback * factor_num, activation ='linear',
                                             kernel_regularizer = regularizers.L1L2(l1 = 1e-4*(-self.price_vol+1), l2 = 1e-4*(-self.price_vol+1)),
                                             bias_regularizer   = regularizers.L2(1e-4),
                                             activity_regularizer = regularizers.L2(1e-5))(recurrent_nn)
        
        dense_rnn_1 = tf.keras.layers.Dense(units = 1, activation ='linear',
                                             kernel_regularizer   = regularizers.L1L2(l1 = 1e-4*(-self.price_vol+1), l2 = 1e-4*(-self.price_vol+1)),
                                             bias_regularizer     = regularizers.L2(1e-4),
                                             activity_regularizer =regularizers.L2(1e-5))(dense_rnn)
        
        dense_rnn_1_reshaped = tf.keras.layers.Reshape((1, 1, ))(dense_rnn_1)
        
        
        # Create the second stage of the model : sentiment analysis and sectr correlation
        s_input = tf.keras.Input(shape=(1,sentiment_factor_num))
    

        # Combine the outputs of the LSTM model and the score input
        sentiment_model_inputs  = tf.keras.layers.Concatenate(axis=2)([dense_rnn_1_reshaped, s_input])
        
        sentiment_dense = tf.keras.layers.Dense(10 * sentiment_factor_num, activation='linear',
                                             kernel_regularizer=regularizers.L1L2(l1=1e-4*(-self.price_vol+1), l2=1e-4*(-self.price_vol+1)),
                                             bias_regularizer=regularizers.L2(1e-4),
                                             activity_regularizer=regularizers.L2(1e-5))(sentiment_model_inputs)
        
        sentiment_predictions = tf.keras.layers.Dense(1, activation='linear',
                                             kernel_regularizer=regularizers.L1L2(l1=1e-4*(-self.price_vol+1), l2=1e-4*(-self.price_vol+1)),
                                             bias_regularizer=regularizers.L2(1e-4),
                                             activity_regularizer=regularizers.L2(1e-5))(sentiment_dense)
    
        self.model = tf.keras.Model(inputs=[recurrent_model_inputs,s_input], outputs=sentiment_predictions)
           
           
        self.logger.info('Compile model...')
        self.model.compile(optimizer='adam', loss='mae', metrics=['mse'])
        
        # Train model
        self.logger.info('Training model ...')
        self.model.fit([x_train,x_sentiment_train], y_train,
                        epochs=self.epochs,
                        validation_data=([self.x_test,self.x_sentiment_test], self.y_test),
                        callbacks=[earlystop])
        
        self.logger.info('calibration complete.')
        
        
    def get_forecasted_price(self, denormalize = True):
        self.logger.info('predict next day market price ') 
        next_biz_day_input = [ np.array([self.data[len(self.data) - self.lookback:len(self.data)+1]]),
                               np.array([self.sentiment_data[len(self.sentiment_data) - 1:len(self.sentiment_data)+1]])
                             ]
        print("Get forecasted price: {} , {}".format( str(next_biz_day_input[0].shape), str(next_biz_day_input[1].shape) ) ) 
        next_biz_day_price = self.model.predict(next_biz_day_input)[0]
        print(next_biz_day_price[0])
        if (denormalize):
            return ((next_biz_day_price[0] * self.price_denormalization_factor) + self.price_denormalization_sum ), ((self.data[len(self.data)-1][0]* self.price_denormalization_factor) + self.price_denormalization_sum )
        else:
            return next_biz_day_price[0],self.data[len(self.data)-1][0]
        
        
    def get_num_forecasted_prices(self):
        return 1
    

    def compute_predictions(self, denormalize : bool = False):
        if (self.model):
            self.logger.info('back-testing model ...') 
        
            inputs           = np.array([self.data[i - self.lookback:i]  for i in range(self.lookback, len(self.data))])
            sentiment_inputs = np.array([self.sentiment_data[i - 1:i] for i in range(self.lookback, len(self.sentiment_data))])
            labels =  self.df['Close'][self.lookback:]
            train_size = int(len(inputs) * self.training_percentage)

            x_train, self.x_test = inputs[:train_size], inputs[train_size:]
            y_train, self.y_test = labels[:train_size], labels[train_size:]
            x_sentiment_train, self.x_sentiment_test = sentiment_inputs[:train_size], sentiment_inputs[train_size:]

            self.train_time, self.test_time = self.time[:train_size], self.time[train_size+self.lookback:]
            # Generate prediction
            self.predictions = np.squeeze(self.model.predict([self.x_test, self.x_sentiment_test]),  axis=2)
            
            self.logger.debug("Predictions Shape")
            self.logger.debug(self.predictions.shape)
            print(self.predictions.shape)
            
            if (denormalize):
                self.logger.debug("price_denormalization_factor: "+str(self.price_denormalization_factor) )
                self.logger.debug("price_denormalization_sum: "   +str(self.price_denormalization_sum)    )
                self.y_test      = (self.y_test      * self.price_denormalization_factor) + self.price_denormalization_sum 
                self.predictions = (self.predictions * self.price_denormalization_factor) + self.price_denormalization_sum
        
   
                        
        
        
from tensorflow.keras.callbacks import EarlyStopping

# Define your early stopping criteria
earlystop = EarlyStopping(monitor='val_loss',  # Quantity to be monitored.
                          min_delta=0.0001,  # Minimum change in the monitored quantity to qualify as an improvement.
                          patience=6,  # Number of epochs with no improvement after which training will be stopped.
                          verbose=1,  # Verbosity mode.
                          mode='auto')  # Direction of improvement is inferred        
        
        

def evaluate_ticker(class_type, input_file:str, calibrate: bool, scenario_id: int, model_date: str):
    print("Calibrate {} ...".format(class_type))
    seq_model = create_instance_of_class(class_type,   
                                         input_data_price_csv = input_file,
                                         input_data_rates_csv = FOLDER_MARKET_DATA+"/usd_rates.csv",
                                         input_fear_and_greed_csv = FOLDER_MARKET_DATA+"/fear_and_greed.csv") 
    
    if calibrate:
        seq_model.calibrate_model()
    else:
        seq_model.load_model(path=FOLDER_REPORD_PDF+model_date+"/", scenario_id=str(scenario_id))
        
    if (seq_model.model == None):
        return None, None
    ticker = get_ticker(input_file)
    return ticker, seq_model 

def evaluate_ticker_distribution(class_type, input_file:str, scenarios: int = 10, calibrate : bool = True, model_date: str = "15-12-2023" ):
    from scipy.stats import norm,lognorm
    
    fprices_ml = list()
    for i in range (0,scenarios):
        fprices_ml.append(list())
        ticker, seq_model = evaluate_ticker(class_type, input_file=input_file, calibrate=calibrate, scenario_id=i, model_date=model_date)
        p_list,p_1 = seq_model.get_forecasted_price()
        for j in p_list:
            fprices_ml[i].append(j)
      
    num_sub_plots = seq_model.get_num_forecasted_prices()
    fig, axs = plt.subplots(num_sub_plots  if num_sub_plots > 1  else 2)

    for i in range (0,num_sub_plots):

        fprices = [fprices_ml[s][i] for s in range (0,scenarios) ]
   
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
    
    
    
def check_data_correlation(input_file:str, rates = True, indexes = False, fng = True):
    sm1s = SequentialModel1StockMultiFactor(input_data_price_csv = input_file,
                                            input_data_rates_csv = FOLDER_MARKET_DATA+FILE_NAME_RATES,
                                            input_fear_and_greed_csv = FOLDER_MARKET_DATA+FILE_NAME_FNG) 
    
    print (sm1s.df_not_normalized[['Close']].describe())


    if (rates):
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='3 Mo', color = "green")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='4 Mo', color = "blue")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='6 Mo', color = "green")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='1 Yr', color = "red")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='2 Yr', color = "red")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='5 Yr', color = "red")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='7 Yr', color = "red")
        sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='10 Yr', color = "red")

    if(indexes):
        for t in ['IYK', 'ITA', 'RTH', 'XRT', 'XHB', 'XLY', 'IBD', 'XRT', 'VDC', 'IYC', 'IYC', 'VDE', 'XOP', 'XLE', 'VCR', 'XES', 'KBE', 'IHF',
                        'IYF', 'KRE', 'IBB', 'XLV', 'XPH', 'XLP', 'XLB', 'USD', 'IYJ', 'XLI', 'IYT', 'IGV', 'IYW', 'XLB', 'XME', 'IP', 'IYR', 'IYR', 'IYT', 
                        'ITB', 'IYR', 'IVV', 'IYZ', 'XLU', 'XLU', 'IDU'] :
            sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y="['"+t+"']", color = "red")
    
    if(fng):
       sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='Fear Greed', color = "red")
       sm1s.df_not_normalized.plot(kind="scatter",  x="Close", y='Scores', color = "red")
       sm1s.df_not_normalized.plot(kind="scatter",  x="Fear Greed", y='Scores', color = "red")
        
    plt.show()
        
        
def main():
    init_config()
    print("Running in one ticker mode")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #check_data_correlation(FOLDER_MARKET_DATA+"price_fetcher_IONQ.csv")
    evaluate_ticker_distribution(SequentialModel1StockMultiFactor, FOLDER_MARKET_DATA + "price_fetcher_PYPL.csv",
                                3,
                                calibrate = True,
                                model_date= None)
   
    
if __name__ == '__main__':  
    main()
      