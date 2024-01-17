#  U.S. Stock Market Price Estimator

This is a simple multi-factor model to estimate the next 1-3 days marekt price based on neural networks.
It is 43 factors, plus sentiment analysis bias. <br>
The price estimator models are implemented using Tensor-Flow, while the Sentiment Analysis rely on 
pretrained Distil-Bert fined tuned on FinGPT dataset from HuggingFace.<br>


![plot](./imgs/arch_d.png?raw=true)
<p style="text-align: center;"><b>fig.1</b></p>


## 1. Models overview

There are two basics model based on Dense layer, which use few factors:
1.  <i>Sequential Model 1 (price) Stock</i>: this uses only prices and volumes.
2.  <i>Sequential Model 1 (price) Stock and Rates</i>: this uses prices, volumes and rates.

Those are just internal and not really interesting. They are simple multilayer perceptron 
Then there are two other models that can be used to actual price estimate: 

3. <i>Sequential 1 (price) Stock Multifactor, plus a 3 prices estimate version</i>: this is the model describe in the opening <b>fig. 1.</b>
4. <i>Transformer 1 (price) Stock Multifactor </i>: this is a model based on trasformers and the same input of the previous one

Model #3 can also be run using LSTM in place of RNN, but tests show RNN produce closer to actual results. 


## 2. How to fetch the data 

The data can be fetched trough on python [script](./market_price_fetcher/data_fetcher.py) that takes care of 
- Stocks last close price (SPX, NASDAQ and NYSE)  - Yahoo  Finiance
- US treasury yield crv tenors: '1 Mo','2 Mo','3 Mo','4 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr  - treasury.gov
- Feed & Greed index value - dataviz.cnn.io
- GICS subsector index prices - Yahoo Finanace
 <br>

Sentiment data, i.e. news are fetched throug web scarpers by using BeautifulSoup and Selenium. There are two sources 
 - news from [yahoo finance news](./scrapers/yahoof_scraper.py)
 - news from [investing.com news](./scrapers/investing_dot_com.py)
 <br>

 The news are fetched and interpreted by the sentiment model engine that produce a score store in a file that must be copied in the 
 same data folder where the price and the other market data are stored.
 <br>See next section on how to configure the folders.<br>
 


## 3. Folder configurations


### How to use the models
### TODO ...



## Sentiment Analysis
Sentiment analysis is based on market news fetched from Yahoo Finance and Investing.com. This is for demostration poprpouses and 
no commercial use or distribution of the data is allowed.
<br>
The sentiment model engine is based on [Distil-BERT from HuggingFace](https://huggingface.co/docs/transformers/model_doc/distilbert) that can be trained on 
 two different datasets: 
- [eengel7](https://huggingface.co/datasets/eengel7/sentiment_analysis_training)
- [FinGPT](https://huggingface.co/FinGPT/fingpt-forecaster_dow30_llama2-7b_lora)
<br>

Eengel7 is very small dataset and can be used to quickly fine tune Distil-BERT and test it. For better score FinGPT is more appropriate but the fine tuning might take several hours.
<br> 

### How to train the Sentiment Analysis model (Distil-Bert)
### TODO ...



## config.json
the file [config.json](./config.json) contains few info about the file location and the file naming: 
    <li>"FOLDER_PREFIX_DISTRIBUTION_CSV": "/Volumes/reports/distributions__" -> the file name prefix of the output report of a dask job
    <li>"PREFIX_PRICE_FETCHER": "price_fetcher_",  -> the file name prefix of the stock price and volumes .csv files 
    <li>"FOLDER_MARKET_DATA": "/Volumes/data/",  -> the location of the market data
    <li>"FOLDER_REPORD_PDF": "/Volumes/reports/" -> the location of the output of a dusk job. 



## Tools: 
### back_test.py: 
This is an example of a Sequential model mergin a RNN for historical factors and a Dense layer that combine the output of the RNN with the sentiment analysis data<br>
![Alt text](./imgs/back_test.png?width=250&height=150)

This is an example of a Sequential model mergin a LSTM for historical factors and a Dense layer that combine the output of the RNN with the sentiment analysis data<br>
![Alt text](./imgs/back_test_lstm.png?width=250&height=150)

This is an example of a Transformer getting as input both historical factors and sentiment historical data.<br>
![Alt text](./imgs/back_test_t.png?width=250&height=150)
<br>
As you can see RNN + Dense model performs better than the LSTM and Transformer one. This is also confirmed in [this paper](./docs/1.pdf). 
LSTM show a consistent underestimation in this example, while Transformes are more affected by the typcal 1-day shift on the historical series estimation.<br> 
This effect is quite frequent as the previous day value is a good esitimator of the current day value ( meaning the correlation betwenn the current day price and 
the yesterday price si very high).
Sentiment Analysis and rates change have the purpouse to reduce the similarity with the previous day price that mis induced as (over-fitting) effect of the training. 

Back tester across many strocks show overall good performance as in most of the case the probabilty to get an error less than 5% is greater than 80%.
As shoed in this [report](./docs/back_tester_batch.pdf). Here below a simple page from it: 
![Alt text](./imgs/back_test_mult.png?width=250&height=150)
The report has been bult trainig the model multiple time. Each curve show the probabilty desnsity for each calibration.  It easy to see there are good calibration and less good calibrations. 


