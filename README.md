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
The folder configuration is set in the [config.json](./config.json) file :
```
{
    "FOLDER_MARKET_DATA": "/Volumes/data/",
    "FOLDER_REPORD_PDF": "/Volumes/reports/"
}
```
These entries specifies where to get the data and where to store the output report. The data location must be accessible from every Dask worker (e.g.: a shared folder on a local network).
The report folder has to be accessible only from the node the task is launched. <br>
<i>Make sure to copy [static_tickers.list.csv](./static_tickers_list.csv) into your FOLDER_MARKET_DATA.</i>

### How to use the models
The Sequential 1 (price) Stock Multifactor is the only one you can use as real estimator by using [price_estimator.py](./price_estimator/price_estimator.py).
Even if you can modify the code to use other models. To  trigger the report generation type:
```
python price_estimator.py
```
This is based on Dask. So you have to have run a dask scheduler and at least a worker on the same machine. The report generator create 50 scenarios (i.e. calibration/training)
for each ticker in the [ticker.json](./tickers.json) file. So it could be very expensive computation. It is highly recomended to have at least 8 workers having 4 GiB each. 
It is not necesssary to have more than 1 thread per worker. 

All the other models are accessible by the back_tester tool (see "Tool" section).





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
Sentiment model is not exposed as a tool but there are test-units in [sentiment_model.py](./sentiment_model/sentiment_model.py#175).
Throgh those test units you can fine tune, by removing the comment to:<br>
```
def test_fine_tune(self)
```
or start generating your own sentiments scores by<br>

```
def test_score_yahoo(self)
def test_score_investing(self)
```
It is straightforward, looking at these methods to add additional soruce. Also adding additional fine tune dataset should be quite easy.
Every contribution here is welcome. The better and more data we have the better the sentiment analysis is. 



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
you can run it by the following command
<br><br>

``` 
user@myserver path  
$ python back_tester.py IONQ 5 S1SMF
```
<br><br>
Where ```IONQ``` is the ticker, ```5``` is the number of scenarios, ```S1DMF``` is the model name. To check the list of the model name 
look at [const_and_utils.py](./price_estimator/const_and_utils.py#63)<br><br>
<br>
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

The back tester across many strocks shows overall good performance as in most of the case the probabilty to get an error less than 5% is greater than 80%.
As showed in this [report](./docs/back_tester_batch.pdf). Here below a simple page from it: <br>
![Alt text](./imgs/back_test_mult.png?width=250&height=150)
<br>
The report has been built trainig the model multiple times. Each curve show the probabilty density for each calibration.  It easy to see there are good calibrations and less good calibrations. 
To run the report you can can use the following command
<br><br>

``` 
user@myserver path  
$ python back_tester.py ticker.json 5 S1SMF
```
<br><br>


### quality_checker.py: 
This tool compare the estimate of day t for t+1 with the actuals and offer an overall evaulation of the estimate done on day t. 
The tool assume on both day t and t+1 the price_estimator report has been generated. Additionally the ticker set might change between t and t+1
so the tool check only the tickers that are evaluated both at t and t+1.<br>
The tool gets in input the day t, e.g.: 
<br><br>

``` 
user@myserver path  
$ python quality_checker.py 2024-01-12
```
<br><br>
and returns a chart that can be zoomed in and out of actual vs estiamte pluse the 1,2,3 sigma areas.  Additionally return the ditribution of the difference between 
the estimate return and the real return on %. See picture below:<br>
![Alt text](./imgs/quality_checker.png?width=250&height=150)




