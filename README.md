# US Stock Market Price Estimator
This is a simple multi-factor model to estimate the next 1-3 days marekt price based on neural networks.
It is 43 factors, plus sentiment analysis bias. <br>
The price estimator models are implemented using Tensor-Flow, while the Sentiment Analysis rely on 
pretrained Distil-Bert fined tuned on FinGPT dataset from HuggingFace.<br>


![plot](./imgs/arch_d.png?raw=true)

## Models 
There are two basics model based on Dense layer, which use few factors:
1. Sequential Model 1 (price) Stock
2. Sequential Model 1 (price) Stock and rates 

Those are just internal and not really interesting. They are simple multilayer perceptron 
Then there are two other models that can be used to actual price estimate: 

1. Sequential 1 (price) Stock Multifactor, plus its 3 prices estimate version 
2. Transformer 1 (price) Stock Multifactor

### How to use the models

...

### How to fetch the data 

...


## Sentiment Analysis
Sentiment analysis is based on market news fetched from Yahoo Finance and Investing.com. This is for demostration poprpouses and 
no commercial use or distribution of the data is allowed. 

### How to train the Sentiment Analysis model (Distil-Bert)
...




## config.json
the file [config.json](./config.json) contains few info about the file location and the file naming: 
    <li>"FOLDER_PREFIX_DISTRIBUTION_CSV": "/Volumes/reports/distributions__" -> the file name prefix of the output report of a dask job
    <li>"PREFIX_PRICE_FETCHER": "price_fetcher_",  -> the file name prefix of the stock price and volumes .csv files 
    <li>"FOLDER_MARKET_DATA": "/Volumes/data/",  -> the location of the market data
    <li>"FOLDER_REPORD_PDF": "/Volumes/reports/" -> the location of the output of a dusk job. 



## Tools: 
### back_test.py: 
This is an example of a Sequential model mergin a RNN for historical factors and a Dense layer that combine the output of the RNN with the sentiment analysis data
![Alt text](./imgs/back_test.png?width=250&height=150)

This is an example of a Transformer getting as input both historical factors and sentiment historical data. 
![Alt text](./imgs/back_test_t.png?width=250&height=150)

As you can see RNN + Dense model performs better than the Transformer one. This is also confirmed in [this paper]("./docs/Time Series Dense Encoder.pdf")
