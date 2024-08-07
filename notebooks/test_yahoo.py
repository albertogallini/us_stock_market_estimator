# %%

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import time
import pandas as pd
ticker = "PFE"
start_date = (datetime.today()-timedelta(days=10000) ).date()
end_date =  (datetime.today()+timedelta(days=1) ).date() #ticker_dates[len(ticker_dates)-1]
print(end_date)
tickerData = yf.Ticker(ticker)
if "quoteType" in tickerData.info:
        print("Quote type : {}".format(tickerData.info["quoteType"]))

ticker_prices =  tickerData.history(period='1d', start=start_date, end=end_date, auto_adjust=False)
ticker_prices.reset_index(inplace=True)
ticker_prices
# %%
