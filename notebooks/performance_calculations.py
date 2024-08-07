#%%
import pandas as pd
import datetime as dt
from datetime import datetime
import  notebook_constants 
import  functions

#%%
    
functions.unroll_daily_holdings()
                
#%%

portfolio = pd.read_csv(notebook_constants.DATA_FOLDER + notebook_constants.PORTFOLIO_HOLDINGS_DAILY)
portfolio = portfolio[portfolio.ticker == "PFE"]
functions.fetch_prices(portfolio, mode="online")


#%%
import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np 
import os

prefix = notebook_constants.DAILY_HOLDINGS_PREFIX
portfolio_dir = notebook_constants.DATA_FOLDER
files = [filename for filename in os.listdir(portfolio_dir) if filename.startswith(prefix)]
portfolio_performance_daily = None

print("Loading position files ...")

for f in files:
    #print("Loading {}.".format(f))    
    position_performance_daily  = functions.compute_pnl(pd.read_csv(portfolio_dir+f))
    #display(position_performance_daily)
    portfolio_performance_daily = pd.concat([position_performance_daily,portfolio_performance_daily], ignore_index=True)        

#portfolio_performance_daily.to_csv("test.csv")
    
#%%
import functions
functions.plot_char(portfolio_performance_daily)



# %%