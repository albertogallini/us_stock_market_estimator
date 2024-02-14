import os
current_directory = os.getcwd()
print("Current working directory:", current_directory)
new_directory = './notebooks'
os.chdir(new_directory)
updated_directory = os.getcwd()
print("Updated working directory:", updated_directory)


import notebook_constants
from functions import generate_transaction_file,generate_holdings,unroll_daily_holdings,fetch_prices,compute_pnl,plot_char

import yfinance as yf
from datetime import datetime
import pandas as pd
import sys
from openfigipy import OpenFigiClient




if len(sys.argv) > 2 and sys.argv[1] == 'LT':
    print("Load Transactions Data ...")
    generate_transaction_file()
    transactions = pd.read_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE_TICKERS)
    transactions['Date'] = transactions.Operazione.apply(lambda d : datetime.strptime(d, "%d/%m/%Y"))
    transactions.sort_values(by=['Date'])
    holdings = pd.DataFrame(columns=['ticker','isin','date','quantity','ccy'])
    dates    = [ d.to_pydatetime() for d in sorted(transactions['Date'].unique())]

    holdings.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_HOLDINGS)

    print("Generate Holdings ...")
    generate_holdings()

    print("Creating Daily snapshots ...")
    unroll_daily_holdings()
    portfolio = pd.read_csv(notebook_constants.DATA_FOLDER + notebook_constants.PORTFOLIO_HOLDINGS_DAILY)

    print("Load Asset Prices ...")
    if len(sys.argv) > 2:
        fetch_prices(portfolio, mode = sys.argv[2])
    else:
        fetch_prices(portfolio, mode = 'offline')



import os
prefix = notebook_constants.DAILY_HOLDINGS_PREFIX
portfolio_dir = notebook_constants.DATA_FOLDER
files = [filename for filename in os.listdir(portfolio_dir) if filename.startswith(prefix)]
portfolio_performance_daily = None
print("Loading position files ...")
for f in files:    
    position_performance_daily  = compute_pnl(pd.read_csv(portfolio_dir+f))
    portfolio_performance_daily = pd.concat([position_performance_daily,portfolio_performance_daily], ignore_index=True)        
        
plot_char(portfolio_performance_daily)
