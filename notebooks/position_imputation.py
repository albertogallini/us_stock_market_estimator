# %%
import notebook_constants
import pandas as pd
import yfinance as yf
import time
import pandas as pd
from openfigipy import OpenFigiClient

#%%
def get_ticker(isin):
    try:
        time.sleep(5)
        ofc = OpenFigiClient()
        ofc.connect()  # Establish a requests session

        # Create a dataframe with ISINs (you can modify this with your own data)
        df = pd.DataFrame({
            'idType': ['ID_ISIN'],
            'idValue': [isin],  # Replace with your ISIN
        })

        # Map ISINs to tickers
        result = ofc.map(df)
        print(result['ticker'][0])
        return result['ticker'][0]
    except Exception as e:
            print(" No ticker found : {} ".format(e))
            return ""


# %%
def generate_transaction_file():
    transactions = pd.read_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE)
    assets = pd.DataFrame(columns=['ISIN']) 
    assets['ISIN']   = transactions['ISIN'].unique()
    assets['ticker'] = assets['ISIN'].apply(lambda n : get_ticker(n))
    assets.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_ASSETS_FILE)
    transactions = transactions.merge(assets, on='ISIN', how='left')
    transactions.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE_TICKERS)

generate_transaction_file()

# %%
import pandas as pd
from datetime import datetime
import notebook_constants


transactions= pd.read_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE_TICKERS)
transactions['Date'] = transactions.Operazione.apply(lambda d : datetime.strptime(d, "%d/%m/%Y"))
transactions.sort_values(by=['Date'])
holdings = pd.DataFrame(columns=['ticker','isin','date','quantity','ccy'])
dates    = [ d.to_pydatetime() for d in sorted(transactions['Date'].unique())]


for d in dates: 
    t_snapshot = transactions[transactions['Date'].apply(lambda td: td.to_pydatetime() == d)]
    
    for tindex, t  in t_snapshot.iterrows():
        last_q = 0
        last_d = datetime(1999,1,1)

        if ( holdings.shape[0] > 0):
            #display(holdings)
            if len(holdings['ticker']) == 0:
                continue
            asset_holds = holdings[holdings['ticker'] == t['ticker']]
            if len(asset_holds) > 0:
                last_q = float(asset_holds['quantity'].iloc[-1])
                last_d = asset_holds['date'].iloc[-1]
    
        if t['Date'] != last_d:
            if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'A'):
                new_hold = {'ticker': t[notebook_constants.TRANSACTION_FIELD_TICKER],
                            'isin':   t[notebook_constants.TRANSACTION_FIELD_ISIN],
                            'date':   t[notebook_constants.TRANSACTION_FIELD_DATE],
                            'quantity':last_q+float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY].replace(",","")),
                            'ccy':    t[notebook_constants.TRANSACTION_FIELD_CCY]}
                holdings.loc[len(holdings)] = new_hold
            
            if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'V'):
                new_hold = {'ticker': t[notebook_constants.TRANSACTION_FIELD_TICKER],
                            'isin':   t[notebook_constants.TRANSACTION_FIELD_ISIN],
                            'date':   t[notebook_constants.TRANSACTION_FIELD_DATE],
                            'quantity':last_q-float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY].replace(",","")),
                            'ccy':    t[notebook_constants.TRANSACTION_FIELD_CCY]}
                holdings.loc[len(holdings)] = new_hold

        else:

            if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'A'):
                new_amt = last_q + float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY].replace(",",""))
                print("buy {} : {} + {} = {}".format(t[notebook_constants.TRANSACTION_FIELD_TICKER],last_q, t[notebook_constants.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                filtered_rows = holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]]
                last_row_index = filtered_rows.index[-1]
                holdings.at[last_row_index, 'quantity'] = new_amt
                #display(holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]])

            if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'V'):
                new_amt = last_q - float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY].replace(",",""))
                print("sell {} : {} - {} = {}".format(t[notebook_constants.TRANSACTION_FIELD_TICKER],last_q, t[notebook_constants.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                filtered_rows = holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]]
                last_row_index = filtered_rows.index[-1]
                holdings.at[last_row_index, 'quantity'] = new_amt
                #display(holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]])

        
holdings.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_HOLDINGS)




        
     
     
     


# %%
