# %%
import notebook_constants
import pandas as pd
import yfinance as yf
import time
import pandas as pd
from openfigipy import OpenFigiClient


def get_ticker(isin):
    try:
        time.sleep(1)
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
            print(" {} No ticker found : {} ".format(isin,e))
            return ""



def load_fineco_transaction_file (path:str) -> pd.DataFrame :
    df = pd.read_excel('./../file.xls')
    # Use row 5 as column names
    df.columns = df.iloc[4]  # Access row 5 (index 4) for column names
    # Remove row 5 after using it for column names
    df = df.iloc[5:]  
    return df



def generate_transaction_file(path:str = "./file.xls"):
    try:
        transactions = load_fineco_transaction_file(path)
        transactions.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE)
        assets = pd.DataFrame(columns=['ISIN']) 
        assets['ISIN']   = transactions['ISIN'].unique()
        assets['ticker'] = assets['ISIN'].apply(lambda n : get_ticker(n))
        assets.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_ASSETS_FILE)
        transactions = transactions.merge(assets, on='ISIN', how='left')
        transactions.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.TRANSACTION_FILE_TICKERS)
    except Exception as e:
        print("Error in loading the transactions: {}".format(e))


import pandas as pd
from datetime import datetime
import notebook_constants

def  generate_holdings():
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
                                'quantity':last_q+float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY]),
                                'ccy':    t[notebook_constants.TRANSACTION_FIELD_CCY]}
                    holdings.loc[len(holdings)] = new_hold
                
                if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'V'):
                    new_hold = {'ticker': t[notebook_constants.TRANSACTION_FIELD_TICKER],
                                'isin':   t[notebook_constants.TRANSACTION_FIELD_ISIN],
                                'date':   t[notebook_constants.TRANSACTION_FIELD_DATE],
                                'quantity':last_q-float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY]),
                                'ccy':    t[notebook_constants.TRANSACTION_FIELD_CCY]}
                    holdings.loc[len(holdings)] = new_hold

            else:

                if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'A'):
                    new_amt = last_q + float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY])
                    print("buy {} : {} + {} = {}".format(t[notebook_constants.TRANSACTION_FIELD_TICKER],last_q, t[notebook_constants.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                    filtered_rows = holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]]
                    last_row_index = filtered_rows.index[-1]
                    holdings.at[last_row_index, 'quantity'] = new_amt
                    #display(holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]])

                if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'V'):
                    new_amt = last_q - float(t[notebook_constants.TRANSACTION_FIELD_QUANTITIY])
                    print("sell {} : {} - {} = {}".format(t[notebook_constants.TRANSACTION_FIELD_TICKER],last_q, t[notebook_constants.TRANSACTION_FIELD_QUANTITIY], new_amt ))
                    filtered_rows = holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]]
                    last_row_index = filtered_rows.index[-1]
                    holdings.at[last_row_index, 'quantity'] = new_amt
                    #display(holdings[holdings['ticker'] == t[notebook_constants.TRANSACTION_FIELD_TICKER]])

            
    holdings.to_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_HOLDINGS)


    
def unroll_daily_holdings():
    holdings =  pd.read_csv(notebook_constants.DATA_FOLDER+notebook_constants.PORTFOLIO_HOLDINGS).sort_values(by=['date'])[['ticker','isin','date','quantity','ccy']]

    daily_snapshots = pd.DataFrame(columns=holdings.columns)

    start_date = holdings.iloc[0]['date']
    print("Portoflio inception is {}".format(start_date))
    prev_date     =  pd.to_datetime(start_date)
    prev_snapshot = pd.DataFrame(columns=holdings.columns)

    snap_cont = 0
    for sd, snapshot in holdings.groupby(['date']):
        #if snap_cont > 2:
        #    break
        s_date = pd.to_datetime(sd[0])
        print("*** {} -> {}".format(prev_date,s_date))

        daily_snapshots = pd.concat([daily_snapshots, snapshot], ignore_index=True)

        if s_date == prev_date:   
            prev_date       = s_date
            prev_snapshot   = snapshot
            continue

        date_range = pd.date_range(start=prev_date+dt.timedelta(days=1), end=s_date, freq='D')
    
        for ticker in prev_snapshot['ticker'].unique():
           
            if (pd.isna(ticker)):
                continue
            
            #print("{}: {} -> {}".format(ticker,prev_date,s_date))
            #if snap_cont > 100:
            #    display(prev_snapshot)
            
            prev_ticker_snapshot = prev_snapshot[prev_snapshot['ticker']==ticker]
            
            if prev_ticker_snapshot['quantity'].values[0] != 0:       
                for d in date_range:
                    ffill_hold = {'ticker': ticker,
                                    'isin': prev_ticker_snapshot['isin'].values[0],
                                    'date': pd.to_datetime(d).date(),
                                    'quantity':prev_ticker_snapshot['quantity'].values[0],
                                    'ccy':prev_ticker_snapshot['ccy'].values[0]}
                    #do not add to the daily snapshot the position if it is already in the next snapshot
                    if ticker in snapshot.ticker.values and d == s_date:
                        print("removed {}".format(ticker))
                        continue
                    daily_snapshots.loc[daily_snapshots.index[-1] + 1] = ffill_hold

            #for i,j in daily_snapshots.groupby(['ticker']):
            #display(j)

        lps = daily_snapshots[daily_snapshots['date'] ==  pd.to_datetime(s_date).date()]
        close_positions = list(prev_ticker_snapshot[prev_ticker_snapshot['quantity'] == 0].ticker.values)
        
        lps = lps[lps.ticker.apply(lambda t : t not in close_positions)]
        lps = lps[lps.ticker.apply(lambda t : t not in snapshot.ticker.values)]
        
        #if snap_cont > 110 and snap_cont < 125:
        #    print("Last previous snapshot")
        #    display(lps)
        #    display(snapshot)
        

        prev_date = s_date    
        prev_snapshot = pd.concat([snapshot,lps], ignore_index=True)
        snap_cont +=1
                    
    #display(daily_snapshots)
    daily_snapshots.to_csv(notebook_constants.DATA_FOLDER + notebook_constants.PORTFOLIO_HOLDINGS_DAILY)
    


import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
from price_estimator import const_and_utils

def fetch_prices(portfolio: pd.DataFrame, mode:str = 'offline') -> None:

    portoflio_and_market_data = dict()
    portfolio['date'] = portfolio.date.apply(lambda d : pd.to_datetime(datetime.strptime(d, "%Y-%m-%d")).date() )
            

    for ticker in portfolio['ticker'].unique():
        try:
            print("Fetiching {}".format(ticker))
            if mode == "offline":
                ticker_prices = pd.read_csv(const_and_utils.FOLDER_MARKET_DATA+const_and_utils.PREFIX_PRICE_FETCHER+ticker+".csv")
            else:
                ticker_dates = portfolio[portfolio['ticker'] == ticker]['date'].values
                start_date = ticker_dates[0]
                end_date = ticker_dates[len(ticker_dates)-1]
                tickerData = yf.Ticker(ticker)
                if "quoteType" in tickerData.info:
                     print("Quote type : {}".format(tickerData.info["quoteType"]))
                if tickerData.info["quoteType"] != 'EQUITY':
                    continue
                ticker_prices =  tickerData.history(period='1d', start=start_date, end=end_date+timedelta(days=1))
                ticker_prices.reset_index(inplace=True)

            t_p = ticker_prices[["Date","Close","Dividends"]].rename(columns={'Date':'date'})
            t_p['date'] = t_p.date.apply(lambda d : pd.to_datetime(d).date())
            positions = portfolio[portfolio['ticker'] == ticker]
            portoflio_and_market_data[ticker] = positions.merge(t_p, on='date', how='left')
        except Exception as e:
            print(" Ticker [{}]  -->error in fetching price {}".format(ticker, e))
            continue

    for pos in portoflio_and_market_data.items():    
        pos[1].to_csv(notebook_constants.DATA_FOLDER+notebook_constants.DAILY_HOLDINGS_PREFIX+pos[0]+".csv")


import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np 


def compute_transaction_pnl(p: pd.DataFrame,  daycount:int, transactions : pd.DataFrame = None,) -> None:
     if transactions is not None :
            holdings_date       = datetime.strptime(p.at[daycount,'date'], "%Y-%m-%d").date()
            monthstr = str(holdings_date.month)
            if len(monthstr) == 1 :
                monthstr = '0'+ monthstr
            daystr   = str(holdings_date.day)
            if len(daystr) == 1 :
                daystr = '0'+ daystr
            holdings_datestr    = daystr+'/'+ monthstr+'/'+str(holdings_date.year)
            ticker_transactions = transactions[transactions[notebook_constants.TRANSACTION_FIELD_TR_DATE] == holdings_datestr ]
            transaction_value_open = 0
            for _,t in ticker_transactions.iterrows():
                transaction_value  = (t[notebook_constants.TRANSACTION_FIELD_QUANTITIY] * t[notebook_constants.TRANSACTION_FIELD_PRICE])
                eod_mktvalue =  p.at[daycount,'mkt_value_eod'] 
                if eod_mktvalue == 0:
                    eod_mktvalue =  p.at[daycount,'mkt_value'] 

                if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'A'):
                    transaction_value_open += transaction_value
                    p.at[daycount,'pnl'] += eod_mktvalue  -  transaction_value
                if (t[notebook_constants.TRANSACTION_FIELD_BUY_SELL] == 'V'):
                    #transaction_value_open += transaction_value
                    p.at[daycount,'pnl'] +=  transaction_value - eod_mktvalue

                print("{} -> transcation pnl detected: {} {}".format(ticker_transactions[notebook_constants.TRANSACTION_FIELD_TICKER].values[0],transaction_value,t[notebook_constants.TRANSACTION_FIELD_BUY_SELL]))
            
            if p.at[daycount,'mkt_value_bod'] == 0 and not ticker_transactions.empty:
                p.at[daycount,'mkt_value_bod'] = transaction_value_open


def compute_pnl(p: pd.DataFrame, transactions : pd.DataFrame = None) -> None:
    daycount = 0
    p['Close'].ffill(inplace=True)
    p['quantity'].ffill(inplace=True)
    p['Dividends'].replace(to_replace=np.NaN, value=0., inplace=True)

    
    

    for r in p.iterrows():
        if daycount == 0:
            p['mkt_value']     = float(p.at[daycount,'quantity'])  *  p.at[daycount,'Close'] 
            p['mkt_value_bod'] = 0
            p['mkt_value_eod'] = 0
            p['pnl']           = 0
            p['r']             = 1
            compute_transaction_pnl(p,daycount,transactions)
            daycount += 1
            continue
        p.at[daycount,'mkt_value']     = float(p.at[daycount,'quantity'])  *  p.at[daycount,'Close']    
        p.at[daycount,'mkt_value_bod'] = float(p.at[daycount-1,'quantity'])  *  p.at[daycount-1,'Close']
        p.at[daycount,'mkt_value_eod'] = p.loc[daycount-1,'quantity'] * (  p.at[daycount,'Close'] +  p.at[daycount,'Dividends'])
        p.at[daycount,'pnl'] = p.loc[daycount,'mkt_value_eod'] - p.loc[daycount,'mkt_value_bod']
        if (p.loc[daycount,'mkt_value_bod']) != 0:
            p.at[daycount,'r'] = (p.loc[daycount,'pnl'] / p.loc[daycount,'mkt_value_bod']) + 1
        else:
            p.at[daycount,'r'] = 1

        compute_transaction_pnl(p,daycount,transactions)
        daycount += 1

    return p



def plot_char(portfolio_performance_daily):    
    sc = 0
    portfolio_series = dict()
    metrics = ['pnl', 'mkt_value', 'mkt_value_bod', 'mkt_value_eod', 'r']
    portfolio_series['t'] = list()
    for m in metrics:
        portfolio_series[m] = list()

        for day,snapshot in portfolio_performance_daily.groupby(['date']):
            
            s = snapshot[snapshot[m].apply(lambda x : (x is not None))]
            s = s[pd.notna(s[m])]
            d = datetime.strptime(day[0], "%Y-%m-%d").date()
            if not s.empty:
                if m == 'r':
                    portfolio_series['t'].append(d)
                    denom = sum(s['mkt_value_bod'])
                    if (denom != 0):
                        portfolio_series[m].append(sum(s['pnl']) / sum(s['mkt_value_bod']) +1)
                    else:
                        portfolio_series[m].append(1)
                else:
                    portfolio_series[m].append(sum(s[m]))
            else:
                if m == 'r':
                    portfolio_series['t'].append(d)
                    portfolio_series[m].append(1)
                else:
                    portfolio_series[m].append(0)

    for m in metrics:

        transactions_df = pd.read_csv(notebook_constants.DATA_FOLDER+
                                    notebook_constants.TRANSACTION_FILE_TICKERS) [
                                        [notebook_constants.TRANSACTION_FIELD_TR_DATE,
                                        notebook_constants.TRANSACTION_FIELD_BUY_SELL,
                                        notebook_constants.TRANSACTION_FIELD_QUANTITIY,
                                        notebook_constants.TRANSACTION_FIELD_TICKER]
                                        ]
        
        df = pd.DataFrame({ 't': portfolio_series['t'], m: portfolio_series[m]})
        df.plot(figsize=(12,6),grid=True, x='t')        


        if m == 'r':
            acc_r = [1] 
            for i in range(1, len(portfolio_series[m])):
                pp = acc_r[-1] * portfolio_series[m][i]
                acc_r.append(pp)

            import matplotlib.pyplot as plt

            plt.figure(figsize=(20, 10))
            df = pd.DataFrame({ 't': portfolio_series['t'], m: acc_r})
            plt.plot(df['t'],
                    df[m],
                    label='acc. return', color='blue')




            plt.grid(True)
            plt.xticks(np.arange(min(df['t']), max(df['t']) , 30))
            plt.xticks(rotation=45)
            
            r_count = 0
            transactions_df[notebook_constants.TRANSACTION_FIELD_TR_DATE] = transactions_df[notebook_constants.TRANSACTION_FIELD_TR_DATE].apply(lambda d: datetime.strptime(d, "%d/%m/%Y").date())
            transactions_df = transactions_df.sort_values(by=[notebook_constants.TRANSACTION_FIELD_TR_DATE])
            for event_date in transactions_df[notebook_constants.TRANSACTION_FIELD_TR_DATE]:
                event_types = transactions_df[transactions_df[notebook_constants.TRANSACTION_FIELD_TR_DATE] == event_date][notebook_constants.TRANSACTION_FIELD_BUY_SELL].values
                event_source = transactions_df[transactions_df[notebook_constants.TRANSACTION_FIELD_TR_DATE] == event_date][notebook_constants.TRANSACTION_FIELD_TICKER].values
                event_color = 'green'
                spread = 0.1
                event_type_lbl =""
                eventset = set()
        
                for i  in range(len(event_types)):
                    spread += abs(spread) + 0.1
                    if (event_types[i],event_source[i]) in eventset:
                        continue
                    eventset.add((event_types[i],event_source[i]))
                    if event_types[i] == 'A':
                        event_type_lbl +='B(' + str(event_source[i])+ ')\n'
                    if event_types[i] == 'V':
                        event_type_lbl ='S(' + str(event_source[i]) + ')\n'
                        event_color = 'red'

                    

                    plt.text(event_date,
                            df[df['t']==event_date]['r'].values[0] + spread,
                            event_type_lbl,
                            color=event_color, 
                            fontsize=7,
                            horizontalalignment='center',
                            verticalalignment='top')
                    
                    plt.plot(event_date, df[df['t']==event_date]['r'].values[0], 'o', markersize=2, markerfacecolor=event_color, label=event_type_lbl)
                    #plt.annotate('A', xy=(pos, m[int(pos)] + 0.1), ha='center', va='center', fontsize=15)  # Adjust annotation position and size as needed
                    r_count += 1
                
        
            plt.title('Accumulated returns')
            plt.xlabel('t')
            plt.ylabel('r')
            #plt.legend()
            plt.show()



