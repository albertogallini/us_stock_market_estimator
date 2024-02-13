#%%
import pandas as pd
import datetime as dt
from datetime import datetime


def unroll_daily_holdings():
    holdings =  pd.read_csv("./../../reports/holdings.csv").sort_values(by=['date'])[['ticker','isin','date','quantity','ccy']]

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
    daily_snapshots.to_csv("./../../reports/daily_holdings.csv")
    
unroll_daily_holdings()
                
#%%

import pandas as pd
import datetime as dt
from datetime import datetime

def fetch_prices(portfolio: pd.DataFrame) -> None:

    portoflio_and_market_data = dict()
    portfolio['date'] = portfolio.date.apply(lambda d : pd.to_datetime(datetime.strptime(d, "%Y-%m-%d")).date() )
            

    for ticker in portfolio['ticker'].unique():
        try:
            print("Fetiching {}".format(ticker))
            ticker_prices = pd.read_csv("./../../data/price_fetcher_"+ticker+".csv")
            t_p = ticker_prices[["Date","Close","Dividends"]].rename(columns={'Date':'date'})
            t_p['date'] = t_p.date.apply(lambda d : pd.to_datetime(d).date())
            positions = portfolio[portfolio['ticker'] == ticker]
            portoflio_and_market_data[ticker] = positions.merge(t_p, on='date', how='left')
        except Exception as e:
            print("price not found: {}".format(e))
            continue

    for pos in portoflio_and_market_data.items():    
        pos[1].to_csv("./../../reports/daily_holdings_and_prices"+pos[0]+".csv")

portfolio = pd.read_csv("./../../reports/daily_holdings.csv")
fetch_prices(portfolio)


#%%
import pandas as pd
import datetime as dt
from datetime import datetime
import numpy as np 

def compute_pnl(p: pd.DataFrame) -> None:
    daycount = 0
    p['Close'].ffill(inplace=True)
    p['quantity'].ffill(inplace=True)
    p['Dividends'].replace(to_replace=np.NaN, value=0., inplace=True)
    

    for r in p.iterrows():
        if daycount == 0:
            p['mkt_value']     = None
            p['mkt_value_bod'] = None
            p['mkt_value_eod'] = None
            p['pnl']           = None
            p['r']             = None
            daycount += 1
            continue
        p.at[daycount,'mkt_value']     = float(p.at[daycount,'quantity'])  *  p.at[daycount,'Close']    
        p.at[daycount,'mkt_value_bod'] = float(p.at[daycount-1,'quantity'])  *  p.at[daycount-1,'Close']
        p.at[daycount,'mkt_value_eod'] = p.loc[daycount-1,'quantity'] * (  p.at[daycount,'Close'] +  p.at[daycount,'Close'] *  p.at[daycount,'Dividends'])
        p.at[daycount,'pnl'] = p.loc[daycount,'mkt_value_eod'] - p.loc[daycount,'mkt_value_bod']
        p.at[daycount,'r'] = (p.loc[daycount,'pnl'] / p.loc[daycount,'mkt_value_bod']) + 1
        daycount += 1

    return p


import os
prefix = 'daily_holdings_and_prices'
portfolio_dir = './../../reports/'
files = [filename for filename in os.listdir(portfolio_dir) if filename.startswith(prefix)]
portfolio_performance_daily = None

print("Loading position files ...")
for f in files:
    #print("Loading {}.".format(f))    
    position_performance_daily  = compute_pnl(pd.read_csv(portfolio_dir+f))
    #display(position_performance_daily)
    portfolio_performance_daily = pd.concat([position_performance_daily,portfolio_performance_daily], ignore_index=True)        
        
#portfolio_performance_daily.to_csv("test.csv")
    
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
                portfolio_series[m].append(sum(s['pnl']) / sum(s['mkt_value_bod']) +1)
            else:
                portfolio_series[m].append(sum(s[m]))
        else:
            if m == 'r':
                portfolio_series['t'].append(d)
                portfolio_series[m].append(1)
            else:
                portfolio_series[m].append(0)

for m in metrics:

    transactions_df = pd.read_csv("./../../reports/transactions_si_tickers.csv") [['Operazione','Segno','Quantita']]
    
    df = pd.DataFrame({ 't': portfolio_series['t'], m: portfolio_series[m]})
    df.plot(figsize=(12,6),grid=True, x='t')        


    if m == 'r':

        acc_r = [1] 
        for i in range(1, len(portfolio_series[m])):
            pp = acc_r[-1] * portfolio_series[m][i]
            acc_r.append(pp)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        df = pd.DataFrame({ 't': portfolio_series['t'], m: acc_r})
        plt.plot(df['t'],
                 df[m],
                 label='acc. return', color='blue')
        plt.grid(True)
        plt.xticks(np.arange(min(df['t']), max(df['t']) , 90))
        plt.xticks(rotation=45)
        
      
        r_count = 0
        transactions_df['Operazione'] = transactions_df['Operazione'].apply(lambda d: datetime.strptime(d, "%d/%m/%Y").date())
        transactions_df = transactions_df.sort_values(by=['Operazione'])
        for event_date in transactions_df['Operazione']:
            event_type = transactions_df[transactions_df['Operazione'] == event_date]['Segno'].values[0]
            event_color = 'green'
            if event_type == 'A':
                event_type ='b'
            if event_type == 'V':
                event_type ='s'
                event_color = 'red'

            plt.text(event_date,
                    df[df['t']==event_date]['r'].values[0],
                    event_type,
                    color=event_color, 
                    fontsize=7,
                    horizontalalignment='center',
                    verticalalignment='bottom')
            r_count += 1
      

        # Customize the plot
        plt.title('Accumulated returns')
        plt.xlabel('t')
        plt.ylabel('r')
        plt.legend()

        # Show the plot
        plt.show()



        #df.plot(figsize=(12,6),grid=True , x='t')


           


    
    

    # %%


