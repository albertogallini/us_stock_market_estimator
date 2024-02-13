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

def compute_pnl(portfolio: pd.DataFrame) -> None:

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
        display(pos[1].sort_values(by=['date']))



portfolio = pd.read_csv("./../../reports/daily_holdings.csv")
compute_pnl(portfolio)
        
        

    # %%
