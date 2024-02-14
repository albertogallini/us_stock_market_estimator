# %%
import notebook_constants
import pandas as pd
import yfinance as yf
import time
import pandas as pd
from openfigipy import OpenFigiClient
from functions import generate_holdings,generate_transaction_file



generate_transaction_file()


# %%
generate_holdings()
