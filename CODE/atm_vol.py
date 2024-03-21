import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import functions
import pandas_datareader.data as web
from scipy.stats import norm
import datetime
import warnings
import matplotlib.animation as animation
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
from scipy import optimize
import plotly.graph_objects as go

def add_midprice_column(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe with a new column 'midprice' that contains the midprice
    """
    # adds the midprice
    data['midprice'] = (data.best_bid + data.best_offer) / 2

    return data

def check_risk_free_rate(data):
    """
    Entry:  data: dataframe

    Exit:   None
    """
    # deletes all the options with risk free rate = NaN
    data = data.dropna(subset=['risk_free_rate'])

    return data

def add_underlying_price_column(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe with a new column 'underlying_price' that contains the underlying price
    """

    # get the underlying price using the csv file we have
    index_data = pd.read_csv('data/SPX-index.csv')

    # change index data to keep just date and close
    index_data = index_data[['date', 'close']]
    
    # Convert the 'date' column in both dataframes to datetime
    data['date'] = pd.to_datetime(data['date'])
    index_data['date'] = pd.to_datetime(index_data['date'])

    # multiplies by 100 to get the price in dollars
    index_data['close'] = index_data['close'] * 1000

    # Merge the dataframes on date
    data = pd.merge(data, index_data, left_on='date', right_on='date', how='left')

    # Rename the 'close' column to 'underlying_price'
    data.rename(columns={'close': 'underlying_price'}, inplace=True)

    return data

def no_volumes(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe without the rows with volume = 0
    """

    # deletes all the options with volume = 0
    data = data[data['volume'] != 0]

    return data

def filter_spread(data):
    '''Filter out options with a spread greater than four standard deviations higher than the mean'''

    # adds the spread column
    data['Spread'] = - (data.best_bid - data.best_offer)

    # Calculate the mean and standard deviation of the spread
    mean_spread = data['Spread'].mean()
    std_spread = data['Spread'].std()

    # Filter out options with spreads greater than four standard deviations higher than the mean
    data = data[data['Spread'] <= mean_spread + 4 * std_spread]

    return data

def get_atm_data(option_data):
    # Calculate the absolute difference between 'strike_price' and 'underlying_price'
    diff = (option_data['strike_price'] - option_data['underlying_price']).abs()

    # Find the index of the minimum difference
    atm_index = diff.idxmin()

    # Get the ATM strike price
    atm_strike = option_data.loc[atm_index, 'strike_price']
    # Filter the option_data dataframe to only include rows with the ATM strike price
    atm_data = option_data[option_data['strike_price'] == atm_strike]
    
    return atm_data

def data_at_date(option_data, date):
    # retrieves the data for the date
    data = option_data[option_data['date'] == date]

    return data

def plot_atm_brennervol(option_data,date):
    # get the brenner IV for the atm options af a function of time to expiration
    
    # retrieves the data for the date
    atm_data = data_at_date(option_data, date)

    # adds the column time to exp
    atm_data['timeToExp'] = (pd.to_datetime(atm_data.exdate) - pd.to_datetime(atm_data.date)).dt.days

    # takes off the data with time to expiration = 0
    atm_data = atm_data[atm_data['timeToExp'] != 0]

    # adds the underlying price column
    atm_data = add_underlying_price_column(atm_data)

    # gets just the atm options
    atm_data = get_atm_data(atm_data)
    
    # adds the IV
    atm_data = add_brenner_IV(atm_data)

    print(atm_data)

    plt.plot(atm_data['timeToExp'], atm_data['brenner_IV'])

    plt.xlabel('Time to Expiration')
    plt.ylabel('Brenner-Subrahmanyam Implied Volatility')
    plt.title('Brenner-Subrahmanyam Implied Volatility for ATM Options')
    plt.show()

def add_brenner_IV(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe with a new column 'brenner_IV' that contains the Brenner-Subrahmanyam implied volatility in percentage
    """

    # adds the brenner implied volatility
    data['brenner_IV'] = np.nan

    # add the midprice
    data = add_midprice_column(data)

    # for each row, calculate the brenner implied volatility
    for i in tqdm(range(len(data))):

        # get the option data
        time_to_expiration = (data['timeToExp'].iloc[i] / 252).round(3)
        underlying_price = data['underlying_price'].iloc[i]
        midprice = data['midprice'].iloc[i]

        # calculate the brenner implied volatility
        if time_to_expiration == 0:
            iv = 0
        else:
            iv = midprice / (0.4 * underlying_price/1000 * np.sqrt(time_to_expiration))
        
        # use the actual index of the DataFrame
        data.loc[data.index[i], 'brenner_IV'] = iv*100

    return data

def plot_avg_atm_vol(option_data):
    # get the brenner IV for the atm options as a function of time to expiration across all dates

    # adds the column time to exp
    option_data['timeToExp'] = (pd.to_datetime(option_data.exdate) - pd.to_datetime(option_data.date)).dt.days

    # adds the underlying price column
    option_data = add_underlying_price_column(option_data)

    # gets just the atm options
    atm_data = get_atm_data(option_data)

    # keeps just the options with traded volumes
    atm_data = no_volumes(atm_data)
    
    # adds the IV
    atm_data = add_brenner_IV(atm_data)

    # remove rows where timeToExp is 0
    atm_data = atm_data[atm_data.timeToExp != 0]

    # calculate average IV for each time to expiration
    avg_atm_data = atm_data.groupby('timeToExp')['brenner_IV'].mean().reset_index()

    plt.plot(avg_atm_data['timeToExp'], avg_atm_data['brenner_IV'])

    plt.xlabel('Time to Expiration')
    plt.ylabel('Average Brenner-Subrahmanyam Implied Volatility')
    plt.title('Average Brenner-Subrahmanyam Implied Volatility for ATM Options')
    plt.show()








