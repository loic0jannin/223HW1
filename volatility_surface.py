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
from scipy.optimize import brentq
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import plotly.graph_objects as go


def black_scholes_option_price(S, K, r, T, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'C':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price

# plot the volatility surface in 3D
def plot_volatility_surface(option_data):
    pivot_df = option_data.pivot_table(index='moneyness', columns='time_to_maturity', values='IV')

        # Create the surface plot
    fig = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index)])

        # Update layout for better visualization
    fig.update_layout(title='Implied Volatility Surface', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90),
                        scene = dict(
                            xaxis_title='Time to Expiration',
                            yaxis_title='moneyness',
                            zaxis_title='Implied Volatility'))

        # Show the plot
    fig.show()

def plot_volatility_scatter(option_data):
    # Pivot your DataFrame to form a grid
    pivot_df = option_data.pivot_table(index='moneyness', columns='time_to_maturity', values='IV')

    # Create the scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pivot_df.columns.repeat(len(pivot_df.index)),
        y=np.repeat(pivot_df.index, len(pivot_df.columns)),
        z=pivot_df.values.flatten(),
        mode='markers'
    )])

    # Update layout for better visualization
    fig.update_layout(title='Implied Volatility Scatter Plot', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90),
                    scene=dict(
                        xaxis_title='Time to Expiration',
                        yaxis_title='Moneyness',
                        zaxis_title='Implied Volatility'))

    # Show the plot
    fig.show()

# function to add the implied volatility to the option data
def add_IV(option_data, date):
    # data only on date
    option_data = option_data[option_data['date'] == date]

    # delete the data with best_bid = 0
    option_data = option_data[option_data['best_bid'] != 0]

    # Import the index data
    index_data = pd.read_csv('data/SPX-index.csv')

    # Convert 'date' columns to datetime if they're not already
    index_data['date'] = pd.to_datetime(index_data['date'])
    option_data['date'] = pd.to_datetime(option_data['date'])

    # Merge option_data with index_data on 'date' column to add 'close' column
    option_data = pd.merge(option_data, index_data[['date', 'close']], on='date', how='left')

    # Merge option_data with index_data on 'date' column to add 'open' column:
    option_data = pd.merge(option_data, index_data[['date', 'open']], on='date', how='left')

    # we multiply the close price by 1000
    option_data['close'] = option_data['close'] * 1000
    option_data['open'] = option_data['open'] * 1000

    # we add a ccolumn with the price of the option (average of bid and ask)
    option_data['price'] = (option_data['best_bid'] + option_data['best_offer']) / 2

    # Convert 'exdate' column to datetime
    option_data['exdate'] = pd.to_datetime(option_data['exdate'])

    # Calculate the implied volatility
    for i in tqdm(range(len(option_data))):
        am_settlement = option_data['am_settlement'].iloc[i]
        if am_settlement == 'True':
            S = option_data['open'].iloc[i]
        else:
            S = option_data['close'].iloc[i]
        K = option_data['strike_price'].iloc[i]
        r = 0.01
        T = (option_data['exdate'].iloc[i] - option_data['date'].iloc[i]).days / 365
        option_price = option_data['price'].iloc[i]
        option_type = option_data['cp_flag'].iloc[i].lower()
        
        q = 0
        if T == 0:
            option_data.at[i, 'IV'] = 0
        else:
            try:
                option_data.at[i, 'IV'] = implied_volatility(option_price, S, K, T, r, q, option_type)
            except:
                option_data.at[i, 'IV'] = 0

    return option_data

# add moneynesss and do some modifitactions
def add_moneyness_column(option_data):
    # we add a column time to maturity
    option_data['time_to_maturity'] = (option_data['exdate'] - option_data['date']).dt.days

    # we delete the data with time to maturity = 0  
    option_data = option_data[option_data['time_to_maturity'] != 0]

    # we multiply the IV by 100 to have the same unit as the option price
    option_data['IV'] = option_data['IV'] * 100

    # we add the column moneyness
    option_data['moneyness'] = option_data['strike_price'] / option_data['close']

    # percentage of data with IV = 0
    print('Percentage of data with IV = 0 :', len(option_data[option_data['IV'] == 0]) / len(option_data) * 100, '%')

    # we delete the data with IV = 0
    option_data = option_data[option_data['IV'] != 0]

    return option_data


