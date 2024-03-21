# rolls model to copute the spread

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import functions
import warnings

import math


def get_ds_opt(data, day, symbol, N):
    """
    Entry:    data: dataframe
              day: string
              symbol
              N: int, number of days to consider

    Exit:   d: array of the d_i until day, where d_i = P_i - P_{i-1}, len N
            d_1: array of the d_i until day-1 , where d_i = P_i - P_{i-1}, len N
            (P_i = midprice of the option at day i)
    """

    # defines the date N days before day
    N_days_ago = pd.to_datetime(day) - pd.DateOffset(N)  # gets the date N days before day

    # gets the data for only the 50 days before day:
    data['date'] = pd.to_datetime(data['date'])
    data = data[(data.date <= day) & (data.date >= N_days_ago)]

    # gets the data for the option with the given strike price, time to maturity and option type:
    data = data[data.symbol == symbol]

    # adds a column with the midprice price of the option:
    data['midprice'] = (data.best_offer + data.best_bid) / 2

    # adds a column with the midprice price of the option today minus the midprice closing price of the previous day:
    data['d'] = data.midprice.diff()

    # creates the d and d_1 arrays, d = the last N values of d, d_1 = the last N-1 values of d
    d = data.d.values
    # deletes the first value of d
    d = np.delete(d, 0)

    d_1 = d[:-1]
    d = d[1:]

    # returns the d_i for the day and the day-1:
    return d, d_1

def rolls_model_opt(data, day, symbol, N):

    # gets the random variables values for the day
    d, d_1 = get_ds(data, day, symbol, N)

    # computes the covariance between the two random variables
    cov = np.cov(d, d_1)[0, 1]

    # returns the rolls model for the day
    if cov > 0:
        return 0
    else:
        return 2*math.sqrt(-cov)
    
def add_ds(data):
    # add the d column to the dataframe which is the diference between close and close yesterday
    data['d'] = data.close - data.close.shift(1)

    # add the d_1 column to the dataframe which is d shifted
    data['d_1'] = data.d.shift(1)

    return data




def rolls_model(data,N):
    # add the d columnsa
    data = add_ds(data)

    # adds a column with the rolls model spread
    data['rolls_model'] = 0

    # for each day, computes the rolls model spread
    for day in data.date.unique():
        # retrives the N last values of d and d_1 before day 
        d = data.loc[data.date == day, 'd'].values[-N:]
        d_1 = data.loc[data.date == day, 'd_1'].values[-N:]

        # computes the covariance between the two random variables
        cov = np.cov(d, d_1)[0, 1]

        data.loc[data.date == day, 'rolls_model'] = 2*math.sqrt(-cov)

    return data

def percentage_roll(data):
    ''' This function returns the percentage of day when the rolls model is not defined'''
    # the column with the roll model computed is already available: data.loc[data.date == day, 'rolls_model'] = 2*math.sqrt(-cov)
    count = 0

    for day in data.date.unique():
        if math.isnan(data.loc[data.date == day, 'rolls_model'].values[0]):
            count += 1

    return count/len(data.date.unique())

def plot_rollsModel(data):
    # plots the percentage of days when the rolls model is not defined for diff values of N
    N = np.arange(1, 1000)
    percentage = []

    for n in tqdm(N):
        # ignore the warnings
        warnings.filterwarnings('ignore')
        percentage.append(percentage_roll(data,n))

    plt.plot(N, percentage)
    plt.xlabel('N')
    plt.ylabel('Percentage of days when the rolls model is not defined')
    plt.title('Percentage of days when the rolls model is not defined for diff values of N')

    plt.show()


# test
file_path = 'data/SPX-index.csv'
data = functions.read_data(file_path)

# test
plot_rollsModel(data)

