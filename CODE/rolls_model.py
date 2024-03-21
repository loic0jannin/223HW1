# rolls model to copute the spread

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import functions
import warnings

import math


def rolls_model(data, N):
    # add a column with the rolls model to the data

    # add a d column to the data
    data['d'] = data['close'] - data['close'].shift(1)

    # add a column d_1 to the data
    data['d_1'] = data['d'].shift(1)

    # add a column with the covariance of d and d_1 to the data using a rolling window of size N
    data['cov'] = data['d'].rolling(window=N).cov(data['d_1'])

    # add a column with the rolls spread
    data['rolls_model'] = 2*np.sqrt(-1 * (data['cov']))

    return data

def percentage_rolls(data):
    # returns the perdcentage of days when the rolls model is not defined
    return data['rolls_model'].isna().mean().round(4)*100

def plot_rollsModel(data):
    # plots the percentage of days when the rolls model is not defined for diff values of N
    N = np.arange(10, 1000)
    percentage = []

    # computes the rolls model once for the maximum value of N
    data = rolls_model(data, N.max())

    for n in tqdm(N):
        # adds the rolls model to the data
        data = rolls_model(data, n)

        # computes the percentage of days when the rolls model is not defined for the current value of N
        percentage.append(percentage_rolls(data))

    plt.plot(N, percentage)
    plt.xlabel('N')
    plt.ylabel('Percentage of days when the rolls model is not defined')
    plt.title('Percentage of days when the rolls model is not defined for diff values of N')

    plt.show()

