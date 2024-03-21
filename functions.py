# Functions for the main program

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import seaborn as sns




def read_data(file_path):
    # puts the csv file into a dataframe and returns it

    data = pd.read_csv(file_path)
    return data

def data2023(data):
    # returns the first three months of the data
    data = data[data.date >= "2023-01-01"]
    return data

def plot_volume(data):
    # plots the volume as a function of time

    # regroups all the volume for each day
    data = data.groupby('date').volume.sum().reset_index()

    
    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(data.date, data.volume)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title('Volume as a function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_monthly_volume(data):
    # plots the volume as a function of time

    # convert 'date' to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])

    # set 'date' as the index
    data.set_index('date', inplace=True)

    # resample to get monthly volumes
    monthly_data = data['volume'].resample('M').sum().reset_index()

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(monthly_data.date, monthly_data.volume, 'k')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title('Monthly Volume as a function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_volume_strike(option_data):
    # plots the volume as a function of the strike price

    # regroups all the volume for each strike price
    option_data = option_data.groupby('strike_price').volume.sum().reset_index()

    # print the strike with the biggest volume:
    print(option_data[option_data.volume == option_data.volume.max()])

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(option_data.strike_price, option_data.volume)
    ax.set_xlabel('Strike price')
    ax.set_ylabel('Volume')
    ax.set_title('Volume as a function of the strike price')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_closing_price(data):
    # plots the closing price as a function of time

    # convert 'date' to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])

    # set 'date' as the index
    data.set_index('date', inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(data['close'], color='black')
    plt.title('Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.tight_layout()
    plt.show()

def plot_volume_strike_putCall(option_data):
    # plots the volume as a function of the strike price for calls and puts in different colors

    # regroups all the volume for each strike price
    call_data = option_data[option_data.cp_flag == 'C']
    put_data = option_data[option_data.cp_flag == 'P']
    call_data = call_data.groupby('strike_price').volume.sum().reset_index()
    put_data = put_data.groupby('strike_price').volume.sum().reset_index()
    
    # get the strike with the biggest volume for calls and puts:
    max_vol_strike_call = call_data[call_data.volume == call_data.volume.max()].strike_price.values[0]
    max_vol_strike_put = put_data[put_data.volume == put_data.volume.max()].strike_price.values[0]

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(call_data.strike_price, call_data.volume, label='Call', color = "blue")
    ax.plot(put_data.strike_price, put_data.volume, label='Put', color = "red", alpha=0.5)
    ax.set_xlabel('Strike price')
    ax.set_ylabel('Volume')
    ax.set_title('Volume as a function of the strike price')
    ax.legend()

    # add text to the plot
    plt.text(0.01, 0.95, f'Max volume strike for call: {max_vol_strike_call}', transform=ax.transAxes)
    plt.text(0.01, 0.90, f'Max volume strike for put: {max_vol_strike_put}', transform=ax.transAxes)

    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def get_1DTE(data, day):
    """
    This function returns the options that expires one day after "day" for a specific day to expiration.
    """
    day = pd.to_datetime(day)  # convert 'day' to datetime
    next_day = day + pd.DateOffset(days=1)  # add one day
    return data[data['exdate'] == next_day.strftime('%Y-%m-%d')]  # filter the DataFrame
    
def percentage_volume_1DTE(data, day):
    """
    This function returns the volume percentage of 1DTE compared to other maturity, on a specific day.
    """
    # slects the data of a given day:
    data = data[data['date'] == day]

    # selects the 1DTE data:
    data_1DTE = get_1DTE(data, day)

    # returns the volume of 1DTE:
    volume_1DTE = data_1DTE.volume.sum()

    # returns the volume of all maturity:
    volume_all = data.volume.sum()

    # returns the percentage of 1DTE volume compared to other maturity:
    return volume_1DTE/volume_all

def plot_1DTE_volume(data):
    """
    This function plots the volume of 1DTE as a function of time.
    """
    volumes = []
    percentage = []
    dates = []

    for day in tqdm(data.date.unique()):

        dates.append(day)
        u = percentage_volume_1DTE(data, day)
        percentage.append(u)
        volumes.append(u* data[data['date'] == day].volume.sum())

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(dates, volumes)
    ax.set_xlabel('Date')
    ax.set_ylabel('volume of 1DTE')
    ax.set_title('volume of 1DTE as a function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_1DTE_volume_percentage(data):
    # plot an histogram of the percentage of 1DTE volume compared to other maturity as function of time
    percentage = []
    dates = []

    for day in tqdm(data.date.unique()):
        dates.append(day)
        percentage.append(percentage_volume_1DTE(data, day))
    
    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(dates, percentage)
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of 1DTE volume compared to other maturity as function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def add_spread_column(data):
    """
    This function adds a column to the dataframe with the spread of the option.
    """
    data['spread'] = data.best_offer - data.best_bid
    return data

def average_spread_call_put(data):
    """
    This function returns the average spread for calls and puts. 
    Returns: (average spread for calls, average spread for puts)
    """
    data = add_spread_column(data)
    call_data = data[data.cp_flag == 'C']
    put_data = data[data.cp_flag == 'P']
    return call_data.spread.mean(), put_data.spread.mean()

def plot_spread_strike(data):
    """
    This function plots the spread as a function of the strike price.
    """
    data = add_spread_column(data)
    data = data.groupby('strike_price').spread.mean().reset_index()

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(data.strike_price, data.spread)
    ax.set_xlabel('Strike price')
    ax.set_ylabel('Spread')
    ax.set_title('Spread as a function of the strike price')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_spread_timeToExp(data):
    """
    This function plots the spread as a function of time to expiration.
    """
    data = add_spread_column(data)
    
    # creates a new column with the time to expiration
    data['timeToExp'] = (pd.to_datetime(data.exdate) - pd.to_datetime(data.date)).dt.days

    data = data.groupby('timeToExp').spread.mean().reset_index()

    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    ax.plot(data.timeToExp, data.spread)
    ax.set_xlabel('Time to expiration')
    ax.set_ylabel('Spread')
    ax.set_title('Spread as a function of time to expiration')
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()



