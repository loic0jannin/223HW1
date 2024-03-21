import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import atm_vol as vs
import warnings
from tqdm import tqdm

def get_the_call_and_put(option_data, date):
    # get the price of the put and the call ATM with time to expiration the next possible trading day 

    # get the data for the day
    data = option_data[option_data['date'] == date]

    # adds the underlying price
    data = vs.add_underlying_price_column(data)

    # get the atm data
    atm_data = vs.get_atm_data(data)

    # addds the time to expiration
    atm_data['timeToExp'] = (pd.to_datetime(data.exdate) - pd.to_datetime(data.date)).dt.days

    #  delete the data with time to matyrity 0
    atm_data = atm_data[atm_data['timeToExp'] != 0]

    # get the options with the min time to maturity
    atm_data = atm_data[atm_data['timeToExp'] == atm_data['timeToExp'].min()]

    # get the strike price
    K = atm_data['strike_price'].mean()

    # add midprice
    atm_data['midprice'] = (atm_data['best_bid'] + atm_data['best_offer']) / 2

    # select only the option where symbol start with SPXW
    atm_data = atm_data[atm_data['symbol'].str.startswith('SPXW')]

    # get the price of the call
    call = atm_data[atm_data['cp_flag'] == 'C'].midprice.item()

    # get the price of the put
    put = atm_data[atm_data['cp_flag'] == 'P'].midprice.item()

    return call, put, K

def cost_at_day(option_data, date):
    # get the price of the put and the call ATM with time to expiration the next possible trading day
    call, put, K = get_the_call_and_put(option_data, date)

    return call + put, K
 
def payoff_at_day(data, date, K):
    # get the data for the day
    data = data[data['date'] == date]

    # adds the underlying price
    data = vs.add_underlying_price_column(data)

    # get the underlying price its the same for all the options
    underlying_price = data['underlying_price'].iloc[0]
    
    # compute the payoff for each option
    call_payoff = np.maximum(underlying_price - K , 0)/1000
    put_payoff = np.maximum(K - underlying_price, 0)/1000

    return call_payoff+put_payoff

def pnl1(start_date, end_date, option_data):
    
    # define the day list
    days = getdays(start_date, end_date, option_data)

    # Initialize an empty DataFrame
    pnl_df = pd.DataFrame(columns=['date', 'payoff', 'cost'])

    first_cost, K = cost_at_day(option_data, days[0])
    pnl_df.loc[len(pnl_df)] = [days[0], 0, -first_cost]

    for date in tqdm(days[1:-1]):
        # ignore warnings
        warnings.filterwarnings('ignore')

        payoff = payoff_at_day(option_data, date, K)
        cost, K = cost_at_day(option_data, date)
        pnl_df.loc[len(pnl_df)] = [date, payoff, -cost]

    last_payoff = payoff_at_day(option_data, days[-1], K)
    pnl_df.loc[len(pnl_df)] = [days[-1], last_payoff, 0]

    return pnl_df

def pnl2(start_date, end_date, option_data):
    
    # define the day list
    days = getdays(start_date, end_date, option_data)

    # Initialize an empty DataFrame
    pnl_df = pd.DataFrame(columns=['date', 'payoff', 'cost'])

    first_cost, K = cost_at_day(option_data, days[0])
    pnl_df.loc[len(pnl_df)] = [days[0], 0, first_cost]

    for date in tqdm(days[1:-1]):
        # ignore warnings
        warnings.filterwarnings('ignore')
        payoff = payoff_at_day(option_data, date, K)
        cost, K = cost_at_day(option_data, date)
        pnl_df.loc[len(pnl_df)] = [date, -payoff, +cost]

    last_payoff = payoff_at_day(option_data, days[-1], K)
    pnl_df.loc[len(pnl_df)] = [days[-1], -last_payoff, 0]

    return pnl_df

def getdays(start_date, end_date, data):
    # returns the list of days that are actually present in the data between these two dates
    days = pd.date_range(start_date, end_date)
    days = days[days.isin(data['date'])]

    # Convert to list of strings
    days = days.strftime('%Y-%m-%d').tolist()

    return days

def plot_pnl(pnl_df, price_data):
    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    # Plot PnL
    ax.plot(pnl_df['date'], pnl_df['payoff'].cumsum() + pnl_df['cost'].cumsum(), label='PnL')
    ax.set_xlabel('Date')
    ax.set_ylabel('PnL')
    ax.set_title('PnL and Underlying Price as function of time')

    # # Plot underlying price
    # ax2 = ax.twinx()  # Create a second y-axis
    # ax2.plot(price_data['date'], price_data['close'], color='r', label='Underlying Price')
    # ax2.set_ylabel('Underlying Price')

    # Add a legend
    lines, labels = ax.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def add_volatility_column(index_data):
    # add the volatility column

    # compute the log return
    index_data['log_return'] = np.log(index_data['close'] / index_data['close'].shift(1))

    # compute the rolling standard deviation
    index_data['volatility'] = index_data['log_return'].rolling(window=252).std() * np.sqrt(252)

    return index_data

def plot_volatility(index_data):
    # plot the volatility
    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    plt.plot(index_data['date'], index_data['volatility'])
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title('Volatility as a function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def get_mean_vol(start_date, end_date, index_data):
    # Convert 'date' column to datetime
    index_data['date'] = pd.to_datetime(index_data['date'])
    
    return index_data[(index_data['date'] >= start_date) & (index_data['date'] <= end_date)]['volatility'].mean()

def get_previous_date(date, df):
    # Convert 'date' column to datetime if it's not already
    df['date'] = pd.to_datetime(df['date'])

    # Calculate 'yesterday'
    yesterday = pd.to_datetime(date) - pd.Timedelta(days=1)

    # If 'yesterday' is not in df's 'date' column, get the max date that is less than 'date'
    if yesterday not in df['date'].values:
        yesterday = df[df['date'] < date]['date'].max()

    return yesterday

def forecast1_vol(date, index_data):
    # the vol at date is the vol at the previous day

    # get the date of the previous diff date in the dataset
    yesterday = get_previous_date(date, index_data)

    vol = index_data[index_data['date'] == yesterday]['volatility'].item()

    return vol

def pnl3( start_date, end_date, option_data, index_data, N):

    # define the day list
    trading_days = getdays(start_date, end_date, option_data)

    # Initialize an empty DataFrame
    pnl_df = pd.DataFrame(columns=['date', 'payoff', 'cost'])

    # create a dummy varianle to store the status of the last trade:
    # 1 if we bought the straddle
    # -1 if we sold the straddle
    # 0 if we did not trade
    last_trade = 0

    # intialise the strike
    K = 0

    for date in tqdm(trading_days):
        # ignore warnings
        warnings.filterwarnings('ignore')
        # we compute the average vol for the last N days
        N_days_ago = pd.to_datetime(date) - pd.Timedelta(days=N)
        avg_vol = get_mean_vol(N_days_ago, date, index_data)

        # we cash in the last trade
        pnl_df.loc[len(pnl_df)] = [date, last_trade * payoff_at_day(option_data,date,K), 0]
        
        # we compute the current vol
        current_vol = forecast1_vol(date, index_data)

        # we trade
        if current_vol > avg_vol:
            # we short the straddle
            cost, K = cost_at_day(option_data, date)
            pnl_df.loc[len(pnl_df)] = [date, 0, cost]
            # update the last_trade
            last_trade = -1

        else:
            # we buy the straddle
            cost, K = cost_at_day(option_data, date)
            pnl_df.loc[len(pnl_df)] = [date, 0, - cost]
            # update the last_trade
            last_trade = 1

        # put the warnings back
        warnings.filterwarnings('default')
            
    return pnl_df

def add_avg_vol(index_data,N):
    # Convert 'date' column to datetime if it's not already
    index_data['date'] = pd.to_datetime(index_data['date'])
    
    # Calculate the rolling average of 'volatility' over the last N days
    index_data['avg_vol'] = index_data['volatility'].rolling(window=N).mean()

    return index_data

def plot_averageVSvolatility(index_data):
    # Calculate the difference between average and real volatility
    index_data['vol_diff'] = (index_data['avg_vol'] - index_data['volatility'])

    # plot the volatility
    ax = plt.figure().gca()  # get current axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    plt.plot(index_data['date'], index_data['vol_diff'], label='Difference')
    plt.axhline(0, color='r', linestyle='--', label='y=0')  # Add line at y=0

    plt.xlabel('Date')
    plt.ylabel('Difference in Volatility')
    plt.title('Average - Real Volatility as a function of time')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.legend()
    plt.show()


# get the pnl
# pnl_df = pnl3(start_date, end_date, option_data, index_data, N)

# plot the pnl
# plot_pnl(pnl_df, index_data)
