# functions for the stylized fact part
import functions
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
import seaborn as sns
import matplotlib.ticker as ticker




def add_returns_column(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe with a new column 'returns' that contains the returns of the midprice
    """

    # adds a column with the midprice price of the option today minus the midprice closing price of the previous day:
    data['returns'] = data.close.pct_change()

    return data

def correlation_returns(data, N):
    """
    This function returns the correlation of the returns of the index with lagged by N days.

    Entry:  data: dataframe

    Exit:   float: correlation of the returns of the index with lagged by N days
    """

    data = add_returns_column(data)

    # Shift the returns by N days
    lagged_returns = data.returns.shift(N)

    # Drop the NaN values that were created by shifting
    returns = data.returns.dropna()
    lagged_returns = lagged_returns.dropna()

    
    # delete the first N element of returns
    returns = returns[N:]

    # Calculate the correlation
    correlation = np.corrcoef(returns, lagged_returns)[0, 1]

    return correlation

def correlation_absolute_returns(data, N):
    """
    This function returns the correlation of the absolute returns of the index with lagged by N days.

    Entry:  data: dataframe

    Exit:   float: correlation of the absolute returns of the index with lagged by N days
    """

    data = add_returns_column(data)

    # Shift the returns by N days
    lagged_returns = data.returns.shift(N)

    # Drop the NaN values that were created by shifting
    returns = data.returns.dropna()
    lagged_returns = lagged_returns.dropna()

    # delete the first N element of returns
    returns = returns[N:]

    # Calculate the correlation
    correlation = np.corrcoef(np.abs(returns), np.abs(lagged_returns))[0, 1]

    return correlation

def correlation_squared_returns(data, N):
    """
    This function returns the correlation of the squared returns of the index with lagged by N days.

    Entry:  data: dataframe

    Exit:   float: correlation of the squared returns of the index with lagged by N days
    """

    data = add_returns_column(data)

    # Shift the returns by N days
    lagged_returns = data.returns.shift(N)

    # Drop the NaN values that were created by shifting
    returns = data.returns.dropna()
    lagged_returns = lagged_returns.dropna()

    # delete the first N element of returns
    returns = returns[N:]

    # Calculate the correlation
    correlation = np.corrcoef(returns ** 2, lagged_returns ** 2)[0, 1]

    return correlation

def plot_autocorrelation(data):
    """
    This function plots the autocorrelation of the returns of the index.

    Entry:  data: dataframe
    """

    data = add_returns_column(data)

    days = []
    autocorrelation = []
    absolute_autocorrelation = []
    squared_autocorrelation = []


    for i in tqdm(range(1, 31)):
        days.append(i)
        autocorrelation.append(correlation_returns(data, i))
        absolute_autocorrelation.append(correlation_absolute_returns(data, i))
        squared_autocorrelation.append(correlation_squared_returns(data, i))


    x = plt.figure().gca()  # get current axes
    x.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    x.plot(days, autocorrelation, marker='o', label='Autocorrelation')  # Add points on each data point
    x.plot(days, absolute_autocorrelation, marker='x', label='Absolute Autocorrelation')  # Add points on each data point
    x.plot(days, squared_autocorrelation, marker='s', label='Squared Autocorrelation')  # Add points on each data point
    x.set_xlabel('Lag in days')
    x.legend()  # Add a legend
    x.set_ylabel('Autocorrelation')
    x.set_title('Autocorrelation of returns')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.grid(True)  # Add a grid
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_long_autocorrelation(data):
    """
    This function plots the autocorrelation of the returns of the index for a lot of lag numbers.

    Entry:  data: dataframe
    """

    data = add_returns_column(data)

    days = []
    autocorrelation = []
    absolute_autocorrelation = []
    squared_autocorrelation = []


    for i in tqdm(range(1, 250)):
        days.append(i)
        absolute_autocorrelation.append(correlation_absolute_returns(data, i))
        squared_autocorrelation.append(correlation_squared_returns(data, i))


    x = plt.figure().gca()  # get current axes
    x.xaxis.set_major_locator(MaxNLocator(nbins=10))  # set the number of x-axis labels to 10

    x.plot(days, absolute_autocorrelation, label='Absolute Autocorrelation')  # Add points on each data point
    x.plot(days, squared_autocorrelation, label='Squared Autocorrelation')  # Add points on each data point
    x.set_xlabel('Lag in days')
    x.legend()  # Add a legend
    x.set_ylabel('Autocorrelation')
    x.set_title('Long term Autocorrelation of returns')
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.grid(True)  # Add a grid
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_returns_distribution(data):
    """
    This function plots the distribution of the returns of the index. It also plots the Gaussian distribution 
    with the same mean and standard deviation as the returns.
    Entry:  data: dataframe
    """

    data = add_returns_column(data)

    returns = data.returns.dropna()

    # Calculate the mean and standard deviation of the returns
    mu, std = np.mean(returns), np.std(returns)

    # Generate the x-values for the Gaussian distribution
    x = np.linspace(min(returns), max(returns), 100)

    # Generate the y-values for the Gaussian distribution
    y = norm.pdf(x, mu, std)

    # Plot the KDE of the returns
    sns.kdeplot(returns, color='r',label='SPX returns')

    # Plot the Gaussian distribution
    plt.plot(x, y, color='b', label='Gaussian distribution')

    # Add labels and title
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Density of 1 day returns SP500 vs Gaussian distribution')
    plt.legend()


    # Display the plot
    plt.show()

# test the functions
file_path = 'data/SPX-index.csv'
data = functions.read_data(file_path)

# As one increases the period of time ∆t over which these returns are calculated, asset returns show lower tails
def return_distribution_comparison(data):
    """
    This function plots the distribution of the 1 day returns, weekly returns and monthly returns of the index.
    """

    # creates column with the returns of the index
    data = add_returns_column(data)

    # creates a column with the weekly returns of the index
    data['weekly_returns'] = data.close.pct_change(7)

    # creates a column with the monthly returns of the index
    data['monthly_returns'] = data.close.pct_change(30)

    # Plot the KDE of the returns
    sns.kdeplot(data.returns.dropna(), color='r', label='1 day returns')

    # Plot the KDE of the weekly returns
    sns.kdeplot(data.weekly_returns.dropna(), color='g', label='Weekly returns')

    # Plot the KDE of the monthly returns
    sns.kdeplot(data.monthly_returns.dropna(), color='b', label='Monthly returns')

    # Add labels and title
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Density of returns')
    plt.legend()

    # Display the plot
    plt.show()

# Returns and Volatility are Negatively Correlated
def add_volatility_column(data):
    """
    Entry:  data: dataframe

    Exit:   data: dataframe with a new column 'volatility' that contains the volatility of the index
    """

    data = add_returns_column(data)

    # adds a column with the volatility of the index
    data['volatility'] = data.returns.rolling(window=30).std()

    return data

def plot_volatility(data):
    """
    This function plots the volatility of the index, and the price of the index.

    Entry:  data: dataframe
    """

    data = add_volatility_column(data)
    
    fig, ax1 = plt.subplots()

    # Plot the volatility on the first y-axis
    ax1.plot(data.date, data.volatility, color='b', label='Volatility')
    ax1.set_ylabel('Volatility', color='b')
    ax1.tick_params('y', colors='b')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the returns on the second y-axis
    ax2.plot(data.date, data.close, color='r', label='SPX')
    ax2.set_ylabel('Returns', color='r')
    ax2.tick_params('y', colors='r')

    # Set the x-axis label and title
    ax1.set_xlabel('Date')
    ax1.set_title('Volatility and close prices of the SPX index')

    # Add a legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Set x-axis to only display 10 dates
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Display the plot
    plt.show()


