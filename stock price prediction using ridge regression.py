import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras
from functions import normalize, interactive_plot


pd.set_option('display.max_columns', 20)
stock_price_df = pd.read_csv('/home/kashindra/PycharmProjects/Financial Analysis/stock.csv')
stock_vol_df = pd.read_csv('/home/kashindra/PycharmProjects/Financial Analysis/stock_volume.csv')

stock_price_df = stock_price_df.sort_values(by=['Date'])
stock_vol_df = stock_vol_df.sort_values(by=['Date'])

# print(stock_price_df.isnull().sum())
# print(stock_vol_df.isnull().sum())
#
# print(stock_price_df.info())
# print(stock_vol_df.info())
#
# print(stock_price_df.describe())
# print(stock_vol_df.describe())


def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name],
                         'Volume': vol_df[name]})


def trading_window(data):
    n = 1
    data['Target'] = data[['Close']].shift(-n)
    return data


price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')
print(price_volume_df)

price_volume_target_df = trading_window(price_volume_df)
print(price_volume_target_df)

price_volume_target_df = price_volume_target_df[:-1]
print(price_volume_target_df)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns=['Date']))
print(price_volume_target_scaled_df)

X = price_volume_target_scaled_df[:, :2]
Y = price_volume_target_scaled_df[:, 2:]

print(X.shape, Y.shape)

split = int(0.65 * len(X))
x_train = X[:split]
y_train = Y[:split]
x_test = X[split:]
y_test = Y[split:]


def show_plot(data, title):
    plt.figure(figsize=(13, 5))
    plt.plot(data, linewidth=3)
    plt.title(title)
    plt.grid()
    # plt.show()


show_plot(x_train, "Training Data")
show_plot(x_test, "Testing Data")

from sklearn.linear_model import Ridge
regression_model = Ridge(alpha=0.3)
regression_model.fit(x_train, y_train)
lr_accuracy = regression_model.score(x_test, y_test)
print('Ridge regression score is:', lr_accuracy)

predicted_prices = regression_model.predict(X)
# print(predicted_prices)

predicted = []
for i in predicted_prices:
    predicted.append(i[0])

close = []
for i in price_volume_target_scaled_df:
    close.append(i[0])

df_predicted = price_volume_target_df[['Date']]
df_predicted['close'] = close
df_predicted['prediction'] = predicted

interactive_plot(df_predicted, "Original vs Predictions")




