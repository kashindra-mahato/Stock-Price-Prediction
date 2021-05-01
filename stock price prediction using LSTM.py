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
from functions import normalize, interactive_plot, individual_stock

pd.set_option('display.max_columns', 20)
stock_price_df = pd.read_csv('/home/kashindra/PycharmProjects/Financial Analysis/stock.csv')
stock_vol_df = pd.read_csv('/home/kashindra/PycharmProjects/Financial Analysis/stock_volume.csv')

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')

training_data = price_volume_df.iloc[:, 1:3].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_data)

x = []
y = []
for i in range(1, len(price_volume_df)):
    x.append(training_set_scaled[i-1:i, 0])
    y.append((training_set_scaled[i, 0]))

x = np.asarray(x)
y = np.asarray(y)

split = int(0.7 * len(x))

x_train = x[:split]
y_train = y[:split]
x_test = x[split:]
y_test = y[split:]


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]), 1)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]), 1)


inputs = keras.layers.input( shape = (x_train.shape[1], x_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
outputs = keras.layers.dense(1, activation= 'linear')(x)

model = keras.models(inputs=inputs, outputs=outputs)

model.compile(optimizer= 'adam', loss= 'mse')
model.summary()