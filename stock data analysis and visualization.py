import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


pd.set_option('display.max_columns', 20)
stocks_df = pd.read_csv('/home/kashindra/PycharmProjects/Financial Analysis/stock.csv')
# print(stocks_df.head)

stocks_df = stocks_df.sort_values(by = ['Date'])

print("Total number of stock : {}".format(len(stocks_df.columns[1:])))

for i in stocks_df.columns[1:]:
    print(i)

print(stocks_df.describe())

print(stocks_df.isnull().sum())

print(stocks_df.info())


def show_plot(df, fig_title):
    df.plot(x='Date', figsize=(15, 7), linewidth=3, title=fig_title)
    plt.grid()
    plt.show()

# show_plot(stocks_df, 'RAW STOCK PRICES(WITHOUT NORMALIZATION)')


def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x

# print(normalize(stocks_df))

# show_plot(normalize(stocks_df), 'NORMALIZED STOCK PRICES')


def interactive_plot(df, title):
    fig = px.line(title=title)

    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)

    fig.show()


# interactive_plot(stocks_df, 'Prices')

# interactive_plot(normalize(stocks_df), 'NORMALIZED PRICES')

# df = stocks_df['sp500']
# df_daily_return = df.copy()
#
# for i in range(1, len(df)):
#     df_daily_return[i] = ((df[i] - df[i-1])/df[i-1]) * 100
#
# df_daily_return[0] = 0
# print(df_daily_return)


def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        df_daily_return[i][0] = 0
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1]) * 100)
    return df_daily_return


stocks_daily_return = daily_return(stocks_df)
# print(stocks_daily_return)

# show_plot(stocks_daily_return, "STOCKS DAILY RETURNS")
# interactive_plot(stocks_daily_return, "STOCKS DAILY RETURNS")

cm = stocks_daily_return.drop(columns=['Date']).corr()
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True)
# plt.show()


print(stocks_daily_return.hist(figsize=(10, 10), bins=40));
# plt.show()

df_hist = stocks_daily_return.copy()
df_hist = df_hist.drop(columns = ['Date'])
data = []

for i in df_hist.columns:
    data.append(stocks_daily_return[i].values)

# fig = ff.create_distplot(data, df_hist.columns)
# fig.show()

