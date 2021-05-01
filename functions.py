import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def show_plot(df, fig_title):
    df.plot(x='Date', figsize=(15, 7), linewidth=3, title=fig_title)
    plt.grid()
    plt.show()


def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]
    return x


def interactive_plot(df, title):
    fig = px.line(title=title)

    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)

    fig.show()


def daily_return(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        df_daily_return[i][0] = 0
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j] - df[i][j-1]) * 100)
    return df_daily_return

def individual_stock(price_df, vol_df, name):
    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name],
                         'Volume': vol_df[name]})
