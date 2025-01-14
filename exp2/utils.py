import logging
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from PatternDetectionInCandleStick.Evaluation import Evaluation
import plotly.graph_objs as go

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_PATH = os.getcwd()

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers = []
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file, encoding="utf8")        
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger, handler


def add_train_portfo(train_portfolios, model_name, portfo):
    counter = 0
    key = f'{model_name}'
    
    # while key in train_portfolios.keys():
    #     counter += 1
    #     key = f'{model_name}{counter}'
    
    train_portfolios[key] = portfo
    return train_portfolios


def add_test_portfo(test_portfolios, model_name, portfo):
    counter = 0
    key = f'{model_name}'

    # while key in test_portfolios.keys():
    #     counter += 1
    #     key = f'{model_name}{counter}'

    test_portfolios[key] = portfo
    return test_portfolios


def calc_bh(train_portfolios, test_portfolios, data_loader, initial_investment):
    train_bh_portfo = (data_loader.data_train_with_date["close"] / data_loader.data_train_with_date["close"].iloc[0]) * initial_investment
    test_bh_portfo = (data_loader.data_test_with_date["close"] / data_loader.data_test_with_date["close"].iloc[0]) * initial_investment
    train_bh_portfo = train_bh_portfo.tolist()
    test_bh_portfo = test_bh_portfo.tolist()
    add_train_portfo(train_portfolios, 'B&H', train_bh_portfo)
    add_test_portfo(test_portfolios, 'B&H', test_bh_portfo)


def plot_action_point(mode, dataTrain_agent, dataTest_agent, data_loader, model_kind, DATASET_NAME, begin=0, end=120):

    if mode == "train":
        data_test = dataTrain_agent
        df1 = data_loader.data_train_with_date[begin:end]
    else:
        data_test = dataTest_agent
        df1 = data_loader.data_test_with_date[begin:end]

    actionlist = list(data_test.data[data_test.action_name][begin:end])
    df1[data_test.action_name] = actionlist

    buy = df1.copy()
    sell = df1.copy()
    none = df1.copy()

    # NOTE 应该画在close处比较合适吧
    buy['action'] = [c if a == 'buy' else None for a, o, c in zip(df1[data_test.action_name], df1.open, df1.close)]
    sell['action'] = [c if a == 'sell' else None for a, o, c in zip(df1[data_test.action_name], df1.open, df1.close)]
    none['action'] = [c if a == 'None' else None for a, o, c in zip(df1[data_test.action_name], df1.open, df1.close)]

    data=[
        go.Candlestick(x=df1.index, open=df1['open'], high=df1['high'], low=df1['low'], close=df1['close'], increasing_line_color= 'lightgreen', decreasing_line_color= '#ff6961'),
        go.Scatter(x=df1.index, y=buy.action, mode = 'markers', marker=dict(color='red', colorscale='Viridis'), name="buy"), 
        go.Scatter(x=df1.index, y=none.action, mode = 'markers', marker=dict(color='blue', colorscale='Viridis'), name="none"), 
        go.Scatter(x=df1.index, y=sell.action, mode = 'markers', marker=dict(color='green', colorscale='Viridis'), name="sell")
    ]

    layout = go.Layout(
        autosize = False,
        width = 1000,
        height = 400,
    )

    figSignal = go.Figure(data=data, layout=layout)
    figSignal.show()

    # with open(fig_file, "wb") as f:
    #     f.write(scope.transform(figSignal, format="svg"))


def calc_maxdrawdown(mode, data_loader, portfolio):
    if mode == "train":
        dates = data_loader.data_train_with_date.index 
    else:
        dates = data_loader.data_test_with_date.index 

    mdd, mdd_start_index, mdd_end_index = 0, 0, 0
    for i in range(1, len(portfolio)):
        max_value = np.max(portfolio[:i])
        max_value_index = np.argmax(portfolio[:i])
        min_value = np.min(portfolio[max_value_index:i])
        min_value_index = np.argmin(portfolio[max_value_index:i])
        min_value_index = max_value_index + min_value_index
        ratio = (max_value - min_value) / max_value
        if ratio > mdd:
            mdd = ratio
            mdd_start_index, mdd_end_index = max_value_index, min_value_index

    mdd_start_date, mdd_end_date = dates[mdd_start_index-1], dates[mdd_end_index-1] 
    return mdd, mdd_start_index, mdd_end_index, mdd_start_date, mdd_end_date


def plot_return(mode, DATASET_NAME, data_loader, train_portfolios, test_portfolios, colors, indexes):
    if mode == "train":
        portfolios = train_portfolios
        dates = data_loader.data_train_with_date
    else:
        portfolios = test_portfolios
        dates = data_loader.data_test_with_date
        
    sns.set(rc={'figure.figsize': (15, 10)})

    cols = indexes.index
    ls = []
    for col in cols:
        if mode in col:
            ls.append(col)
    indexes = indexes.loc[ls]
    
    items = list(portfolios.keys())
    first = True
    for k, color in zip(items, colors):
        profit_percentage = [(portfolios[k][i]) / portfolios[k][0] for i in range(len(portfolios[k]))]
        difference = len(portfolios[k]) - len(dates)
        df = pd.DataFrame({
            'date': dates.index, 
            'portfolio': profit_percentage[difference:]
        })
        mdd1, mdd_start_index1, mdd_end_index1, _, _ = calc_maxdrawdown(mode, data_loader, df["portfolio"])
        d = indexes[k]
        cumreturn, sharpe = d.loc[f"cumreturn_{mode}"], d[f"sharpe_{mode}"]
        cumreturn, sharpe = round(cumreturn, 3), round(sharpe, 3)
        if not first:
            df.plot(ax=ax, x='date', y='portfolio', label=f"{k}(cumreturn: {cumreturn}, sharpe: {sharpe}, mdd: {round(mdd1, 3)})", color=color, alpha=0.5)
        else:
            ax = df.plot(x='date', y='portfolio', label=f"{k}(cumreturn: {cumreturn}, sharpe: {sharpe}, mdd: {round(mdd1, 3)})", color=color, alpha=0.5)
            first = False
        
        ax.plot([mdd_start_index1, mdd_end_index1], 
                [df["portfolio"].iloc[mdd_start_index1], df["portfolio"].iloc[mdd_end_index1]], "v", color=color, markersize=5)

    plt.xlabel("Time")
    plt.ylabel("Net value")
    plt.title(f'Net value at each point of time for {mode}ing data of {DATASET_NAME}')
    plt.legend(fontsize=8)


def calc_sharpe(portfo):
    portfo = np.array(portfo)
    mean = np.mean(portfo)
    std = np.std(portfo)
    if std != 0:
        sharpe = np.sqrt(252) * mean / std 
    else:
        sharpe = 0
    return sharpe

def calc_tracking_error(portfo, bh_portfo):
    portfo = np.array(portfo)
    bh_portfo = np.array(bh_portfo)
    excess  = portfo - bh_portfo
    std = np.std(excess)
    return std

def calc_IR(portfo, bh_portfo):
    portfo = np.array(portfo)
    bh_portfo = np.array(bh_portfo)
    excess  = portfo - bh_portfo
    mean = np.mean(excess)
    std = np.std(excess)
    if std != 0:
        ir = np.sqrt(252) * mean / std 
    else:
        ir = 0
    return ir 

def calc_volatility(portfo):
    portfo = np.array(portfo)
    std = np.sqrt(252) * np.std(portfo)
    return std 

def calc_downsiderisk(portfo):
    portfo = np.array(portfo)
    mean = np.mean(portfo)
    x = portfo[portfo < mean]
    std = np.sqrt(252) * np.std(x)
    return std 

def calc_sortino(portfo):
    portfo = np.array(portfo)
    mean = np.mean(portfo)
    std = np.std(portfo[portfo < mean])
    if std != 0:
        sortino = np.sqrt(252) * mean / std 
    else:
        sortino = 0
    return sortino


def calc_beta(portfo, bh_portfo):
    # NOTE beta是代表和市场的变动，所以不用加np.sqrt(252)
    portfo = portfo.values.reshape(-1,)
    bh_portfo = bh_portfo.reshape(-1,)

    v = np.array([portfo, bh_portfo])
    cov = np.cov(v)
    beta = cov[0, 1] / cov[1, 1]
    return beta


def calc_alpha(portfo, bh_portfo, beta):
    portfo_return = portfo.values.reshape(-1,)
    bh_portfo_return = bh_portfo.reshape(-1,)
    
    alpha = np.mean(portfo_return - beta * bh_portfo_return) * np.sqrt(252)
    return alpha


def calc_all(mode, data_loader, dt, portfo, portfo_percentage, bh_percentage):
    dt[f"cumreturn_{mode}"] = portfo[-1] / portfo[0] - 1
    # NOTE 乘以np.sqrt(252)是年化
    dt[f"mdd_{mode}"], _, _, mdd_start_date, mdd_end_date = calc_maxdrawdown(mode, data_loader, portfo)
    dt[f"mdd_date_{mode}"] = f"{mdd_start_date}-{mdd_end_date}"
    dt[f"sharpe_{mode}"] = calc_sharpe(portfo_percentage)
    dt[f"risk_{mode}"] = calc_volatility(portfo_percentage)
    dt[f"downrisk_{mode}"] = calc_downsiderisk(portfo_percentage)
    dt[f"sortino_{mode}"] = calc_sortino(portfo_percentage)
    dt[f"ir_{mode}"] = calc_IR(portfo_percentage, bh_percentage)
    dt[f"tracking_error_{mode}"] = calc_tracking_error(portfo_percentage, bh_percentage)
    dt[f"beta_{mode}"] = calc_beta(portfo_percentage, bh_percentage)
    dt[f"alpha_{mode}"] = calc_alpha(portfo_percentage, bh_percentage, dt[f"beta_{mode}"])


def calc_return(data_loader, train_portfolios, test_portfolios):
    res = []
    
    for key in test_portfolios.keys():
        dt = {}
        bh_percentage = data_loader.data_train_with_date["close"].pct_change(1).fillna(0).values
        portfo_percentage = pd.DataFrame(train_portfolios[key]).pct_change(1)
        if portfo_percentage.shape[0] == bh_percentage.shape[0]:
            portfo_percentage = portfo_percentage.fillna(0)
        elif portfo_percentage.shape[0] - bh_percentage.shape[0] == 1:
            portfo_percentage = portfo_percentage.iloc[1:]
        
        calc_all("train", data_loader, dt, train_portfolios[key], portfo_percentage, bh_percentage)

        bh_percentage = data_loader.data_test_with_date["close"].pct_change(1).fillna(0).values
        portfo_percentage = pd.DataFrame(test_portfolios[key]).pct_change(1)
        if portfo_percentage.shape[0] == bh_percentage.shape[0]:
            portfo_percentage = portfo_percentage.fillna(0)
        elif portfo_percentage.shape[0] - bh_percentage.shape[0] == 1:
            portfo_percentage = portfo_percentage.iloc[1:]
        
        calc_all("test", data_loader, dt, test_portfolios[key], portfo_percentage, bh_percentage)

        dt = dict(sorted(dt.items(), key=lambda x: x[0], reverse=True))
        res.append(dt)

    res = pd.DataFrame(res, index=list(train_portfolios.keys()))
    res = res.fillna(0)
    res = res.T
    return res


# def calculate_hold_days(self,):
#     """
#     TODO 要针对输入调整一下。
#     NOTE 这个其实还包含了胜率。
#     """
#     import pandas as pd
#     df = pd.read_csv("1-action-list.csv")
#     df_return = pd.read_csv("1-return.csv")
#     df = pd.concat([df, df_return], axis=1)
#     df.columns = ["action", "return"]

#     holds = []
#     returns = []
#     flag = "none"

#     for i in range(df.shape[0]):
#         # 0\1\2: buy\none\sell
#         if df.iloc[i]['action'] == 0:
#             if flag == "none":
#                 hold = 0
#                 cumreturn = 1
#                 flag = "hold"
#             if flag == 'hold':
#                 hold += 1
#                 cumreturn = cumreturn * (1 + df.iloc[i]["return"] / 100)
#         elif df.iloc[i]['action'] == 1:
#             hold += 1
#         if df.iloc[i]['action'] == 2:
#             if flag == "hold":
#                 hold += 1
#                 holds.append(hold)
#                 hold = 0
#                 flag = "none"
#                 cumreturn = cumreturn * (1 + df.iloc[i]["return"] / 100)
#                 returns.append(cumreturn)
#                 cumreturn = 1
    
#     returns = np.array(returns)
#     win_ratio = returns[returns > 1].shape[0] / returns.shape[0]
#     actions = pd.DataFrame({"holdday": holds, "returns": returns})
#     actions.groupby("holdday").agg({"returns": ["median", "count"]})
