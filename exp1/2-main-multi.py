import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import random
import multiprocessing
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import distinctipy

from DataLoader.DataLoader import DataLoader
from DataLoader.DataBasedAgent import DataBasedAgent
from DataLoader.DataRLAgent import DataRLAgent
import DeepRLAgent.VanillaInput.Train as Train
from PatternDetectionInCandleStick.Evaluation import Evaluation
import shutil

from utils import add_train_portfo, add_test_portfo, plot_return, calc_return, plot_action_point, calc_bh, setup_logger

Train = reload(Train)
DeepRL = Train.Train

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
CURRENT_PATH = os.getcwd()




def train(
    DATASET_NAME, 
    split_point='2018-01-01', 
    begin_date='2010-01-01', 
    end_date='2020-08-24', 
    initial_investment=1000,
    transaction_cost=0.0001,
    load_from_file=True,
    reward_type="profit",
    seed=42, 
    state_mode=1,
    n_episodes=5,
    lamb=0.0001,
    GAMMA=0.7, 
    n_step=5, 
    BATCH_SIZE=10, 
    ReplayMemorySize=20,
    TARGET_UPDATE=5,
    window_size=None, 
    train_portfolios={},
    test_portfolios={},
    arms={},
    show_all = False,
    ratio_threshold=0.9,
):
    data_loader = DataLoader(DATASET_NAME, split_point=split_point, begin_date=begin_date, end_date=end_date, load_from_file=load_from_file)
    
    dataTrain_agent = DataRLAgent(data_loader.data_train, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
    dataTest_agent = DataRLAgent(data_loader.data_test, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    agent = DeepRL(data_loader, dataTrain_agent, dataTest_agent, 
                DATASET_NAME,  state_mode, window_size, transaction_cost,
                BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, ReplayMemorySize=ReplayMemorySize,
                TARGET_UPDATE=TARGET_UPDATE, n_step=n_step, arms=arms)

    agent.train(arms, n_episodes, ratio_threshold, seed, begin_date, end_date)

    agent_eval = agent.test(initial_investment=initial_investment, test_type='train', model_path="")
    train_portfolio = agent_eval.get_daily_portfolio_value()
    
    agent_test = agent.test(initial_investment=initial_investment, test_type='test', model_path="")
    test_portfolio = agent_test.get_daily_portfolio_value()

    arm = arms[0]
    final_model_name = f"{arm['name']}_{arm['lamb']}_{seed}"

    add_train_portfo(train_portfolios, final_model_name, train_portfolio)
    add_test_portfo(test_portfolios, final_model_name, test_portfolio)

    return data_loader


def add_arm(arms, name, lamb):
    arm = { "index": len(arms), "name": name, "lamb": lamb, "sharpe_list": [], "cumreturn_list": []}
    arms.append(arm)


def check_log(path, epochs):
    # TODO 这里要根据情况改一下
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
            _epochs = re.findall("epoch: (\d+),", content)
            if len(_epochs) > 0 and int(_epochs[-1]) == epochs - 1:
                return True
    
    return False


def run(_file, begin_date, end_date, split_point, initial_investment, seed, epochs):
    train_portfolios = {}
    test_portfolios = {}

    kwargs = {
        "begin_date": begin_date, 
        "end_date": end_date, 
        "split_point": split_point, 
        "load_from_file": True, 
        "transaction_cost": 0.0000,
        "initial_investment": initial_investment,
        "state_mode": 1,
        "seed": 42, 
        "GAMMA": 0.7, 
        "n_step": 5, 
        "BATCH_SIZE": 10, 
        "ReplayMemorySize": 20,
        "TARGET_UPDATE": 5,
        "window_size": None, 
        "train_portfolios": train_portfolios,
        "test_portfolios": test_portfolios,
        "lamb": 0.0,
    }

    arms = []
    # NOTE profit就是FP5
    add_arm(arms, "profit", 0)
    add_arm(arms, "old_profit", 0)
    add_arm(arms, "old_sharpe", 0)
    add_arm(arms, "future_profit_1", 0)
    add_arm(arms, "future_profit_10", 0)
    add_arm(arms, "future_profit_15", 0)
    add_arm(arms, "future_profit_20", 0)

    # add_arm(arms, "sharpe", 0.01)
    # add_arm(arms, "volatility", 10)
    # add_arm(arms, "regularized", 0.01)
    # add_arm(arms, "regularized", 0.05)
    # add_arm(arms, "regularized", 0.1)
    # add_arm(arms, "regularized", 0.2)

    flag = False
    for arm in arms:
        print(_file, seed, begin_date, end_date, f"{arm['name']}-{arm['lamb']}")

        path = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train_log/{arm['name']}-{arm['lamb']}.log"
        final_path = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train_log/{seed}.log"

        is_finished = check_log(path, epochs)
        if is_finished:
            print(f"{arm['name']}已经跑完了，跳过...")
            continue
        
        kwargs.update({
            "DATASET_NAME": _file,
            "reward_type": "",
            "seed": seed,
            "n_episodes": epochs,
            "arms": [arm],
            "show_all": True,
            "ratio_threshold": 3,
            "train_portfolios": train_portfolios,
            "test_portfolios": test_portfolios,
        })
    
        data_loader = train(**kwargs)
        flag = True
    
    if not flag:
        return
    
    # indexes = calc_return(data_loader, train_portfolios, test_portfolios)
    
    # final_model_name = indexes.T["sharpe_train"].sort_values(ascending=False).index[0]
    # src = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train/model_{final_model_name}.pkl"
    # dst = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train/model_{seed}.pkl"
    # shutil.copyfile(src, dst)
    
    # path = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train_log/"
    # logger, handler = setup_logger(f'{_file}-{seed}-final', f'{path}/{seed}.log')
    # logger.info(f"symbol: {_file}, seed: {seed}, final reward type: {final_model_name}")
    # logger.info(f"symbol: {_file}, seed: {seed}, top 3: {indexes.T['sharpe_train'].sort_values(ascending=False).values[:3]}")
    # logger.info(f"symbol: {_file}, seed: {seed}, top 3 name: {list(indexes.T['sharpe_train'].sort_values(ascending=False).index)[:3]}")
    # logger.removeHandler(handler)
    return seed


def worker_main(queue):
    print("启动进程")
    while True:
        try:
            item = queue.get(block=True) #block=True means make a blocking call to wait for items in queue
            if item is None:
                print("运行结束...")
                break
            seed = run(*item)
            print(f"seed: {seed}已经运行完毕...")
            time.sleep(2) 
        except KeyboardInterrupt:
            print("keyboard interrupt")
            import sys
            sys.exit(0)
    
    import sys
    sys.exit(1)


if __name__ == "__main__":

    initial_investment = 1000
    epochs = 20

    kwargs = {
        "begin_date": '2016-01-01', 
        "end_date": '2019-01-01', 
        "split_point": '2018-01-01', 
        "load_from_file": True, 
        "transaction_cost": 0.0000,
        "initial_investment": initial_investment,
        "state_mode": 1,
        "GAMMA": 0.7, 
        "n_step": 5, 
        "BATCH_SIZE": 10, 
        "ReplayMemorySize": 20,
        "TARGET_UPDATE": 5,
        "window_size": None, 
        "lamb": 0.0,
        "n_episodes": epochs,
    }

    import multiprocessing

    NUM_PROCESSES = 2
    NUM_QUEUE_ITEMS = 40

    queue = multiprocessing.Queue(maxsize=NUM_QUEUE_ITEMS)
    pool = multiprocessing.Pool(NUM_PROCESSES, worker_main, (queue,))
    
    _begin_date = '20{}-01-01'
    _end_date = '20{}-01-01'
    _split_point = '20{}-01-01' 

    files = os.listdir("./Data/")
    ls = []
    # TODO 1 - 10的future_20
    # TODO 10 - 30的old_profit
    for _file in files[:10]:
        
        for year in range(1):
            begin_date = _begin_date.format(16+year)
            end_date = _end_date.format(19+year)
            split_point = _split_point.format(18+year)

            for seed in range(100):

                # path = f"./Results/{_file}/{begin_date}~{end_date}/{seed}/train_log/{seed}.log"
                # if os.path.exists(path):
                #     print(f"symbol: {_file}, seed: {seed} 已经跑过了，跳过...")
                #     continue

                item = (_file, begin_date, end_date, split_point, initial_investment, seed, epochs)

                while True:
                    if not queue.full():
                        print(f"加入symbol: {_file}, year: {16+year}, seed: {seed}到队列...")
                        queue.put(item)
                        break
                    else:
                        print("queue已满，休息10秒...")
                        time.sleep(10)
    
    queue.put(None)

    queue.close()
    queue.join_thread()

    pool.close()
    pool.join()
    
    import sys 
    sys.exit(0)
