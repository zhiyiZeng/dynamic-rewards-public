import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from DeepRLAgent.VanillaInput.DeepQNetwork import DQN
from DeepRLAgent.ReplayMemory import ReplayMemory, Transition

# from DeepQNetwork import DQN
# from ReplayMemory import ReplayMemory, Transition


from itertools import count
from tqdm import tqdm
import math
import os
import numpy as np 
import pandas as pd
from pathlib import Path
from utils import setup_logger

from PatternDetectionInCandleStick.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTrain:
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 model_kind,
                 state_mode=1,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is the base class for training across multiple models in the DeepRLAgent directory.
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        self.data_train = data_train
        self.data_test = data_test
        self.DATASET_NAME = dataset_name
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.ReplayMemorySize = ReplayMemorySize
        self.transaction_cost = transaction_cost
        self.model_kind = model_kind
        self.state_mode = state_mode
        self.window_size = window_size
        
        self.split_point = data_loader.split_point
        self.begin_date = data_loader.begin_date
        self.end_date = data_loader.end_date

        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_step = n_step

        # self.memory = ReplayMemory(ReplayMemorySize)
        # self.memory2 = ReplayMemory(ReplayMemorySize)

        self.train_test_split = True if data_test is not None else False

        # self.EPS_START = 0.9
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500

        self.steps_done = 0

        # self.PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
        #                          f'Results/{self.DATASET_NAME}/Train')
        
        # self.PATH_log = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
        #                          f'Results/{self.DATASET_NAME}/Train_log')

        # if not os.path.exists(self.PATH):
        #     os.makedirs(self.PATH)
        #     os.makedirs(self.PATH_log)

        # self.model_dir = os.path.join(self.PATH, f'model.pkl')

    def select_action(self, state,):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            return torch.tensor([[random.randrange(3)]], dtype=torch.long)

    def optimize_model(self,):
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        next_state_values = torch.zeros(self.BATCH_SIZE,)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * (self.GAMMA ** self.n_step)) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def calc_sharpe(self, portfo):
        portfo = np.array(portfo)
        mean = np.mean(portfo)
        std = np.std(portfo)
        if std != 0:
            sharpe = np.sqrt(252) * mean / std 
        else:
            sharpe = mean
        return sharpe

    def train(self, arms, num_episodes=10, ratio_threshold=3, seed=0, begin_date="", end_date=""):
        
        self.seed = seed
        
        self.path = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                 f'Results/{self.DATASET_NAME}/{begin_date}~{end_date}')
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path)
            except:
                pass
        
        self.path = f"{self.path}/{self.seed}"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.model_path = f"{self.path}/train"
        self.log_path = f"{self.path}/train_log"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        arm = arms[0]
        _file = f"{self.log_path}/{arm['name']}-{arm['lamb']}.log"
        if os.path.exists(_file):
            os.remove(_file)

        logger, handler = setup_logger(f'{self.DATASET_NAME}-{seed}', _file)
        logger.info(f"start a new seed: symbol: {self.DATASET_NAME}, seed: {self.seed}, reward type: {arm['name']}")

        self.model_dir = os.path.join(self.model_path, f'model.pkl')

        arm["model_dir"] = os.path.join(self.model_path, f'model_{arm["name"]}_{arm["lamb"]}_{seed}.pkl')
        self.model_dir = arm["model_dir"]

        # NOTE epoch相当于是
        for i_episode in range(num_episodes):

            self.memory = arm["memory"]
            self.policy_net = arm["policy_net"]
            self.target_net = arm["target_net"]
            self._model_dir = arm["model_dir"]
            self.reward_type = arm["name"]
            self.optimizer = arm["optimizer"]
            self.lamb = arm["lamb"] if arm.get("lamb") else 0

            # Initialize the environment and state
            self.data_train.reset()
            state = torch.tensor([self.data_train.get_current_state()], dtype=torch.float,)
            loss = 0
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                done, reward, next_state = self.data_train.step(action.item(), self.lamb, self.reward_type)

                reward = torch.tensor([reward], dtype=torch.float,)

                if next_state is not None:
                    next_state = torch.tensor([next_state], dtype=torch.float,)

                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                if not done:
                    state = torch.tensor([self.data_train.get_current_state()], dtype=torch.float,)

                # Perform one step of the optimization (on the target network)
                _loss = self.optimize_model()
                loss += _loss

                if done:
                    break
            
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            logger.info(f"symbol: {self.DATASET_NAME}, seed: {self.seed}, epoch: {i_episode}, reward type: {self.reward_type}-{self.lamb}, loss: {round(loss, 4)}")


        arm = arms[0]
        self.reward_type = arm["name"]
        self.policy_net = arm["policy_net"]
        torch.save(self.policy_net.state_dict(), self.model_dir)

        logger.info(f"symbol: {self.DATASET_NAME}, seed: {self.seed}, best arm: {arm['name']}-{arm['lamb']}")
        logger.info(f"symbol: {self.DATASET_NAME}, seed: {self.seed}, detail result:")
        
        test_agent = self.test(initial_investment=1000, test_type='train', epoch=i_episode, arm=arm)
        arithmetic_daily_return = test_agent.arithmetic_return()
        sharpe = self.calc_sharpe(arithmetic_daily_return / 100)
        cumreturn = ((1 + arithmetic_daily_return / 100).cumprod() - 1).iloc[-1]
        arm["sharpe_list"].append(sharpe)
        arm["cumreturn_list"].append(cumreturn)

        logger.info(f"name: {arm['name']}-{arm['lamb']}")
        _ = np.round(arm['sharpe_list'], 4)
        logger.info(f"sharpe_list: [{', '.join([str(__) for __ in _])}]")
        _ = np.round(arm['cumreturn_list'], 4)
        logger.info(f"cumreturn_list: [{', '.join([str(__) for __ in _])}]")
        logger.info("-" * 20)
        logger.info(f"symbol: {self.DATASET_NAME}, seed: {self.seed}, detail result end")
        logger.removeHandler(handler)

    def save_model(self, model):
        torch.save(model, self.model_dir)

    def test(self, initial_investment=1000, test_type='test', epoch=0, arm={}, model_path="",):
        """
        :@param file_name: name of the .pkl file to load the model
        :@param test_type: test results on train data or test data
        :@return returns an Evaluation object to have access to different evaluation metrics.
        """
        data = self.data_train if test_type == 'train' else self.data_test
        
        # NOTE arm == {} 说明是外部最优调用，最优的已经存到了model_dir里头了。加这个是因为只有从保存路径判断哪个是最优的。
        # NOTE arm != {} 说明是内部调用
        if model_path == "":
            if arm == {}:
                self.test_net.load_state_dict(torch.load(self.model_dir))
            else:
                self.test_net.load_state_dict(torch.load(f"{arm['model_dir']}"))
        else:
            self.test_net.load_state_dict(torch.load(model_path))

        action_list = []
        data.__iter__()
        
        # NOTE 直接把所有action拿出来了
        for batch in data:
            try:
                action_batch = self.test_net(batch).max(1)[1]
                action_list += list(action_batch.cpu().numpy())
            except ValueError:
                action_list += [1]

        data.make_investment(action_list)
        ev_agent = Evaluation(data.data, data.action_name, initial_investment, self.transaction_cost)
        return ev_agent

    def test_MA(self, initial_investment=1000, test_type='test', epoch=0, arm={}, model_path="", symbol="",):
        # NOTE 做MA金叉和死叉
        import talib

        # _data = self.data_train if test_type == 'train' else self.data_test
        data = pd.read_csv(f"./Data/{symbol}/data_processed.csv")
        # data = _data.data
        data["MA5"] = talib.MA(data["close_norm"], 5)
        data["MA10"] = talib.MA(data["close_norm"], 10)
        data["MA5>MA10"] = (data["MA5"] > data["MA10"]).replace({True: 1, False: 0})
        data["MA5<MA10"] = (data["MA5"] < data["MA10"]).replace({True: 1, False: 0})
        data["MA5<MA10_shift1"] = data["MA5<MA10"].shift(1)
        data["is_gold"] = (data["MA5>MA10"] + data["MA5<MA10_shift1"]).apply(lambda x: 1 if x == 2 else 0)
        del data["MA5<MA10_shift1"]
        data["MA5>MA10_shift1"] = data["MA5>MA10"].shift(1)
        data["is_dead"] = (data["MA5<MA10"] + data["MA5>MA10_shift1"]).apply(lambda x: 1 if x == 2 else 0)
        del data["MA5>MA10_shift1"]

        def make_action(x):
            if x[0] == 1:
                return 0
            elif x[1] == 1:
                return 2
            else:
                return 1

        data["action"] = data[["is_gold", "is_dead"]].apply(make_action, axis=1)
        data = data.query(f"Date >= '{self.split_point}' & Date <= '{self.end_date}'")

        cols = ["MA5", "MA10", "MA5>MA10", "MA5<MA10", "is_gold", "is_dead"]
        for col in cols:
            del data[col]
        
        action_list = data["action"].tolist()
        data = self.data_train if test_type == 'train' else self.data_test
        data.make_investment(action_list)
        ev_agent = Evaluation(data.data, data.action_name, initial_investment, self.transaction_cost)
        return ev_agent

    def test_MACD(self, initial_investment=1000, test_type='test', epoch=0, arm={}, model_path="", symbol="",):
        # NOTE 做MACD金叉和死叉
        import talib

        data = pd.read_csv(f"./Data/{symbol}/data_processed.csv")
        # _data = self.data_train if test_type == 'train' else self.data_test
        # data = _data.data
        data["DIF"], data["DEA"], _ = talib.MACD(data["close_norm"])
        data["DIF>DEA"] = (data["DIF"] > data["DEA"]).replace({True: 1, False: 0})
        data["DIF<DEA"] = (data["DIF"] < data["DEA"]).replace({True: 1, False: 0})
        data["DIF<DEA_shift1"] = data["DIF<DEA"].shift(1)
        data["is_gold"] = (data["DIF>DEA"] + data["DIF<DEA_shift1"]).apply(lambda x: 1 if x == 2 else 0)
        del data["DIF<DEA_shift1"]
        data["DIF>DEA_shift1"] = data["DIF>DEA"].shift(1)
        data["is_dead"] = (data["DIF<DEA"] + data["DIF>DEA_shift1"]).apply(lambda x: 1 if x == 2 else 0)
        del data["DIF>DEA_shift1"]

        def make_action(x):
            if x[0] == 1:
                return 0
            elif x[1] == 1:
                return 2
            else:
                return 1

        data["action"] = data[["is_gold", "is_dead"]].apply(make_action, axis=1)
        data = data.query(f"Date >= '{self.split_point}' & Date <= '{self.end_date}'")

        cols = ["DIF", "DEA", "DIF>DEA", "DIF<DEA", "is_gold", "is_dead"]
        for col in cols:
            del data[col]
        
        action_list = data["action"].tolist()
        data = self.data_train if test_type == 'train' else self.data_test
        data.make_investment(action_list)
        ev_agent = Evaluation(data.data, data.action_name, initial_investment, self.transaction_cost)
        return ev_agent

    def test_CCI(self, initial_investment=1000, test_type='test', epoch=0, arm={}, model_path="", symbol=""):
        import talib

        data = pd.read_csv(f"./Data/{symbol}/data_processed.csv")
        data = data[["Date", "high_norm", "low_norm", "close_norm"]]
        data["CCI"] = talib.CCI(data["high_norm"], data["low_norm"], data["close_norm"])

        data["is_buy"] = data["CCI"] >= 100
        data["is_sell"] = data["CCI"] <= -100

        def convert_action(x):
            if x[0] == 1:
                return 2
            elif x[1] == 1:
                return 0
            else:
                return 1

        data["action"] = data[["is_sell", "is_buy"]].apply(convert_action, axis=1)
        data = data.query(f"Date >= '{self.split_point}' & Date <= '{self.end_date}'")

        cols = ["CCI", "is_sell", "is_buy"]
        for col in cols:
            del data[col]
        action_list = data["action"].tolist()

        data = self.data_train if test_type == 'train' else self.data_test
        data.make_investment(action_list)
        ev_agent = Evaluation(data.data, data.action_name, initial_investment, self.transaction_cost)
        return ev_agent

    def test_MV(self, initial_investment=1000, test_type='test', epoch=0, arm={}, model_path="", symbol="",):
        from scipy.optimize import minimize

        data = pd.read_csv(f"./Data/{symbol}/data_processed.csv")
        data = data[["Date", "close_norm"]]
        data["close_pct"] = data["close_norm"].pct_change(1).replace({np.inf: 0})
        del data["close_norm"]

        treasure_df = pd.read_csv("USA-treasure-bill.csv")
        treasure_df = treasure_df[["日期", "涨跌幅"]]
        treasure_df.columns = ["Date", "treasure_pct"]
        treasure_df = treasure_df.sort_values("Date")
        treasure_df["Date"] = treasure_df["Date"].apply(lambda x: x.replace("年", "-").replace("月", "-").replace("日", ""))
        def convert_date(x):
            year, month, day = x.split("-")
            month = "0" * (2 - len(month)) + month
            day = "0" * (2 - len(day)) + day
            return f"{year}-{month}-{day}"
        treasure_df["Date"] = treasure_df["Date"].apply(convert_date)
        treasure_df = treasure_df.reset_index()
        del treasure_df["index"]
        treasure_df["treasure_pct"] = treasure_df["treasure_pct"].str.replace("%", "").astype(float) / 100

        data = pd.merge(data, treasure_df, on="Date", how="left")
        data = data.dropna()
        dates = data.query(f"Date >= '{self.split_point}' & Date <= '{self.end_date}'")["Date"].tolist()
        indexes = data.query(f"Date >= '{self.split_point}' & Date <= '{self.end_date}'").index

        window = 10
        data_tmp = data[["close_pct", "treasure_pct"]].rolling(window)
        mean_df = data_tmp.mean().dropna()
        mean_df = mean_df.loc[indexes]

        cov_df = data_tmp.cov().dropna()
        cov_df = cov_df.loc[indexes]

        def f(w, i):
            cov = np.array(cov_df.loc[i, :])
            mean = np.array(mean_df.loc[i]).reshape(2, 1)
            return 1 / 2 * np.dot(np.dot(w, cov), w) - np.dot(w, mean) 

        cons = (
            {'type': 'ineq', 'fun': lambda w:  w[0] + w[1] - 1},
            {'type': 'ineq', 'fun': lambda w: 1 - w[0] - w[1]},
        )

        actions = []
        for i in tqdm(indexes):
            if i < window:
                action = 1
                actions.append(action)
                continue

            w0 = np.array([0.5, 0.5]).reshape(1, 2)
            res = minimize(f, w0, args=(i,), constraints=cons, bounds=[(0, 1), (0, 1)])
            if res["success"]:
                if 0.45 <= res["x"][0] <= 0.55:
                    action = 1
                else:
                    if res["x"][0] > res["x"][1]:
                        action = 0
                    else:
                        action = 2
                
                actions.append(action)

        # pd.DataFrame({"action": actions})
        # hold = False
        # treasures = []
        # for i, action in enumerate(actions):
        #     if (action == 2 and hold == False) or (action == 1 and hold == False):
        #         date = dates[i]
        #         pct = treasure_df.query(f"Date == '{date}'")["treasure_pct"].iloc[0]
        #         treasures.append(pct)
        #     if action == 0:
        #         treasures.append(0)
        #         hold = True
        #     if action == 1 and hold == True:
        #         treasures.append(0)

        # pd.DataFrame({"index": indexes, "action": actions, "Date": dates})
        data = self.data_train if test_type == 'train' else self.data_test
        data.make_investment(actions)
        ev_agent = Evaluation(data.data, data.action_name, initial_investment, self.transaction_cost)
        return ev_agent







    # NOTE 似乎下面的都没啥用，但是又不敢删。
    def valid(self, initial_investment=1000, test_type='valid', arms=[], ratio_threshold=3,):
        """
        :@param file_name: name of the .pkl file to load the model
        :@param test_type: test results on train data or test data
        :@return returns an Evaluation object to have access to different evaluation metrics.
        """
        data = self.data_train if test_type == 'train' else self.data_test
        _data = torch.cat([_ for _ in data])
        import copy 
        test_net = copy.deepcopy(self.test_net)
        own_share = False

        for index, state in enumerate(_data):
            state = state.reshape(1, -1)

            for arm in arms:
                # NOTE: ts
                theta_hat = np.random.beta(arm["a"], arm["b"])
                # NOTE: greedy
                # theta_hat = arm["a"] / (arm["a"] + arm["b"])
                arm["theta"] = theta_hat

            max_index = int(sorted(arms, key=lambda x: x["theta"], reverse=True)[0]["index"])
            arm = arms[max_index]
            test_net.load_state_dict(torch.load(f"{arm['model_dir']}"))

            with torch.no_grad():
                test_net.eval()
                action = test_net(state).max(1)[1].view(1, 1)
                test_net.train()

            if action == 0:
                own_share = True
            if action == 2:
                own_share = False
            reward = self.get_valid_reward(_data, index, action.item(), own_share)
            arm["reward_list"].append(reward)
            sharpe = self.calc_sharpe(arm["reward_list"])            
                
            _count = 0
            for arm in arms:
                if arm["used"] == 0: continue
                _sharpe = self.calc_sharpe(arm["reward_list"])            
                if sharpe > _sharpe:
                    _count += 1

            if arm["used"] >= 2:
                if _count >= ratio_threshold:
                    arms[max_index]["a"] = arms[max_index]["a"] + 1
                else:
                    arms[max_index]["b"] = arms[max_index]["b"] + 1
            else:
                arms[max_index]["a"] = arms[max_index]["a"] + 1

            arm["used"] += 1

        for arm in arms:
            if arm["reward_list"].__len__() == 0:
                arm["theta"] = 0
            else:
                arm["theta"] = arm["a"] / (arm["a"] + arm["b"])
        
        arms = sorted(arms, key=lambda x: x["theta"], reverse=True)
        # print(arms)
        arm = arms[0]
        print(f"{arm['name']}-{arm['lamb']}", arm["a"], arm["b"])
        return arm

    def get_valid_reward(self, _data, index, action, own_share):
        """
        @param action: based on the action taken it returns the reward
        @return: reward
        """
        self.trading_cost_ratio = 0

        reward_index_first = index
        reward_index_last = index + 5
        if reward_index_last >= len(_data):
            return 0
        p1 = _data[reward_index_first][3]
        p2 = _data[reward_index_last][3]

        # NOTE 用每日的profit作为reward，导致的结果应该是agent会倾向选择在持有的时候价格上涨的周期进行持有。
        if action == 0 or (action == 1 and own_share):
            return (p2 / p1).item() - 1
        elif action == 2 and own_share:
            return (p1 / p2).item() - 1
        
        return 0