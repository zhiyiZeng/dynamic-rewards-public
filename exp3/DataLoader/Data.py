import torch
import math
import numpy as np


class Data:
    def __init__(self, data, action_name, device, gamma, n_step=4, batch_size=50, start_index_reward=0,
                 transaction_cost=0):
        """
        This class is the environment that interacts with the agent.
        @param data: this is the data_train or data_test in the DataLoader
        @param action_name: This is the name of the action (typically the name of the model who generated
        those actions) column in the original data-frame of the input data.
        @param device: cpu or gpu
        @param gamma: in the algorithm
        @param n_step: number of future steps of reward
        @param batch_size:
        @param start_index_reward: for sequential input, the start index for reward is not 0. Therefore, it should be
        provided as a function of window-size.
        @param transaction_cost: cost of each transaction applied in the reward function.
        """
        self.data = data
        self.states = []
        self.current_state_index = -1
        self.own_share = False
        self.batch_size = batch_size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.close_price = list(data.close)
        self.action_name = action_name
        self.code_to_action = {0: 'buy', 1: 'None', 2: 'sell'}
        self.start_index_reward = start_index_reward

        self.trading_cost_ratio = transaction_cost

    def get_current_state(self):
        """
        @return: returns current state of the environment
        """
        self.current_state_index += 1
        if self.current_state_index == len(self.states):
            return None
        return self.states[self.current_state_index]

    def step(self, action, lamb, reward_type):
        """
        right now, the reward is one step...
        TODO: I should promote it to n-step reward with baseline (average reward)
        :param action:  0 -> Buy
                        1 -> None
                        2 -> Sell
        :return:
        """
        done = False
        next_state = None
        self.lamb = lamb
        if self.current_state_index + self.n_step < len(self.states):
            next_state = self.states[self.current_state_index + self.n_step]
        else:
            done = True

        if action == 0:
            self.own_share = True
        elif action == 2:
            self.own_share = False

        reward = 0
        if not done:
            reward = self.get_reward(action, lamb, reward_type)

        return done, reward, next_state

    def calc_downsiderisk(self, first_index, last_index):
        x = self.close_price[first_index:last_index+1]
        x = np.array(x)
        mean = np.mean(x)
        std = np.std(x[x < mean])
        return std

    def calc_sharpe(self, first_index, last_index):
        x = self.close_price[first_index:last_index+1]
        x = np.array(x)
        mean = np.mean(x)
        std = np.std(x)
        if std != 0:
            return mean / std
        else:
            return mean
    
    def calc_upsiderisk(self, first_index, last_index):
        x = self.close_price[first_index:last_index+1]
        x = np.array(x)
        mean = np.mean(x)
        std = np.std(x[x > mean])
        return std

    def get_reward(self, action, lamb, reward_type):
        """
        @param action: based on the action taken it returns the reward
        @return: reward
        """
        self.lamb = lamb

        reward_index_first = self.current_state_index + self.start_index_reward
        reward_index_last = self.current_state_index + self.start_index_reward + self.n_step \
            if self.current_state_index + self.n_step < len(self.states) else len(self.close_price) - 1

        p1 = self.close_price[reward_index_first]
        p2 = self.close_price[reward_index_last]

        # NOTE 用每日的profit作为reward，导致的结果应该是agent会倾向选择在持有的时候价格上涨的周期进行持有。
        reward = 0
        if reward_type == "profit":
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = (1 - self.trading_cost_ratio) * p2 / p1 - 1  # profit in percent
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = (1 - self.trading_cost_ratio) * p1 / p2 - 1
            reward = reward * 100

        if reward_type == "regularized":
            downside_risk = self.calc_downsiderisk(reward_index_first, reward_index_last)
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = (((1 - self.trading_cost_ratio) * p2 / p1 - 1) - self.lamb * downside_risk)  # profit in percent
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = (((1 - self.trading_cost_ratio) * p1 / p2 - 1) + self.lamb * downside_risk)
            reward = reward * 100

        if reward_type == "sharpe":
            sharpe = self.calc_sharpe(reward_index_first, reward_index_last)
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = self.lamb * sharpe * (1 - self.trading_cost_ratio)
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = -self.lamb * sharpe * (1 - self.trading_cost_ratio)
        
        if reward_type == "volatility":
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                risk = self.calc_upsiderisk(reward_index_first, reward_index_last)
                reward = self.lamb * risk * (1 - self.trading_cost_ratio)
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                risk = self.calc_downsiderisk(reward_index_first, reward_index_last)
                reward = self.lamb * risk * (1 - self.trading_cost_ratio)
        
        if reward_type == "old_profit":
            p2 = self.close_price[reward_index_first]
            p1 = self.close_price[reward_index_first - 1]
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = (1 - self.trading_cost_ratio) * p2 / p1 - 1  # profit in percent
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = (1 - self.trading_cost_ratio) * p1 / p2 - 1
            reward = reward * 100

        if reward_type == "future_profit_1":
            p1 = self.close_price[reward_index_first]
            reward_index_last = self.current_state_index + self.start_index_reward + 1 \
                if self.current_state_index + 1 < len(self.states) else len(self.close_price) - 1
            p2 = self.close_price[reward_index_last]

            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = (1 - self.trading_cost_ratio) * p2 / p1 - 1  # profit in percent
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = (1 - self.trading_cost_ratio) * p1 / p2 - 1
            reward = reward * 100
        
        if reward_type == "future_profit_10":
            p1 = self.close_price[reward_index_first]
            reward_index_last = self.current_state_index + self.start_index_reward + 10 \
                if self.current_state_index + 10 < len(self.states) else len(self.close_price) - 1
            p2 = self.close_price[reward_index_last]
            if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
                reward = (1 - self.trading_cost_ratio) * p2 / p1 - 1  # profit in percent
            elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
                reward = (1 - self.trading_cost_ratio) * p1 / p2 - 1
            reward = reward * 100
        
        return reward


    def calculate_reward_for_one_step(self, action, index, rewards):
        """
        The reward for selling is the opposite of the reward for buying, meaning that if some one sells his share and the
        value of the share increases, thus he should be punished. In addition, if some one sells appropriately and the value
        of the share decreases, he should be awarded
        :param action:
        :param index:
        :param rewards:
        :param own_share: whether the user holds the share or not.
        :return:
        """
        index += self.start_index_reward  # Last element inside the window
        if action == 0 or (action == 1 and self.own_share):  # Buy Share or Hold Share
            difference = self.close_price[index + 1] - self.close_price[index]
            rewards.append(difference)

        elif action == 2 or (action == 1 and not self.own_share):  # Sell Share or No Share
            difference = self.close_price[index] - self.close_price[index + 1]
            rewards.append(difference)

    def reset(self):
        self.current_state_index = -1
        self.own_share = False

    def __iter__(self):
        self.index_batch = 0
        self.num_batch = math.ceil(len(self.states) / self.batch_size)
        return self

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [torch.tensor([s], dtype=torch.float, device=self.device) for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch)

        raise StopIteration

    def get_total_reward(self, action_list):
        """
        You should call reset before calling this function, then it receives action batch
        from the input and calculate rewards.
        :param action_list:
        :return:
        """
        total_reward = 0
        for a in action_list:
            if a == 0:
                self.own_share = True
            elif a == 2:
                self.own_share = False
            self.current_state_index += 1
            total_reward += self.get_reward(a)

        return total_reward

    def make_investment(self, action_list):
        """
        Provided a list of actions at each time-step, it converts the action to its original name like:
        0 -> Buy
        1 -> None
        2 -> Sell
        @param action_list: ...
        @return: ...
        """
        self.data[self.action_name] = 'None'
        i = self.start_index_reward + 1
        for a in action_list:
            self.data[self.action_name][i] = self.code_to_action[a]
            i += 1
