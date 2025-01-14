from .Data import Data
import numpy as np


class DataRLAgent(Data):
    def __init__(self, data, state_mode, action_name, device, gamma, n_step=4, batch_size=50, window_size=1,
                 transaction_cost=0.0):
        start_index_reward = 0 if state_mode != 5 else window_size - 1
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=start_index_reward, transaction_cost=transaction_cost)

        self.data_kind = 'AutoPatternExtraction'

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
        self.state_mode = state_mode

        self.state_size = 4

        for i in range(len(self.data_preprocessed)):
            self.states.append(self.data_preprocessed[i])

    def find_trend(self, window_size=20):
        self.data['MA'] = self.data.mean_candle.rolling(window_size).mean()
        self.data['trend_class'] = 0

        for index in range(len(self.data)):
            moving_average_history = []
            if index >= window_size:
                for i in range(index - window_size, index):
                    moving_average_history.append(self.data['MA'][i])
            difference_moving_average = 0
            for i in range(len(moving_average_history) - 1, 0, -1):
                difference_moving_average += (moving_average_history[i] - moving_average_history[i - 1])

            # trend = 1 means ascending, and trend = 0 means descending
            self.data['trend_class'][index] = 1 if (difference_moving_average / window_size) > 0 else 0
 