import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, state_length, action_length):
        super(DQN, self).__init__()
        
        # num = 128
        # self.policy_network = nn.Sequential(
        #     nn.Linear(state_length, num),
        #     nn.BatchNorm1d(num),
        #     nn.Linear(num, num),
        #     nn.BatchNorm1d(num),
        #     nn.Linear(num, action_length)
        # )

        self.policy_network = nn.Sequential(
            nn.Linear(state_length, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, action_length)
        )


    def forward(self, x):
        return self.policy_network(x)
