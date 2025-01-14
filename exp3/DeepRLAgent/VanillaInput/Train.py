import torch
import torch.optim as optim
from DeepRLAgent.VanillaInput.DeepQNetwork import DQN
from DeepRLAgent.BaseTrain import BaseTrain
from DeepRLAgent.ReplayMemory import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 state_mode=1,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10,
                 arms={}):
        """
        This class is inherited from the BaseTrain class to initialize networks and other stuff that are specific to this
        model. For those parameters in the following explanation that I wrote: "for using in the name of the result file"
        the effect of those parameters has been applied in the Data class and are mentioned here only for begin used as
        part of the experiment's result filename.
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param n_classes: this is the feature vector size of the encoder's output.
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """

        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'DeepRL',
                                    state_mode,
                                    window_size,
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step)
        
        for arm in arms:
            arm["policy_net"] = DQN(data_train.state_size, 3)
            arm["target_net"] = DQN(data_train.state_size, 3)
            arm["target_net"].load_state_dict(arm["policy_net"].state_dict())
            arm["target_net"].eval()
            arm["optimizer"] = optim.Adam(arm["policy_net"].parameters())
            arm["memory"] = ReplayMemory(ReplayMemorySize)
            
         # # TODO 怎么设置根据arm数量生成变量？—— 直接装进dict就好了
        # self.policy_net2 = DQN(data_train.state_size, 3).to(device)
        # self.target_net2 = DQN(data_train.state_size, 3).to(device)
        # self.target_net2.load_state_dict(self.policy_net2.state_dict())
        # self.target_net2.eval()
        # self.optimizer2 = optim.Adam(self.policy_net2.parameters())

        self.test_net = DQN(self.data_train.state_size, 3)
        self.test_net