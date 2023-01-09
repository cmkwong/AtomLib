import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func

def default_states_preprocessor(states, unitVector=True, train_on_gpu=True):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    assert isinstance(states, list)
    # choose device
    if train_on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # pre-process the states
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    t_v = func.normalize(torch.from_numpy(np_states).to(device), p=2, dim=1) if unitVector else torch.from_numpy(np_states).to(device)
    return t_v

def attention_states_preprocessor(states):
    """
    :param states: [{ encoderInput: np.array(), status: np.array() }]
    :return:
    """
    device = torch.device('cuda')
    # pre-process the states
    if len(states) == 0:
        print("states cannot be empty")
        return False
    # assign the first states
    np_encoderInput = np.expand_dims(states[0]['encoderInput'], 0)
    np_status = np.expand_dims(states[0]['status'], 0)
    # if there is more than one, loop for concat
    for s in states[1:]:
        np_encoderInput = np.concatenate((np_encoderInput, np.expand_dims(s['encoderInput'], 0)), axis=0)
        np_status = np.concatenate((np_status, np.expand_dims(s['status'], 0)), axis=0)
    t_encoderInput = torch.from_numpy(np_encoderInput).type(torch.float32).to(device)
    t_status = torch.from_numpy(np_status).type(torch.float32).to(device)
    return {'encoderInput': t_encoderInput, 'status': t_status}

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.from_numpy(np_states)

class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError

class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, preprocessor=default_states_preprocessor):
        self.device = torch.device("cuda")
        self.model = dqn_model
        self.target_model = copy.deepcopy(dqn_model)
        self.action_selector = action_selector
        self.preprocessor = preprocessor

    def unpack_batch(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in batch:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)  # the result will be masked anyway
            else:
                last_states.append(exp.last_state)
        return states, np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), last_states

    def calc_loss(self, batch, gamma):
        """
        :param batch: [state]
        :param gamma: float
        :return:
        """
        states, actions, rewards, dones, next_states = self.unpack_batch(batch)

        states_v = states
        next_states_v = next_states
        actions_v = torch.from_numpy(actions).to(self.device)
        rewards_v = torch.from_numpy(rewards).to(self.device)
        done_mask = torch.cuda.BoolTensor(dones)

        state_action_values = self.get_Q_value(states_v).gather(1, actions_v).squeeze(-1)
        next_state_actions = self.get_Q_value(next_states_v).max(1)[1]
        next_state_values = self.get_Q_value(next_states_v, tgt=True).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        loss = nn.L1Loss()(state_action_values, expected_state_action_values)
        # if torch.isnan(loss):
        #     print('error')
        return nn.L1Loss()(state_action_values, expected_state_action_values)

    def getActionIndex(self, states, agent_states=None):
        """
        :param states: [torch tensor] with shape (1, 57)
        :param agent_states:
        :return:
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        states = self.preprocessor(states) # states is a list
        q_v = self.model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states

    def get_Q_value(self, states, tgt=False):
        """
        :param states: [state]
        :return: pyTorch tensor
        """
        states = self.preprocessor(states)  # states is a list
        if not tgt:
            q_v = self.model(states)
        else:
            q_v = self.target_model(states)
        return q_v

    def sync(self):
        """
        sync the model and target model
        """
        self.target_model.load_state_dict(self.model.state_dict())

class Supervised_DQNAgent(BaseAgent):
    def __init__(self, dqn_model, action_selector, sample_sheet, assistance_ratio=0.2):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.sample_sheet = sample_sheet # name tuple
        self.assistance_ratio = assistance_ratio

    def __call__(self, states, agent_states=None):
        batch_size = len(states)
        if agent_states is None:
            agent_states = [None] * batch_size
        sample_mask = np.random.random(batch_size) <= self.assistance_ratio
        sample_actions_ = []
        dates = [state.date for state in states[sample_mask]]
        for date in dates:
            for i, d in enumerate(self.sample_sheet.date):
                if d == date:
                    sample_actions_.append(self.sample_sheet.action[i])
        sample_actions = np.array(sample_actions_)   # convert into array

        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        actions[sample_mask] = sample_actions
        return actions, agent_states

class DQNAgentAttn(DQNAgent):
    def __init__(self, dqn_model, action_selector, preprocessor=attention_states_preprocessor):
        super(DQNAgentAttn, self).__init__(dqn_model, action_selector, preprocessor)

    # def unpack_batch(self, batch):
    #     pass
    #
    # def calc_loss(self, batch, gamma):
    #     pass

# class TargetNet:
#     """
#     Wrapper around model which provides copy of it instead of trained weights
#     """
#     def __init__(self, model):
#         self.model = model
#         self.target_model = copy.deepcopy(model)
#
#     def sync(self):
#         self.target_model.load_state_dict(self.model.state_dict())
#
#     def alpha_sync(self, alpha):
#         """
#         Blend params of target net with params from the model
#         :param alpha:
#         """
#         assert isinstance(alpha, float)
#         assert 0.0 < alpha <= 1.0
#         state = self.model.state_dict()
#         tgt_state = self.target_model.state_dict()
#         for k, v in state.items():
#             tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
#         self.target_model.load_state_dict(tgt_state)