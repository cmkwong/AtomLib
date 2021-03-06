import copy
import numpy as np
import torch

def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

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
        self.model = dqn_model
        self.target_model = copy.deepcopy(dqn_model)
        self.action_selector = action_selector
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        :param states: [state]
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