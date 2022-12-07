import numpy as np
import torch
import torch.nn as nn

class Tracker:
    def __init__(self, writer, group_losses=1):
        self.writer = writer
        self.loss_buf = []
        self.total_loss = []
        self.steps_buf = []
        self.group_losses = group_losses
        self.capacity = group_losses*10

    def loss(self, loss, frame):
        assert (isinstance(loss, np.float))
        self.loss_buf.append(loss)
        if len(self.loss_buf) < self.group_losses:
            return False
        mean_loss = np.mean(self.loss_buf)
        self.loss_buf.clear()
        self.total_loss.append(mean_loss)
        movingAverage_loss = np.mean(self.total_loss[-100:])
        if len(self.total_loss) > self.capacity:
            self.total_loss = self.total_loss[1:]

        self.writer.add_scalar("loss_100", movingAverage_loss, frame)
        self.writer.add_scalar("loss", mean_loss, frame)

def calc_loss(batch, agent, gamma, train_on_gpu):
    if train_on_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = states
    next_states_v = next_states
    actions_v = torch.tensor(actions, device=device)
    rewards_v = torch.tensor(rewards, device=device)
    if train_on_gpu:
        done_mask = torch.cuda.BoolTensor(dones)
    else:
        done_mask = torch.BoolTensor(dones)

    state_action_values = agent.get_Q_value(states_v).gather(1, actions_v).squeeze(-1)
    next_state_actions = agent.get_Q_value(next_states_v).max(1)[1]
    next_state_values = agent.get_Q_value(next_states_v, tgt=True).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)