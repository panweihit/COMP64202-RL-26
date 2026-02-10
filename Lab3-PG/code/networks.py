from torch import nn as nn
from torch.nn import functional as f
from torch.distributions import Normal
import torch as th


class ValueFunction(nn.Module):

    def __init__(
            self,
            input_size,
    ):
        super(ValueFunction, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, activation):
        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        activation = self.fc3(activation)

        return activation


class GaussianPolicy(nn.Module):

    def __init__(
            self,
            output_size,
            input_size,
            max_action,
            min_action,
            soft_clamp_function=None
    ):
        super(GaussianPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.soft_clamp_function = soft_clamp_function
        self.max_action = max_action
        self.min_action = min_action
        self.max_log_sig = 2
        self.min_log_sig = -20

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, self.output_size)
        self.log_sig_head = nn.Linear(256, self.output_size)

    def forward(self, activation):

        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        mu = self.mu_head(activation)
        log_sig = self.log_sig_head(activation)
        log_sig = log_sig.clamp(min=self.min_log_sig, max=self.max_log_sig)
        sig = th.exp(log_sig)

        return mu, sig

    def get_action(self, state, eval_deterministic=False):

        mu, sig = self.forward(state)
        if eval_deterministic:
            action = mu
        else:
            gauss = Normal(loc=mu, scale=sig)
            action = gauss.sample()
            action.detach()

        action = self.max_action * th.tanh(action / self.max_action)
        return action

    def get_action_and_log_prob(self, state):

        mu, sig = self.forward(state)
        gauss = Normal(loc=mu, scale=sig)
        action = gauss.sample()
        action.detach()
        action = action.clamp(min=self.min_action, max=self.max_action)
        log_prob = gauss.log_prob(action)

        return action, log_prob

    def r_sample(self, state):

        mu, sig = self.forward(state)
        loc = th.zeros(size=[state.shape[0], 1], dtype=th.float32)
        scale = loc + 1.0
        unit_gauss = Normal(loc=loc, scale=scale)
        gauss = Normal(loc=mu, scale=sig)
        epsilon = unit_gauss.sample()
        action = mu + sig * epsilon
        action = action.requires_grad_()
        action = self.max_action * th.tanh(action / self.max_action)
        log_prob = gauss.log_prob(action.data)

        return action, log_prob


class TanhGaussianPolicy(nn.Module):

    def __init__(
            self,
            output_size,
            input_size,
            max_action,
            min_action,
            soft_clamp_function=None
    ):
        super(TanhGaussianPolicy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.soft_clamp_function = soft_clamp_function
        self.max_action = max_action
        self.min_action = min_action
        self.max_log_sig = 2
        self.min_log_sig = -20
        self.a_diff = 0.5 * (self.max_action - self.min_action)
        self.a_shift = 0.5 * (self.max_action + self.min_action)
        self.epsilon = 1e-6

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, self.output_size)
        self.log_sig_head = nn.Linear(256, self.output_size)

    def forward(self, activation):

        activation = f.relu(self.fc1(activation))
        activation = f.relu(self.fc2(activation))
        mu = self.mu_head(activation)
        log_sig = self.log_sig_head(activation)
        log_sig = log_sig.clamp(min=self.min_log_sig, max=self.max_log_sig)
        sig = th.exp(log_sig)

        return mu, sig

    def tanh_function(self, a):

        a = self.a_diff * th.tanh(a / self.a_diff) + self.a_shift

        return a

    def tanh_function_derivative(self, a):

        return 1 - (th.tanh(a / self.a_diff) ** 2) + self.epsilon

    def get_action(self, state, eval_deterministic=False):

        mu, sig = self.forward(state)
        if eval_deterministic:
            action = self.tanh_function(mu)
        else:
            gauss = Normal(loc=mu, scale=sig)
            action = gauss.sample()
            action = self.tanh_function(action)

        return action

    def get_action_and_log_prob(self, state):

        mu, sig = self.forward(state)
        gauss = Normal(loc=mu, scale=sig)
        pre_tanh_action = gauss.sample()
        pre_tanh_log_prob = gauss.log_prob(pre_tanh_action)
        action = self.tanh_function(pre_tanh_action)
        log_prob = pre_tanh_log_prob - th.log(self.tanh_function_derivative(pre_tanh_action))

        return action, log_prob

    def r_sample(self, state):

        mu, sig = self.forward(state)
        loc = th.zeros(size=[state.shape[0], 1], dtype=th.float32)
        scale = loc + 1.0
        unit_gauss = Normal(loc=loc, scale=scale)
        epsilon = unit_gauss.sample()
        pre_tanh_action = mu + sig * epsilon
        action = self.tanh_function(pre_tanh_action)

        gauss = Normal(loc=mu, scale=sig)
        pre_tanh_log_prob = gauss.log_prob(pre_tanh_action)
        log_prob = pre_tanh_log_prob - th.log(self.tanh_function_derivative(pre_tanh_action))

        return action, log_prob
