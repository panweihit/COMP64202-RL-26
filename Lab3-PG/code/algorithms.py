import numpy as np
import torch as th
from torch import nn
from storage import Buffer
from torch.distributions import Uniform
from copy import deepcopy

class ActorCritic:

    def __init__(
            self,
            policy,
            qf,
            env,
            discount,
            qf_optimiser,
            policy_optimiser,
            max_evaluation_episode_length=200,
            num_evaluation_episodes=5,
            num_training_episode_steps=1000,
            batch_size=128,
            buffer_size = 10000,
            eval_deterministic = True,
            training_on_policy = False,
            vf=None,
            vf_optimiser=None
    ):

        self.policy = policy
        self.qf = qf
        self.vf = vf
        self.target_vf = deepcopy(vf)
        self.tau = 1e-2
        self.vf_optimiser = vf_optimiser
        self.qf_optimiser = qf_optimiser
        self.policy_optimiser = policy_optimiser
        self.env = env
        self.discount = discount
        self.batch_size = batch_size
        self.max_evaluation_episode_length = max_evaluation_episode_length
        self.num_evaluation_episodes = num_evaluation_episodes
        self.num_training_episode_steps = num_training_episode_steps
        self.training_on_policy = training_on_policy
        self.buffer = Buffer(buffer_size=buffer_size)
        self.loss = nn.MSELoss()
        self.pretraining_policy = Uniform(high=th.Tensor([policy.max_action]), low=th.Tensor([policy.min_action]))
        self.eval_deterministic = eval_deterministic

        self.R_av = None
        self.R_tot = 0

    def reset(self):
        self.state = th.from_numpy(self.env.reset()).float()

    def evaluate(self, render=False):

        total_return = 0

        for _ in range(self.num_evaluation_episodes):
            state = th.from_numpy(self.env.reset()).float()
            episode_return = 0

            for _ in range(self.max_evaluation_episode_length):
                action = self.policy.get_action(state, self.eval_deterministic)
                action = np.array([action.item()])

                if render:
                    self.env.render()
                state, reward, terminal, _ = self.env.step(action)
                state = th.from_numpy(state).float()

                episode_return += reward

                if terminal:
                    break

            total_return += episode_return

        self.average_return = total_return/self.num_evaluation_episodes

    def sample_episode(self, exploration_mode=False):

        self.reset()
        state = self.state

        for _ in range(self.num_training_episode_steps):

            if exploration_mode:
                action = self.pretraining_policy.sample()
            else:
                action = self.policy.get_action(state)
            next_state, reward, terminal, _ = self.env.step(action.numpy())
            next_state = th.from_numpy(next_state).float()
            reward = th.Tensor([reward])
            terminal = th.Tensor([terminal])

            self.buffer.add(state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            terminal=terminal)

            state = next_state
            if terminal:
                self.reset()
                state = self.state

    def env_step(self):

        state = self.state
        action = self.policy.get_action(state)
        next_state, reward, terminal, _ = self.env.step(action.numpy())
        next_state = th.from_numpy(next_state).float()
        reward = th.Tensor([reward])
        terminal = th.Tensor([terminal])

        self.buffer.add(state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        terminal=terminal)

        self.state = next_state

    def train_score(self):
        if self.training_on_policy:
            batch = self.buffer.whole_batch()
            self.buffer.clear()
        else:
            batch = self.buffer.random_batch(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        terminals = batch['terminals']

        new_actions, log_pis = self.policy.get_action_and_log_prob(states)
        values = self.vf(states)
        state_actions = th.cat((states, actions), 1)
        q_values = self.qf(state_actions)
        next_values = self.target_vf(next_states)
        new_state_actions = th.cat((states, new_actions), 1)
        new_q_values = self.qf(new_state_actions)

        """
        Value (Critic) Losses:
        """

        v_targets = new_q_values
        vf_loss = (v_targets.detach() - values).pow(2).mean()

        q_targets = rewards + self.discount * (1 - terminals) * next_values
        qf_loss = (q_targets.detach() - q_values).pow(2).mean()


        """
        Policy (Actor) Losses: TO COMPLETE IN EXERCISE II.2b
        """
        # policy_loss =

        """
        Gradient Updates
        """
        self.qf_optimiser.zero_grad()
        qf_loss.backward()
        self.qf_optimiser.step()

        self.vf_optimiser.zero_grad()
        vf_loss.backward()
        self.vf_optimiser.step()

        self.policy_optimiser.zero_grad()
        # policy_loss.backward()
        self.policy_optimiser.step()

        self.soft_update()

    def train_reparametrisation(self):

        if self.training_on_policy:
            batch = self.buffer.whole_batch()
            self.buffer.clear()
        else:
            batch = self.buffer.random_batch(self.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        terminals = batch['terminals']

        state_actions = th.cat((states, actions), 1)
        q_pred = self.qf(state_actions)
        v_pred = self.vf(states)
        new_actions, log_pis = self.policy.r_sample(states)

        """
        Value (Critic) Losses:
        """
        target_v_values = self.target_vf(next_states)
        q_target = rewards + (1. - terminals) * self.discount * target_v_values
        qf_loss = self.loss(q_pred, q_target.detach())

        new_state_actions = th.cat((states, new_actions), 1)
        q_new_actions = self.qf(new_state_actions)
        v_target = q_new_actions
        vf_loss = self.loss(v_pred, v_target.detach())

        """
        Policy (Actor) Loss: TO COMPLETE IN EXERCISE II.3c     
        """
        # policy_loss =


        """
        Gradient Updates
        """
        self.qf_optimiser.zero_grad()
        qf_loss.backward()
        self.qf_optimiser.step()

        self.vf_optimiser.zero_grad()
        vf_loss.backward()
        self.vf_optimiser.step()

        self.policy_optimiser.zero_grad()
        policy_loss.backward()
        self.policy_optimiser.step()

        self.soft_update()


    def soft_update(self):
        for target_param, param in zip(self.target_vf.parameters(), self.vf.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )