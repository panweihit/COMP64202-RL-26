import random
import torch as th

class Buffer:

    def __init__(self, buffer_size):

        self.clear()
        self.buffer_size=buffer_size

    def add(self,
            state,
            action,
            reward,
            next_state,
            terminal
            ):

        if len(self.states) ==  self.buffer_size:
            self.clear_earliest_entry()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(terminal)


    def clear(self):

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []


    def clear_earliest_entry(self):

        self.states = self.states[1:]
        self.actions = self.actions[1:]
        self.rewards = self.rewards[1:]
        self.next_states = self.next_states[1:]
        self.terminals = self.terminals[1:]


    def random_batch(self, batch_size=None):

        combined = list(zip(self.states,
                            self.actions,
                            self.rewards,
                            self.next_states,
                            self.terminals))

        random.shuffle(combined)

        if batch_size is not None:
            combined = combined[:batch_size]

        states, actions, rewards, next_states, terminals = zip(*combined)

        batch = {}
        batch['states'] = th.stack(states)
        batch['actions'] = th.stack(actions)
        batch['rewards'] = th.stack(rewards)
        batch['next_states'] = th.stack(next_states)
        batch['terminals'] = th.stack(terminals)


        return batch

    def whole_batch(self,):

        batch = {}
        batch['states'] = th.stack(self.states)
        batch['actions'] = th.stack(self.actions)
        batch['rewards'] = th.stack(self.rewards)
        batch['next_states'] = th.stack(self.next_states)
        batch['terminals'] = th.stack(self.terminals)


        return batch
