import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from time import time

"""
PointEnv from rllab
The goal is to control an agent and get it to the target located at (0,0).
At each timestep the agent gets its current location (x,y) as observation, 
takes an action (dx,dy), and is transitioned to (x+dx, y+dy).
"""


class PointEnv():
    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        state = np.copy(self._state)
        return state

    def step(self, action):
        action = np.clip(action, -1, 1)
        self._state = self._state + 0.1 * action
        x, y = self._state
        reward = -(x ** 2 + y ** 2) ** 0.5 - 0.02 * np.sum(action ** 2)
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_state = np.copy(self._state)
        return next_state, reward, done


class Gauss_Policy():
    def __init__(self):
        self.action_dim = 2
        self.theta = 0.5 * np.ones(4)
        # theta here is a length 4 array instead of a matrix for ease of processing
        # Think of treating theta as a 2x2 matrix and then flatenning it, which gives us:
        # action[0] = state[0]*[theta[0], theta[1]]
        # action[1] = state[1]*[theta[2], theta[3]]

    def get_action_and_grad(self, state):
        # Exercise I.1:
        mean_act = []
        sampled_act = []
        grad_log_pi = []
        return sampled_act, grad_log_pi


# This function collects some trajectories, given a policy
def gather_paths(env, policy, num_paths, max_ts=500):
    paths = []
    for i in range(num_paths):
        ts = 0
        states = []
        act = []
        grads = []
        rwd = []
        done = False
        s = env.reset()
        while not done and ts < max_ts:
            a, grad_a = policy.get_action_and_grad(s)
            next_s, r, done = env.step(a)
            states += [s]
            act += [a]
            rwd += [r]
            grads += [grad_a]
            s = next_s
            ts += 1
        path = {'states': np.array(states),
                'actions': np.array(act),
                'grad_log_pi': np.array(grads),
                'rwd': np.array(rwd)}
        paths += [path]
    return paths


def baseline(paths):
    path_features = []
    for path in paths:
        s = path["states"]
        l = len(path["rwd"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        path_features += [np.concatenate([s, s ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)]
    ft = np.concatenate([el for el in path_features])
    targets = np.concatenate([el['returns'] for el in paths])

    # Exercise I.2(a): Compute the regression coefficents
    coeffs = []
    # Exercise I.2(b): Calculate the values for each state
    for i, path in enumerate(paths):
        path['value'] = []


def process_paths(paths, discount_rate=1):
    grads = []
    for path in paths:
        # Exercise I.3a: Implement the discounted return
        # Hint: This can be done in one line using lfilter from scipy.signal,
        # but it might be much easier to write a separate function for this.
        path['returns'] = []
    baseline(paths)
    for path in paths:
        path['adv'] = path['returns'] - path['value']
        rets_for_grads = np.atleast_2d(path['adv']).T
        rets_for_grads = np.repeat(rets_for_grads, path['grad_log_pi'].shape[1], axis=1)
        path['grads'] = path['grad_log_pi'] * rets_for_grads
        grads += [np.sum(path['grads'], axis=0)]
    grads = np.sum(grads, axis=0) / len(paths)
    return grads


# Run algo
env = PointEnv()
alpha = 0.01
traj_len = 50
perf_stats = []


def run_algo(env, alpha, gamma, traj_len, num_itr=200, runs=10):
    rwd = np.zeros((num_itr, runs))
    for st in range(runs):
        policy = Gauss_Policy()
        for i in range(num_itr):
            paths = gather_paths(env, policy, max_ts=traj_len, num_paths=5)
            rwd[i, st] = np.mean([np.sum(path['rwd']) for path in paths])
            grads = process_paths(paths, discount_rate=gamma)
            policy.theta += alpha * grads
    perf_stats = {'gamma': gamma,
                  'mean_rwd': np.mean(rwd, axis=1),
                  'std_err': np.std(rwd, axis=1) / np.sqrt(runs)}
    return perf_stats


gamma = [0.99, 0.995, 1.0]
for g in gamma:
    print("Starting algorithm with gamma:", g)
    perf_stats += [run_algo(env, alpha, gamma=g, traj_len=traj_len)]

# And plot the results
for el in perf_stats:
    plt.plot(el['mean_rwd'], label='discount factor = ' + str(el['gamma']))
    plt.fill_between(np.arange(len(el['mean_rwd'])), el['mean_rwd'] + el['std_err'], el['mean_rwd'] - el['std_err'],
                     alpha=0.3)
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Returns')
plt.xlim([0, 200])
plt.show()

# Exercise I.3(b): Run the algo again, but with traj_len=500.
# Does the relative performance of learning using discount factors change?