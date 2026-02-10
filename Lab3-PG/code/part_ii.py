import gym
from networks import TanhGaussianPolicy
from algorithms import ActorCritic
from networks import ValueFunction
from torch.optim import Adam
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import csv

# Initialise Environment
env = gym.make("Pendulum-v0").unwrapped
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
state_dim = int(env.observation_space.shape[0])
action_dim = int(env.action_space.shape[0])
training_mode = 'Score' # Options: 'Score' trains the policy using the score function and 'Reparam' trains using the reparametrisation trick.
number_of_trials = 5
save_data = True
render_agent = False


# Set parameters
num_epochs = 30
num_pretrain_episodes = 1
num_training_updates_per_epoch = 200
lr = 3e-4
discount = 0.99

# Initialise data logger
results = np.zeros((num_epochs + 1, number_of_trials))

def run_experiment(s):

    policy = TanhGaussianPolicy(
        max_action=max_action,
        min_action=min_action,
        input_size=state_dim,
        output_size=action_dim
    )
    policy_optimiser = Adam(policy.parameters(), lr=lr)

    qf = ValueFunction(
        input_size=state_dim + action_dim,
    )
    qf_optimiser = Adam(qf.parameters(), lr=lr)

    vf = ValueFunction(
        input_size=state_dim,
    )
    vf_optimiser = Adam(vf.parameters(), lr=lr)

    algorithm = ActorCritic(
        policy=policy,
        policy_optimiser=policy_optimiser,
        qf=qf,
        qf_optimiser=qf_optimiser,
        vf=vf,
        vf_optimiser=vf_optimiser,
        env=env,
        discount=discount
    )

    for _ in range(num_pretrain_episodes):
        algorithm.sample_episode(exploration_mode=True)

    for e in range(num_epochs):

        algorithm.evaluate()
        print('Epoch: {}, Average Test Return: {}'.format(e, algorithm.average_return))
        results[e, s] = algorithm.average_return
        algorithm.reset()

        for n in range(num_training_updates_per_epoch):
            algorithm.env_step()
            if training_mode == 'Score':
                algorithm.train_score()
            else:
                algorithm.train_reparametrisation()


    algorithm.evaluate(render=render_agent)
    print('Epoch: {}, Average Test Return: {}'.format(e+1, algorithm.average_return))
    results[e+1, s] = algorithm.average_return

def write_to_file(mean, std):
    with open('PartII_{}_{}_trials_{}_epochs.csv'.format(training_mode, number_of_trials, num_epochs), 'w') as resultFile:
        wr = csv.writer(resultFile)
        mean.insert(0, 'Mean Returns')
        std.insert(0, 'Standard Deviation')
        data = [mean, std]
        data = list(map(list, zip(*data)))
        wr.writerows(data)

if __name__ == "__main__":

    for s in range(number_of_trials):
        th.manual_seed(s+1)
        run_experiment(s)

    mean_returns = results.mean(axis=1)
    std_returns = results.std(axis=1)
    training_steps = np.linspace(0, num_epochs * num_training_updates_per_epoch, num=num_epochs+1)
    plt.plot(training_steps, mean_returns)
    plt.fill_between(training_steps, mean_returns + std_returns, mean_returns - std_returns, alpha=0.3)
    plt.show()

    if save_data:
        write_to_file(mean_returns.tolist(), std_returns.tolist())
