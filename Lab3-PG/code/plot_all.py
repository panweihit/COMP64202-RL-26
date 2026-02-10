import csv
import matplotlib.pyplot as plt
import numpy as np

with open('PartII_Score_5_trials_30_epochs.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)
  data = data[1:][:]
  data = list(map(list, zip(*data)))
  score_means = data[0][:]
  score_stds = data[1][:]

with open('PartII_Reparam_5_trials_30_epochs.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)
  data = data[1:][:]
  data = list(map(list, zip(*data)))
  reparam_means = data[0][:]
  reparam_stds = data[1][:]

training_steps = np.linspace(0, (len(reparam_means)-1) * 200, num=len(reparam_means))
score_means = np.array(score_means, dtype=np.float64)
score_stds = np.array(score_stds, dtype=np.float64)
reparam_means = np.array(reparam_means, dtype=np.float64)
reparam_stds = np.array(reparam_stds, dtype=np.float64)

fig, ax = plt.subplots()

ax.plot(training_steps, score_means, label='Score Function')
ax.fill_between(training_steps, score_means + score_stds, score_means - score_stds, alpha=0.3)

ax.plot(training_steps, reparam_means, color='orange', label='Reparametrisation Trick')
ax.fill_between(training_steps, reparam_means + reparam_stds, reparam_means - reparam_stds, alpha=0.3, color='orange')

ax.legend()
plt.xlabel('Number of Training Steps')
plt.ylabel('Average Return')

plt.show()