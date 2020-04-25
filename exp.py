import torch
import torch.multiprocessing as mp

from maml_rl.samplers.multi_task_sampler import SamplerWorker
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.utils.reinforcement_learning import reinforce_loss

import gym

from copy import deepcopy

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

'''
Let's see how just non-MAML version adapts

'''



task_queue = mp.JoinableQueue()
train_episodes_queue = mp.Queue()
valid_episodes_queue = mp.Queue()
policy_lock = mp.Lock()
env_name = "2DNavigation-v0"
env_kwargs = {
                    "low": -0.5,
                    "high": 0.5,
                    "task": {"goal": np.array([1, 1])}
                }
env = gym.make(env_name, **env_kwargs)
print(env.)

policy = get_policy_for_env(env,
                            hidden_sizes=(64, 64),
                            nonlinearity='tanh')
policy.share_memory()
baseline = LinearFeatureBaseline(get_input_size(env))
seed = None

worker = SamplerWorker(1,
                env_name,
                env_kwargs,
                20,
                env.observation_space,
                env.action_space,
                policy,
                deepcopy(baseline),
                seed,
                task_queue,
                train_episodes_queue,
                valid_episodes_queue,
                policy_lock)

# worker.sample(index=1, num_steps=5)

'''
Taken from SamplerWorker.sample function
'''
rewards = []
params = None
for step in range(500): # 10 is num_steps
    train_episodes = worker.create_episodes(params=params,
                                            gamma=0.99,
                                            gae_lambda=1.0,
                                            device='cpu')

    rewards.append(train_episodes.rewards.mean().item())
    # train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
    # QKFIX: Deep copy the episodes before sending them to their
    # respective queues, to avoid a race condition. This issue would 
    # cause the policy pi = policy(observations) to be miscomputed for
    # some timesteps, which in turns makes the loss explode.
    # self.train_queue.put((index, step, deepcopy(train_episodes)))
    
    # with self.policy_lock:
    loss = reinforce_loss(worker.policy, train_episodes, params=params)
    params = worker.policy.update_params(loss,
                                        params=params,
                                        step_size=3e-2,
                                        first_order=True)

plt.plot(list(range(len(rewards))), rewards)
plt.show()

pos = train_episodes.observations[::,1]
pos_x = pos[::,0]
pos_y = pos[::,1]

plt.plot(pos_x, pos_y)
plt.scatter([1], [1], c="r")
plt.scatter(pos_x, pos_y, c=list(range(len(pos_x))), cmap="Greens")
plt.show()

