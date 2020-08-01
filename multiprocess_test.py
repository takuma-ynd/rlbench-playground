#!/usr/bin/env python
import numpy as np
import gym
import rlbench.gym
import functools
from multiprocess_vector_env import MultiprocessVectorEnv

def make_env(process_idx, test):
    process_seeds = np.arange(1)
    # env = ResizeObservation(WristObsWrapper(gym.make(args.env)), (64, 64))
    env = gym.make("reach_target-vision-v0")
    # env = gym.make("CartPole-v1")
    # Use different random seeds for train and test envs
    process_seed = int(process_seeds[process_idx])
    env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
    env.seed(env_seed)
    # Cast observations to float32 because our model uses float32
    # env = pfrl.wrappers.CastObservationToFloat32(env)
    # if args.monitor:
    #     env = pfrl.wrappers.Monitor(env, args.outdir)
    # if args.render:
    #     env = pfrl.wrappers.Render(env)
    return env

def make_batch_env(test=False):
    num_envs = 1
    return MultiprocessVectorEnv(
        [
            functools.partial(make_env, idx, test)
            for idx, env in enumerate(range(num_envs))
        ]
    )

if __name__ == '__main__':
    hoge = make_batch_env()
