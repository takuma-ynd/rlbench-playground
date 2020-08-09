#!/usr/bin/env python

import gym
from gym import spaces
import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget as Task
import numpy as np
import os
from multiprocess_vector_env import MultiprocessVectorEnv

from gym.wrappers import ResizeObservation, RescaleAction
from gym.wrappers.flatten_observation import FlattenObservation

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DIRTY!!
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64))

    def observation(self, observation):
        obs = np.transpose(observation, (2, 0, 1))  # hwc --> chw
        return obs


class WristObsWrapper(gym.ObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['front_rgb']
        # prev_space = self.observation_space['front_rgb']
        # DIRTY!!
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64))

    def observation(self, observation):
        obs = observation['front_rgb']
        # obs = np.transpose(obs, (2, 0, 1))  # hwc --> chw
        # print('obs.shape (WristObsWrapper)', obs.shape)
        return obs

class GraspActionWrapper(gym.ActionWrapper):
    r"""Rescales the continuous action space of the environment to a range [a,b].
    Example::
        >>> RescaleAction(env, a, b).action_space == Box(a,b)
        True
    """
    def __init__(self, env, action_size):
        assert isinstance(env.action_space, spaces.Box), (
            "expected Box action space, got {}".format(type(env.action_space)))
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_size,))

    def action(self, act):
        # Append grasp action (closed)
        return np.concatenate((act, [0.0]), axis=0)

    def reverse_action(self, act):
        return act[:-1]

class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env, speed=0.2):
        super().__init__(env)
        self.speed = speed

    def action(self, act):
        return self.speed * (act / np.linalg.norm(act))



"""A training script of PPO on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1709.06560 as much
as possible.
"""
import argparse
import functools

import gym
import gym.spaces
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl import experiments
from pfrl import utils


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="reach_target-ee-vision-v1",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2 * 10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to update model for per PPO iteration.",
    )
    parser.add_argument(
        "--action-size",
        type=int,
        default=3,
        help="Action size (needs to match env.action_space)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        render_mode = 'human' if args.render else None
        env = NormalizeAction(GraspActionWrapper(TransposeObs(ResizeObservation(WristObsWrapper(gym.make(args.env, render_mode=render_mode)), (64, 64))), args.action_size))
        # env = GraspActionWrapper(RescaleAction(FlattenObservation(ResizeObservation(WristObsWrapper(gym.make(args.env)), (64, 64))), -0.5, 0.5))
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = pfrl.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env, process_idx

    def make_batch_env(test):
        if test:
            num_envs = 1
        else:
            num_envs = args.num_envs
        return MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    # sample_env = RescaleAction(GraspActionWrapper(FlattenObservation(ResizeObservation(WristObsWrapper(gym.make(args.env)), (64, 64))), args.action_size), -0.5, 0.5)
    # timestep_limit = sample_env.spec.max_episode_steps
    timestep_limit = 200
    # obs_space = sample_env.observation_space
    # obs_space = spaces.Box(low=0, high=1, shape=(64 * 64 * 3,))
    obs_space = spaces.Box(low=0, high=1, shape=(3, 64, 64))
    # action_space = sample_env.action_space
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(args.action_size,))
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    # assert obs_space == spaces.Box(low=0, high=1, shape=(64 * 64 * 3,))
    # assert action_space == spaces.Box(low=-1.0, high=1.0, shape=(args.action_size,))
    # sample_env.close()

    assert isinstance(action_space, gym.spaces.Box)

    # Normalize observations based on their empirical mean and variance
    # import torch
    # dummy = torch.tensor(np.zeros((11, 3, 64, 64), dtype=np.float32))
    # import ipdb; ipdb.set_trace()
    # hoge = policy(dummy)

    # Combine a policy and a value function into a single model
    #
    env = make_batch_env(False)
    training_steps = 10 * 6
    episode_length = 200
    for i in range(training_steps):
        if i % episode_length == 0:
            obs = env.reset()
        print('step {}'.format(i))
        obs, reward, terminate, _ = env.step(np.asarray([action_space.sample() for _ in range(args.num_envs)], dtype=np.float32))

    print('done')
    env.close()


if __name__ == "__main__":
    main()
