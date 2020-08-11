#from typing import Union, Dict, Tuple

from typing import Union, Dict, Tuple
import gym
import time
from gym import spaces
import random
import numpy as np


class TestRLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        # env.launch()
        print('env.launch().....')
        time.sleep(4)

        # _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,))

        rgb_shape = (64, 64, 3)
        space_dict = {'state': spaces.Box(low=-np.inf, high=np.inf, shape=rgb_shape),
                        'front_rgb': spaces.Box(low=0, high=1, shape=rgb_shape),
                        'left_shoulder_rgb': spaces.Box(low=0, high=1, shape=rgb_shape),
                        'right_shoulder_rgb': spaces.Box(low=0, high=1, shape=rgb_shape),
                        "wrist_rgb": spaces.Box(low=0, high=1, shape=rgb_shape)
        }
        self.observation_space = spaces.Dict(space_dict)

    def reset(self) -> Dict[str, np.ndarray]:
        time.sleep(2 * random.random())
        return 'reset yeah'

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        if random.random() < 0.2:
            time.sleep(500)
        else:
            time.sleep(2)
        return None, None, None, None

    def close(self) -> None:
        pass


from multiprocess_vector_env import MultiprocessVectorEnv
import argparse
import functools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    args = parser.parse_args()

    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    def make_batch_env(test):
        num_envs = args.num_envs
        return MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(num_envs))
            ]
        )
    def make_env(process_idx, test):
        env = TestRLBenchEnv()
        # env = GraspActionWrapper(RescaleAction(FlattenObservation(ResizeObservation(WristObsWrapper(gym.make(args.env)), (64, 64))), -0.5, 0.5))
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        return env, process_idx

    multiprocess_env = make_batch_env(False)

    train_steps = 100
    episode_len = 50
    for i in range(train_steps):
        print('============= step {} ==============='.format(i))
        if i % episode_len == 0:
            multiprocess_env.reset()
        multiprocess_env.step('hoge')



if __name__ == '__main__':
    main()
