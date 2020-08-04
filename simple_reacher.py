import gym
import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget as Task
from pfrl_simple_reacher import WristObsWrapper, GraspActionWrapper, NormalizeAction
from gym.wrappers import ResizeObservation, RescaleAction
from gym.wrappers.flatten_observation import FlattenObservation
import numpy as np
import cv2
import os

env = NormalizeAction(GraspActionWrapper(FlattenObservation(ResizeObservation(WristObsWrapper(gym.make('reach_target-ee-vision-v0', render_mode='human')), (64, 64))), 3))
# env = gym.make('reach_target-vision-v0', render_mode='human')

training_steps = 512
episode_length = 512
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    env.render()  # Note: rendering increases step time.

    print(reward)
    # path = os.path.join(os.path.dirname(__file__), 'observations' , '{:03}.jpg'.format(i))
    # print(path)
    # img_obs = (wrist_obs * 255).astype(np.uint8)
    # img_obs = cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(path, img_obs)
print('done')
env.close()
