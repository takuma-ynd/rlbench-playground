import gym
import rlbench.gym
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget as Task
import numpy as np
import cv2
import os

env = gym.make('reach_target-vision-v0', render_mode='human')

# class Agent(object):

#     def __init__(self, action_size):
#         self.action_size = action_size
#         print('action size:', self.action_size)

#     def act(self, obs):
#         # arm = np.array([float(input('input {}'.format(i))) for i in range(self.action_size - 1)])
#         arm = np.random.normal(0.0, 0.2, size=(self.action_size - 1,))
#         gripper = [1.0]  # Always open
#         return np.concatenate([arm, gripper], axis=-1)


# obs_config = ObservationConfig()
# obs_config.set_all(True)

# action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
# env = Environment(
#     action_mode, obs_config=obs_config, headless=False)
# env.launch()

# task = env.get_task(Task)

# agent = Agent(env.action_size)



training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    env.render()  # Note: rendering increases step time.

    print(reward)
    path = os.path.join(os.path.dirname(__file__), 'observations' , '{:03}.jpg'.format(i))
    print(path)
    wrist_obs = obs['wrist_rgb']
    img_obs = (wrist_obs * 255).astype(np.uint8)
    img_obs = cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img_obs)
print('done')
env.close()
