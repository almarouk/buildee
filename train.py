import os.path
from pathlib import Path

from simulator import Simulator
from model import TimeSformerExtractor

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """Custom callback for saving model checkpoints during training."""

    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f'Iter {self.n_calls}: saved model weights')
        return True


class SimulatorEnv(gym.Env):
    def __init__(self, blend_dir: Path, max_steps: int = 50, show_rgb: bool = False):
        super().__init__()
        self.blend_dir = blend_dir

        # Define maximum number of steps
        self.max_steps = max_steps
        self.current_step = 0

        # Define image observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 8, 224, 224), dtype=np.uint8)

        # Define action space: xyz and yaw
        self.action_space = gym.spaces.Discrete(8)

        # Whether to display rendered image during training
        self.show_rgb = show_rgb

        # Setup frame observations indices
        self.frame_indices = np.int64(np.round(np.logspace(0, np.log10(max_steps * 2 / 3), 7))).tolist()

        # Initialize simulator and previous observations
        self.simulator = None
        self.prev_obs = []

    def reset(self, seed: int = None, options: dict = None) -> (gym.core.ObsType, dict):
        # Load a random scene
        blend_file = np.random.choice(list(self.blend_dir.glob('*.blend')))
        self.simulator = Simulator(blend_file, points_density=100.0)

        # Reset current step
        self.current_step = 0

        # Set initial camera pose
        self.simulator.respawn_camera()

        # Render image and update point cloud
        image = self.render_image()
        self.simulator.get_point_cloud(update_mask=True)

        # Build observations based on rendered image
        observations = np.zeros((3, 8, 224, 224), dtype=np.uint8)
        observations[:, 0] = image

        # Setup previous observations
        self.prev_obs = [image]

        return observations, {}

    def step(
            self, action: gym.core.ActType
    ) -> (gym.core.ObsType, float, bool, bool, dict):
        # Update current step
        self.current_step += 1

        # Apply action to the simulator
        match action:
            case 0:
                collided = self.simulator.move_camera_forward(1)
            case 1:
                collided = self.simulator.move_camera_forward(-1)
            case 2:
                collided = self.simulator.move_camera_down(1)
            case 3:
                collided = self.simulator.move_camera_down(-1)
            case 4:
                collided = self.simulator.move_camera_right(1)
            case 5:
                collided = self.simulator.move_camera_right(-1)
            case 6:
                collided = self.simulator.rotate_camera_yaw(22.5, degrees=True)
            case 7:
                collided = self.simulator.rotate_camera_yaw(-22.5, degrees=True)
            case _:
                raise ValueError(f'Invalid action: {action}')

        # Render image and update point cloud
        image = self.render_image()
        self.simulator.get_point_cloud(update_mask=True)

        # Build observations based on last 8 frames
        observations = np.zeros((3, 8, 224, 224), dtype=np.uint8)
        observations[:, 0] = image
        for i, frame_idx in enumerate(self.frame_indices):
            if frame_idx <= len(self.prev_obs):
                observations[:, i + 1] = self.prev_obs[-frame_idx]

        # Update previous observations
        self.prev_obs.append(image)

        # Display rendered image
        if self.show_rgb:
            cv2.imshow(f'rgb', cv2.cvtColor(np.moveaxis(image, 0, 2), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # Get reward, termination, and truncation
        is_last_step = self.current_step >= self.max_steps
        terminated = False
        reward = 0.0

        if collided:
            terminated = True
            reward = -1.0
        elif is_last_step:
            terminated = True
            reward = self.simulator.observed_points_mask.mean()

        return observations, reward, terminated, False, {}

    def render_image(self):
        rgb, depth = self.simulator.render()
        rgb = np.ascontiguousarray(np.moveaxis(rgb, 2, 0) * 255).astype(np.uint8)
        return rgb


if __name__ == "__main__":
    # Setup environment, logger, and checkpointer
    env = SimulatorEnv(blend_dir=Path('/home/clementin/Data/blendernbv/'), show_rgb=False)
    logger = logger.configure('logs', ['stdout', 'csv', 'tensorboard'])
    checkpointer = CheckpointCallback(save_freq=1000, save_path='checkpoint', verbose=1)

    # Setup policy
    policy_kwargs = {
        'features_extractor_class': TimeSformerExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
    }

    # Load or create model
    if os.path.exists('checkpoint.zip'):
        model = PPO.load('checkpoint', env=env, policy_kwargs=policy_kwargs, verbose=1)
    else:
        model = PPO(
            'CnnPolicy',
            env,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    # Train model
    model.set_logger(logger)
    model.learn(total_timesteps=10000000, callback=checkpointer)
