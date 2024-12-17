import numpy as np
from simulator import Simulator
import gymnasium as gym
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common import logger


class SimulatorEnv(gym.Env):
    def __init__(self, blend_file: str, max_steps: int = 100):
        super().__init__()
        self.blend_file = blend_file

        # Initialize simulator
        self.simulator = None

        # Initialize all previously observed points
        self.observed_points_mask = None

        # Define maximum number of steps
        self.max_steps = max_steps
        self.current_step = 0

        # Define image observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 256, 256), dtype=np.uint8)

        # Define action space: xyz and yaw
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32),
        )

    def reset(self, seed: int = None, options: dict = None) -> (gym.core.ObsType, dict):
        """Reset the environment and return initial observation."""
        self.simulator = Simulator(self.blend_file, points_density=10.0)
        self.current_step = 0

        # Get initial observed points
        self.simulator.get_point_cloud(update_mask=True)
        self.observed_points_mask = self.simulator.observed_points_mask.copy()

        # Get observation
        image = self.render_image()

        return image, {}

    def step(
            self, action: gym.core.ActType
    ) -> (gym.core.ObsType, float, bool, bool, dict):
        terminated = True
        reward = -1.0

        x, y, z, yaw = action
        collided = self.simulator.set_camera_from_next_camera(np.array([
            [np.cos(yaw), 0.0, np.sin(yaw), x],
            [0.0, 1.0, 0.0, y],
            [-np.sin(yaw), 0.0, np.cos(yaw), z],
            [0.0, 0.0, 0.0, 1.0]
        ]))
        image = self.render_image()
        self.simulator.get_point_cloud(update_mask=True)

        cv2.imshow(f'rgb', cv2.cvtColor(np.moveaxis(image, 0, 2), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        if not collided:
            terminated = False
            reward = self.compute_coverage_gain()

        truncated = self.current_step >= self.max_steps
        self.current_step += 1

        return image, reward, terminated, truncated, {}

    def render_image(self):
        rgb, depth = self.simulator.render()
        rgb = np.ascontiguousarray(np.moveaxis(rgb, 2, 0) * 255).astype(np.uint8)
        return rgb

    def compute_coverage_gain(self) -> float:
        new_observed_points_mask = self.simulator.observed_points_mask.copy()
        coverage_gain = np.mean(new_observed_points_mask & ~self.observed_points_mask)
        self.observed_points_mask = new_observed_points_mask
        return coverage_gain


if __name__ == "__main__":
    env = SimulatorEnv(blend_file='liberty.blend')
    logger = logger.configure('logs', ['stdout', 'csv', 'tensorboard'])
    model = PPO(
        "CnnPolicy",  # Use a CNN-based policy for image-based observations
        env,
        verbose=1
    )
    model.set_logger(logger)
    model.learn(total_timesteps=100000)
