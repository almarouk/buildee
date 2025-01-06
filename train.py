import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from model import TimeSformerExtractor

from simulator import Simulator


class SimulatorEnv(gym.Env):
    def __init__(self, blend_file: str, max_steps: int = 20):
        super().__init__()
        self.blend_file = blend_file

        # Initialize simulator
        self.simulator = None

        # Define maximum number of steps
        self.max_steps = max_steps
        self.current_step = 0

        # Define image observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)

        # Define action space: xyz and yaw
        self.action_space = gym.spaces.Discrete(8)

        # Initialize simulator
        self.simulator = Simulator(self.blend_file, points_density=100.0)

    def reset(self, seed: int = None, options: dict = None) -> (gym.core.ObsType, dict):
        """Reset the environment and return initial observation."""
        self.current_step = 0

        # Set initial camera pose
        self.simulator.set_world_from_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -6.0],
            [0.0, -1.0, 0.0, 4.5],
            [0.0, 0.0, 0.0, 1.0]
        ]), check_collisions=False)
        # self.simulator.respawn_camera()

        # Reset observed points
        self.simulator.observed_points_mask[:] = False

        # Get initial observed points
        self.simulator.get_point_cloud(update_mask=True)

        # Get observation
        image = self.render_image()

        return image, {}

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

        # Get observation
        image = self.render_image()
        self.simulator.get_point_cloud(update_mask=True)

        # Display rendered image
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

        return image, reward, terminated, False, {}

    def render_image(self):
        rgb, depth = self.simulator.render()
        rgb = np.ascontiguousarray(np.moveaxis(rgb, 2, 0) * 255).astype(np.uint8)
        return rgb


if __name__ == "__main__":
    env = SimulatorEnv(blend_file='/home/clementin/Data/blendernbv/liberty.blend')
    logger = logger.configure('logs', ['stdout', 'csv', 'tensorboard'])

    policy_kwargs = {
        'features_extractor_class': TimeSformerExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
    }
    model = PPO(
        'CnnPolicy',
        env,
        # policy_kwargs=policy_kwargs,
        verbose=1
    )
    model.set_logger(logger)
    model.learn(total_timesteps=100000)
    model.save('checkpoint.pt')
