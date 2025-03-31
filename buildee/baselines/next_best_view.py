import argparse
import os.path
from pathlib import Path

from ..simulator import Simulator

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
    def __init__(self, blend_file: Path, max_steps: int = 500, show_rgb: bool = False):
        super().__init__()
        self.blend_file = blend_file

        # Define maximum number of steps
        self.max_steps = max_steps
        self.current_step = 0

        # Define image observation space
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)

        # Define action space: xyz and yaw
        self.action_space = gym.spaces.Discrete(5)

        # Whether to display rendered image during training
        self.show_rgb = show_rgb

        # Initialize simulator, initial pose and previous observed points mask
        self.simulator = Simulator(self.blend_file, points_density=5.0, max_pcl_depth=20.0, verbose=True)
        self.init_world_from_camera = self.simulator.get_world_from_camera()
        self.prev_observed_points_mask = None

    def reset(self, seed: int = None, options: dict = None) -> (gym.core.ObsType, dict):
        # Reset camera pose
        self.simulator.set_world_from_camera(self.init_world_from_camera, check_collisions=False)

        # Reset observed point mask
        self.simulator.observed_points_mask = np.zeros(self.simulator.n_points, dtype=bool)

        # Reset current step
        self.current_step = 0

        # Render image and update point cloud
        image = self.render_image()
        self.simulator.compute_point_cloud(update_mask=True)

        # Setup previous observations and observed points mask
        self.prev_observed_points_mask = self.simulator.observed_points_mask.copy()

        return image, {}

    def step(
            self, action: gym.core.ActType
    ) -> (gym.core.ObsType, float, bool, bool, dict):
        # Update current step
        self.current_step += 1

        # Apply action to the buildee
        match action:
            case 0:
                collided = self.simulator.move_camera_forward(3)
            case 1:
                collided = self.simulator.move_camera_right(-3)
            case 2:
                collided = self.simulator.move_camera_right(-3)
            case 3:
                collided = self.simulator.turn_camera_right(22.5)
            case 4:
                collided = self.simulator.turn_camera_right(-22.5)
            case 5:
                collided = self.simulator.move_camera_forward(3)
                while collided:
                    random_turn = (np.random.rand() - 0.5) * 2 * 180
                    self.simulator.turn_camera_right(random_turn)
                    collided = self.simulator.move_camera_forward(3)
            case _:
                raise ValueError(f'Invalid action: {action}')

        # Render image and update point cloud
        image = self.render_image()
        self.simulator.compute_point_cloud(update_mask=True, imshow=True)

        # Display rendered image
        if self.show_rgb:
            cv2.imshow(f'rgb', cv2.cvtColor(np.moveaxis(image, 0, 2), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # Get reward and termination
        reward = self.compute_coverage_gain() * 100.0
        terminated = self.current_step >= self.max_steps

        return image, reward, terminated, False, {}

    def render_image(self):
        rgb, depth, _ = self.simulator.render()
        rgb = np.ascontiguousarray(np.moveaxis(rgb, 2, 0) * 255).astype(np.uint8)
        return rgb

    def compute_coverage_gain(self) -> float:
        new_observed_points_mask = self.simulator.observed_points_mask.copy()
        coverage_gain = np.mean(new_observed_points_mask & ~self.prev_observed_points_mask)
        self.prev_observed_points_mask = new_observed_points_mask
        return coverage_gain


def load_model(
        blend_file: Path,
        show_rgb: bool = False
) -> tuple[PPO, SimulatorEnv, CheckpointCallback]:
    # Parse checkpoint stem
    scene_name = blend_file.stem

    # Setup checkpoint path from scene name
    checkpoint_path = f'{scene_name}.zip'

    # Setup environment, logger, and checkpointer
    env = SimulatorEnv(blend_file=blend_file, show_rgb=show_rgb)
    lgr = logger.configure(f'logs/{scene_name}', ['stdout', 'csv', 'tensorboard'])
    checkpointer = CheckpointCallback(save_freq=1000, save_path=scene_name, verbose=1)

    # Load or create model
    if os.path.exists(checkpoint_path):
        print(f'Loading model from {checkpoint_path}')
        model = PPO.load(checkpoint_path, env=env, verbose=1)
    else:
        print(f'{checkpoint_path} not found, creating new model')
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1
        )

    # Set logger
    model.set_logger(lgr)

    return model, env, checkpointer


def main(
        blend_file: Path,
        train: bool = False,
        random_walk: bool = False,
        random_star: bool = False,
        show_rgb: bool = False
):
    # Load model and checkpointer
    model, env, checkpointer = load_model(
        blend_file=blend_file,
        show_rgb=show_rgb
    )

    if train:  # Train model
        model.learn(total_timesteps=40000000, callback=checkpointer)

    else:  # Evaluate model
        obs, _ = env.reset()
        coverages = [env.simulator.observed_points_mask.mean()]

        for _ in range(env.max_steps):

            if random_walk:
                action = np.array(np.random.randint(0, 5))
            elif random_star:
                action = np.array(5)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, _, _, _, _ = env.step(action)
            coverages.append(env.simulator.observed_points_mask.mean())

        scene_coverage_auc = np.trapz(coverages, np.linspace(0, 1, len(coverages)))
        print(f'Scene coverage AUC: {scene_coverage_auc}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explore a Blender 3D scene.')
    parser.add_argument('--blend-file', type=Path, required=True, help='path to the Blender file')
    parser.add_argument('--train', action='store_true', help='whether to train the model or evaluate it')
    parser_random_walk = parser.add_mutually_exclusive_group()
    parser_random_walk.add_argument('--random-walk', action='store_true',
                                    help='evaluate random walk instead of PPO model')
    parser_random_walk.add_argument('--random-star', action='store_true',
                                    help='evaluate random walk* instead of PPO model')
    parser.add_argument('--show-rgb', action='store_true', help='whether to display the rendered image')
    args = parser.parse_args()
    main(
        blend_file=args.blend_file,
        train=args.train,
        random_walk=args.random_walk,
        random_star=args.random_star,
        show_rgb=args.show_rgb
    )
