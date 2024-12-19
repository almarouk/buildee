import torch
import gymnasium as gym
from timesformer.models.vit import TimeSformer

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TimeSformerExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.timesformer = TimeSformer(
            img_size=224,
            num_classes=400,
            num_frames=8,
            attention_type='divided_space_time',
            pretrained_model='TimeSformer_divST_8x32_224_K600.pyth'
        )

        # Replace the head with a new one
        self.timesformer.model.head = torch.nn.Linear(in_features=768, out_features=features_dim, bias=True)

        # GELU activation
        self.gelu = torch.nn.GELU(approximate='none')

        # Disable grad for all layers except for the last layer
        # for param_name, param in self.timesformer.named_parameters():
        #     if 'model.head' in param_name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.timesformer(observations[:, :, None].repeat(1, 1, 8, 1, 1)))
