from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    # size of the hidden layer
    hidden_size: int = 128
    # embedding size (L) for position (x)
    embed_num_pos: int = 6
    # embedding size (L) for view direction (d)
    embed_num_dir: int = 6

def embed_len_3d(embed_num: int):
    return 3 + 3 * 2 * embed_num

def sinusoidal_encoding(
    # (*, D (3))
    points: torch.Tensor,
    embed_num: int=6,
):
    encoding = [points]

    freqs = 2.0 ** torch.linspace(0.0, embed_num - 1, embed_num)

    for freq in freqs:
        encoding.append(torch.sin(points * freq))
        encoding.append(torch.cos(points * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


class VeryTinyNerfModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        hidden_size = config.hidden_size

        self.layers = nn.Sequential(
            # Layer 1: input (both pos and view dir)
            nn.Linear(
                embed_len_3d(config.embed_num_pos) + embed_len_3d(config.embed_num_dir),
                hidden_size
            ),
            nn.ReLU(),
            # Hidden layers
            torch.nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # Output layer (colors + density)
            torch.nn.Linear(hidden_size, 4),
        )

    def forward(self, queries):
        # query: (B, 6 (3 for pos and 3 for dir))

        # (B, 3)
        points = queries[..., :3]
        # (B, 3)
        viewdirs = queries[..., 3:]

        encoded_points = sinusoidal_encoding(points, self.config.embed_num_pos)
        encoded_viewdirs = sinusoidal_encoding(viewdirs, self.config.embed_num_dir)

        x = torch.cat([encoded_points, encoded_viewdirs], dim=-1)

        return self.layers(x)


class SeparatedHeadsNerfModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        hidden_size = config.hidden_size

        self.block1 = nn.Sequential(
            # Position Input (x) -> hidden_size
            nn.Linear(embed_len_3d(config.embed_num_pos), hidden_size), nn.ReLU(),
            # hidden_size -> hidden_size
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )

        # hidden_size -> 1 (sigma output)
        self.output_sigma = nn.Linear(hidden_size, 1)

        self.output_feature = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )

        self.output_color = nn.Sequential(
            nn.Linear(hidden_size + embed_len_3d(config.embed_num_dir), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )


    def forward(self, queries):
        # query: (B, 6 (3 for pos and 3 for dir))

        # (B, 3)
        points = queries[..., :3]
        # (B, 3)
        viewdirs = queries[..., 3:]

        # (B, Lx)
        x_pos = sinusoidal_encoding(points, self.config.embed_num_pos)
        # (B, Ld)
        x_dir = sinusoidal_encoding(viewdirs, self.config.embed_num_dir)

        # (B, H)
        act = self.block1(x_pos)
        
        # (B, 1)
        sigma = self.output_sigma(act)

        # (B, H)
        feature = self.output_feature(act)

        # (B, H + Ld)
        feature = torch.cat([feature, x_dir], dim=-1)

        # (B, 3)
        color = self.output_color(feature)

        # (B, 4)
        output = torch.cat([color, sigma], dim=-1)

        return output


