import numpy as np
import torch
import torchvision as tv

def resize_to(images: torch.Tensor, height: int, width: int):
    # images: (B, old_H, old_W, C)

    # (B, C, old_H, old_W)
    images = images.permute(0, 3, 1, 2)

    transform = tv.transforms.Compose([
        tv.transforms.Resize((height, width))
    ])

    # (B, C, new_H, new_W)
    resized_images = torch.stack([
        # (C, new_H, new_W)
        transform(image)
        for image in images
    ])

    # (B, new_H, new_W, C)
    resized_images = resized_images.permute(0, 2, 3, 1)

    return resized_images


data = np.load("tiny_nerf_data.npz")

images = data['images']
poses = data['poses']
focal_length = data['focal']

# Assume square images
orig_size = images.shape[1]
downscaled_size = 32

# (B, H, W, C)
images = torch.from_numpy(images)
images = resize_to(images, downscaled_size, downscaled_size)

# 4x4 camera transform matrices (poses) that transform a point
# from camera space to world space
# (B, 4, 4)
poses = torch.from_numpy(poses)

# (1,) (scalar)
focal_length = torch.from_numpy(focal_length)
focal_length = focal_length * downscaled_size / orig_size

# print(images.shape)
# print(poses.shape)
# print(focal_length)

height, width = images.shape[1:3]