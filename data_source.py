def resize_to(images, height, width):
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


images = data['images']
poses = data['poses']
focal_length = data['focal']

# Assume square images
orig_size = images.shape[1]
downscaled_size = 32

# Images
# (B, H, W, C)
images = torch.from_numpy(images)
images = resize_to(images, downscaled_size, downscaled_size)
# Camera extrinsics (poses)
poses = torch.from_numpy(poses)
# Focal length (intrinsics)
focal_length = torch.from_numpy(focal_length)
# Rescale focal length
focal_length = focal_length * downscaled_size / orig_size

# print(images.shape)
# print(poses.shape)
# print(focal_length)

height, width = images.shape[1:3]