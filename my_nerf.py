def resize_to(images, height, width):

    # Convert images to (B, 3, old_H, old_W)
    images = images.permute(0, 3, 1, 2)

    # Define the transform
    resize_transform = tv.transforms.Compose([
        tv.transforms.Resize((height, width))  # Resize to 32x32
    ])

    # Apply the transform to each image in the batch
    resized_images = torch.stack([
        resize_transform(image)
        for image in images
    ])

    # Convert back to (B, 32, 32, 3)
    resized_images = resized_images.permute(0, 2, 3, 1)

    return resized_images


def nf_get_ray_bundle(
    height: int,
    width: int,
    focal_length: torch.Tensor,
    pose: torch.Tensor
):
    points_x, points_y = torch.meshgrid(
        torch.arange(width),
        torch.arange(height),
        indexing='xy'
    )

    points_x = (points_x - width / 2.0) / focal_length
    # Note the -ve here, y in grid increases downwards while
    # y in NDC increases upwards
    points_y = -(points_y - height / 2.0) / focal_length
    points_z = -tr.ones_like(points_x)

    ray_dirs = tr.stack(
        (
            points_x,
            points_y,
            points_z,
        ),
        dim=-1
    )

    transform_rot = pose[:3, :3]
    ray_dirs = ray_dirs @ transform_rot.T

    ray_origins = pose[:3, -1].expand(ray_dirs.shape)

    return ray_origins, ray_dirs


def nf_create_query_points(
    # (H, W, 3)
    ray_origins: torch.Tensor,
    # (H, W, 3)
    ray_dirs: torch.Tensor,
    thresh_near: float,
    thresh_far: float,
    num_samples_per_ray: int,
):
    # TODO: randomize

    # (N,)
    depths = torch.linspace(thresh_near, thresh_far, num_samples_per_ray)

    # (H, W, N, 3)
    query_points = (
        ray_origins[..., None, :]
        + ray_dirs[..., None, :] * depths[:, None]
    )

    return query_points, depths


def nf_render_view(
    # (H, W, N, 4)
    view_field: torch.Tensor,
    # (N,) or (H, W, N)
    depths: torch.Tensor,
):
    # (H, W, N, 3)
    rgb_field = view_field[..., :3]
    # (H, W, N)
    sigma_field = view_field[..., 3]

    rgb_field = F.sigmoid(rgb_field)
    sigma_field = F.relu(sigma_field)

    # (*, N - 1)
    deltas = depths[..., 1:] - depths[..., :1]

    # (*, N)
    deltas = torch.cat(
        (
            # (*, N - 1)
            deltas,
            # (*, 1)
            torch.tensor([1e10]).expand(deltas[..., :1].shape)
        ),
        dim=-1
    )

    # (H, W, N)
    deltas = deltas.reshape(sigma_field.shape)

    # (H, W, N)
    alpha = 1. - torch.exp(-sigma_field * deltas)
    # (H, W, N)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # (H, W, N, 3)
    rgb_map_points = (
      # (H, W, N, 1)
      weights[..., None]
      *
      # (H, W, N, 3)
      rgb
    )

    # (H, W, 3)
    rgb_map = rgb_map_points.sum(dim=-2)

    return rgb_map

