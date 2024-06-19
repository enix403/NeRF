import torch
from torch import nn
import torch.nn.functional as F

def nf_get_ray_bundle(
    height: int,
    width: int,
    focal_length: torch.Tensor,
    # 4x4 transformation matrix
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
    # Camera faces the -ve Z direction in NDC
    points_z = -torch.ones_like(points_x)

    ray_dirs = torch.stack(
        (
            points_x,
            points_y,
            points_z,
        ),
        dim=-1
    )

    ray_dirs = F.normalize(ray_dirs, dim=-1)

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

    #      (H, W, N, 3)  (*, N)
    return query_points, depths


def nf_render_view_field(
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
    deltas = depths[..., 1:] - depths[..., :-1]

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
    alpha = 1. - torch.exp(-sigma_field * deltas)
    # (H, W, N)
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # (H, W, N, 3)
    rgb_map_points = (
      # (H, W, N, 1)
      weights[..., None]
      *
      # (H, W, N, 3)
      rgb_field
    )

    # (H, W, 3)
    rgb_map = rgb_map_points.sum(dim=-2)

    return rgb_map

def nf_render_pose(
    model: torch.nn.Module,
    height: int,
    width: int,
    focal_length: torch.Tensor,
    pose: torch.Tensor,
    thresh_near: int,
    thresh_far: int,
    num_samples_per_ray: int,
    chunk_size: int,
):

    # Create rays
    ray_origins, ray_dirs = nf_get_ray_bundle(
        height,
        width,
        focal_length,
        pose
    )

    # Create query points
    query_points, depths = nf_create_query_points(
        ray_origins,
        ray_dirs,
        thresh_near,
        thresh_far,
        num_samples_per_ray,
    )

    # pass query points to model
    """
    model: (B, 3) -> (B, 4)
    """

    # (H, W, N, 3)
    # query_points

    # ============ create input ============

    # (H*W*N, 3)
    flat_query_points = query_points.view(-1, 3)

    # (H, W, N, 3)
    rd_per_point = ray_dirs[..., None, :].expand(query_points.shape)
    # (H*W*N, 3)
    flat_rd_per_point = rd_per_point.reshape(-1, 3)

    # (H*W*N, 6)
    flat_inputs = torch.cat([flat_query_points, flat_rd_per_point], dim=-1)

    # ============ call model  ============

    # convert flat_inputs to chunks
    chunks = split_points_into_chunks(flat_inputs, chunk_size)
    outputs = []

    for chunk in chunks:
        # (Bi, 4)
        chunk_view_field = model(chunk)
        outputs.append(chunk_view_field)

    # (H*W*N, 4)
    flat_view_field = torch.cat(outputs, dim=0)

    # create view (radiance field)
    # (H, W, N, 4)
    view_field = flat_view_field.view(
        list(query_points.shape[:-1]) + [-1]
    )

    rgb_map = nf_render_view(
        view_field,
        depths   
    )

    return rgb_map

# =============== Utils ===============

def split_points_into_chunks(
    # (B, L)
    points: torch.Tensor,
    chunk_size: int
):
    return [
        points[i:i + chunk_size]
        for i in range(0, points.shape[0], chunk_size)
    ]