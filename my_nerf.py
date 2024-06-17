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

    #      (H, W, N, 3)  (*, N)
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


# ==================================

def positional_encoding(
    # (*, D (3))
    points,
    L=6,
) -> torch.Tensor:
    encoding = [points]

    freqs = 2.0 ** torch.linspace(0.0, L - 1, L)

    for freq in freqs:
        encoding.append(torch.sin(points * freq))
        encoding.append(torch.cos(points * freq))

    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def split_points_into_chunks(
    # (B, L)
    points: torch.Tensor,
    chunk_size: int
):
    return [
        points[i:i + chunk_size]
        for i in range(0, points.shape[0], chunk_size)
    ]

def nf_render_pose(
    model: torch.nn.Module,
    height: int,
    width: int,
    focal_length: int,
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

    # (H*W*N, 3)
    flat_query_points = query_points.view(-1, 3)

    # apply positional encoding
    flat_query_points = positional_encoding(flat_query_points)

    # convert flat_query_points to chunks
    chunks = split_points_into_chunks(
        flat_query_points, chunk_size)
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


# ================================

class VeryTinyNerfModel(torch.nn.Module):
    def __init__(
        self,
        filter_size=128,
        num_encoding_functions=6
    ):

        super(VeryTinyNerfModel, self).__init__()
        # Input layer (default: 39 -> 128)
        self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = torch.nn.Linear(filter_size, 4)
        # Short hand for torch.nn.functional.relu
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = VeryTinyNerfModel()

# rgb_map = nf_render_pose(
#     model,
#     height,
#     width,
#     focal_length,
#     pose=poses[0],
#     thresh_near=2,
#     thresh_far=6,
#     num_samples_per_ray=32,
#     chunk_size=8096,
# )

def predict(pose: tensor.Tensor):
    return nf_render_pose(
        model,
        height,
        width,
        focal_length,
        pose=pose,
        thresh_near=2,
        thresh_far=6,
        num_samples_per_ray=32,
        chunk_size=8096,
    )

# ================================
# =========== Training ===========
# ================================

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

target_img_idx = torch.randint(images.shape[0])
target_img = images[target_img_idx]
target_pose = poses[target_img_idx]

# Run one iteration of TinyNeRF and get the rendered RGB image.
rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                       target_tform_cam2world, near_thresh,
                                       far_thresh, depth_samples_per_ray,
                                       encode, get_minibatches)

loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
optimizer.zero_grad()
loss.backward()
optimizer.step()


print('Done!')