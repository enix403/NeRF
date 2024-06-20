import torch

def mat_rotation(theta, phi):
    theta = torch.tensor(theta) * (torch.pi / 180.0)
    phi = torch.tensor(phi) * (torch.pi / 180.0)

    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta), torch.cos(theta)]])

    R_y = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                        [0, 1, 0],
                        [-torch.sin(phi), 0, torch.cos(phi)]])

    return R_y @ R_x

def create_pose(radius=4, theta=0, phi=0):
    cam_rot = mat_rotation(theta, phi)
    cam_backwards = cam_rot[:, -1]
    cam_pos = radius * cam_backwards

    pose = torch.eye(4)
    pose[:3, :3] = cam_rot
    pose[:3, -1] = cam_pos

    return pose

def random_spherical_pose(radius=4, theta=None, phi=None):
    if theta is None:
        theta = (torch.rand(1) * 360).item()
    if phi is None:
        phi = (torch.rand(1) * 360).item()

    print(f"{theta=}, {phi=}")

    return create_pose(radius=radius, theta=theta, phi=phi)

    