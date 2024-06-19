import torch

def generate_random_angles():
    # Generate random theta between -90 and 90 degrees (pitch)
    theta = torch.rand(1) * 180 - 90  # Random value between -90 and 90
    
    # Generate random phi between 0 and 360 degrees (yaw)
    phi = torch.rand(1) * 360  # Random value between 0 and 360

    return theta.item(), phi.item()


def rot(theta, phi):
    theta = torch.tensor(theta) * (torch.pi / 180)
    phi = torch.tensor(phi) * (torch.pi / 180)

    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta), -torch.sin(theta)],
                        [0, torch.sin(theta), torch.cos(theta)]])

    R_y = torch.tensor([[torch.cos(phi), 0, torch.sin(phi)],
                        [0, 1, 0],
                        [-torch.sin(phi), 0, torch.cos(phi)]])

    return torch.matmul(R_y, R_x)


def random_pose():
    theta, phi = generate_random_angles()
    print(f"{theta=}, {phi=}")

    radius = 4
    cam_rot = rot(theta, phi)
    cam_backwards = cam_rot[:, -1]
    cam_pos = radius * cam_backwards

    fakepose = torch.eye(4)
    fakepose[:3, :3] = cam_rot
    fakepose[:3, -1] = cam_pos

    return fakepose