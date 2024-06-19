import torch
import torch.nn.functional as F

from .arch import *
from .rendering import nf_render_pose
from .data_source import *

model = VeryTinyNerfModel(config=ModelConfig(
    hidden_size=128,
    embed_num_pos=6,
    embed_num_dir=6,
))

def predict(pose: torch.Tensor):
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

if __name__ == '__main__':

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Train
    for i in range(1000):
        # idx = torch.randint(images.shape[0], (1,)).item()
        idx = i % images.shape[0]
        target_pose = poses[idx]
        # (H, W, 3)
        target_image = images[idx]
        
        # (H, W, 3)
        image_predicted = predict(target_pose)

        loss = F.mse_loss(image_predicted, target_image)

        if i % 100 == 0:
            print(f"{i}: {loss.item()}") 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



"""
Todo list

multiheaded nerf
make network bigger
randomize query points
find a useful 3d dataset
create 2d images from dataset
convert radiance field to 3d point cloud

implement hierarchical sampling ?
"""