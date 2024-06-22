# Neural Radiance Fields (NeRF) Implementation

This project is a Python implementation of [**Neural Radiance Fields (NeRF)**](https://arxiv.org/abs/2003.08934), a powerful technique for photorealistic 3D scene reconstruction using deep learning. NeRF represents 3D scenes as a continuous volumetric representation parameterized by a neural network and allows the rendering of novel views from a sparse set of input images.


## Overview
This implementation builds the foundational components of NeRF, including ray marching, volumetric rendering, positional encoding, and a multi-headed neural network for predicting radiance fields. The model is trained to learn a scene representation from input images and corresponding camera poses.

Key functionalities include:
- **Ray casting and query point generation**: Simulates rays passing through each pixel to sample the 3D space.
- **View synthesis**: Renders new views by querying learned radiance fields.
- **Flexible model design**: Offers two architectures, `SimpleNeRF` and `MultiHeadedNeRF`, for experimentation.

### Dataset

The dataset used contains:
- Input images of the scene.
- Camera poses (4x4 transformation matrices) for each image.
- Focal length, image height, and width.

- **Images**: Shape `(Batch, Height, Width, 3)`
- **Poses**: Shape `(Batch, 4, 4)`
- **Focal Length**: Scalar

## Model Architecture

### `SimpleNeRF`
A straightforward fully connected network with three hidden layers. It concatenates positional and directional encodings as input and predicts RGB and density.

### `MultiHeadedNeRF`
A modular design with separate networks for density prediction (`sigma`) , followed by a final color prediction module. This architecture uses residual connections and supports multi-view consistency.

## Training Process
1. **Input Sampling**: Randomly selects an image and its corresponding pose.
2. **Prediction**: Casts rays from the pose, queries the model, and renders the view.
3. **Loss Calculation**: Computes Mean Squared Error (MSE) between predicted and target images.
4. **Optimization**: Backpropagates the loss and updates model parameters.


## Results
- The implementation can render photorealistic novel views of a scene given enough training data and iterations.
- Performance scales with the number of samples per ray and hidden layer size.


## References
- **Original NeRF Paper**: [https://arxiv.org/abs/2003.08934](https://arxiv.org/abs/2003.08934)
