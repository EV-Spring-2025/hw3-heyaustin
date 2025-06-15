[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SdXSjEmH)
# EV-HW3: PhysGaussian

This homework is based on the recent CVPR 2024 paper [PhysGaussian](https://github.com/XPandora/PhysGaussian/tree/main), which introduces a novel framework that integrates physical constraints into 3D Gaussian representations for modeling generative dynamics.

You are **not required** to implement training from scratch. Instead, your task is to set up the environment as specified in the official repository and run the simulation scripts to observe and analyze the results.


## Getting the Code from the Official PhysGaussian GitHub Repository
Download the official codebase using the following command:
```
git clone https://github.com/XPandora/PhysGaussian.git
```


## Environment Setup
Navigate to the "PhysGaussian" directory and follow the instructions under the "Python Environment" section in the official README to set up the environment.


## Running the Simulation
Follow the "Quick Start" section and execute the simulation scripts as instructed. Make sure to verify your outputs and understand the role of physics constraints in the generated dynamics.


## Homework Instructions
Please complete Part 1–2 as described in the [Google Slides](https://docs.google.com/presentation/d/13JcQC12pI8Wb9ZuaVV400HVZr9eUeZvf7gB7Le8FRV4/edit?usp=sharing).

### Part 1
In part 1, I chose `sand` and `metal` as the materials.
[All the videos are available in this YouTube playlist.](https://www.youtube.com/playlist?list=PLFPhkwxmp1ZEAvqXWw0zZDSikVp2KXeyo)

Parameter | base value
--- | ---
grid_v_damping_scale | 0.9999
n_grid | 30
softening | 0.1
substep_dt | 1e-4

Material | GIF
--- | ---
[sand_base](https://youtu.be/bFDuSG9lO_8) | <img src="./media/sand_base.gif" style="width:300px"/>
[metal_base](https://youtu.be/QwxMHnUCYyY) | <img src="./media/metal_base.gif" style="width:300px"/>

### Part 2
I conducted an ablation study on the material `metal` and compiled the table below.

Modified Param → New Value | Avg PSNR (dB) with Base | GIF
--- | --- | ---
[grid_v_damping_scale → 0.99](https://youtu.be/XpypHSgR3mM) | 16.05 | <img src="./media/metal_grid_v_damping_scale_0.99.gif" style="width:300px"/>
[grid_v_damping_scale → 0.999](https://youtu.be/4nt9kE96wkI) | 16.55 | <img src="./media/metal_grid_v_damping_scale_0.999.gif" style="width:300px"/>
[n_grid → 10](https://youtu.be/cNmvkuxVfII) | 16.37 | <img src="./media/metal_n_grid_10.gif" style="width:300px"/>
[n_grid → 20](https://youtu.be/1SETDgQHnEM) | 18.17 | <img src="./media/metal_n_grid_20.gif" style="width:300px"/>
[softening → 0.3](https://youtu.be/aElruDIEZVc) | 40.78 | <img src="./media/metal_softening_0.3.gif" style="width:300px"/>
[softening → 0.5](https://youtu.be/vobB20f56Jo) | 40.18 | <img src="./media/metal_softening_0.5.gif" style="width:300px"/>
[substep_dt → 1e-05](https://youtu.be/2zgkSzLlyDw) | 16.13 | <img src="./media/metal_substep_dt_1e-05.gif" style="width:300px"/>
[substep_dt → 5e-05](https://youtu.be/biUFhJCJde4) | 16.94 | <img src="./media/metal_substep_dt_5e-05.gif" style="width:300px"/>

Softening the material model has by far the biggest impact: raising `softening` from 0.1 to 0.3 boosts average PSNR by roughly 25 dB, eliminating particle ringing and explosions, while a further increase to 0.5 levels off with a similar gain. A moderate grid velocity damping of `grid_v_damping_scale = 0.999` strikes a good balance—only about 0.45 dB below the virtually lossless baseline 0.9999 yet faster to dissipate residual noise—whereas stronger damping at 0.99 erodes nearly a full decibel of detail. Reducing grid resolution from 30³ to 20³ cells costs only about 1 dB but almost halves computation, whereas 10³ introduces noticeable aliasing. Finally, halving the internal time step from 1 × 10⁻⁴ to 5 × 10⁻⁵ sec yields a modest 0.8 dB gain, and pushing to 1 × 10⁻⁵ brings negligible improvement. In practice, a balanced setting of `softening = 0.3`, `grid_v_damping_scale = 0.999`, `n_grid = 20`, and `substep_dt = 5 × 10⁻⁵` offers the best trade-off between visual fidelity and runtime.

### Bonus
To make PhysGaussian material-agnostic, embed a latent-conditioned neural constitutive model inside the differentiable MPM loop: a small MLP maps deformation gradients and a low-dimensional material code `z` to stresses, with physics-aware architectural constraints for stability. A vision-based or diffusion prior gives an initial `z` from a few input frames, and the code is then refined jointly with the 3-D Gaussians by back-propagating photometric, silhouette, and physics residual losses through the differentiable simulator. Pre-train the constitutive network and latent prior on a large synthetic material library, optionally fine-tune on a small real dataset, then perform a handful of gradient steps at inference to adapt `z`—and thus the material parameters—to any unseen substance.


# Reference
```bibtex
@inproceedings{xie2024physgaussian,
    title     = {Physgaussian: Physics-integrated 3d gaussians for generative dynamics},
    author    = {Xie, Tianyi and Zong, Zeshun and Qiu, Yuxing and Li, Xuan and Feng, Yutao and Yang, Yin and Jiang, Chenfanfu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2024}
}
```
