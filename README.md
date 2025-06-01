# Buildee: A 3D Simulation Framework for Scene Exploration and Reconstruction with Understanding

[Paper](https://openreview.net/forum?id=1LmsiOaMTy) • [Project page](https://clementinboittiaux.github.io/buildee/)

This repository hosts [Buildee](https://openreview.net/forum?id=1LmsiOaMTy), a 3D simulation framework built as a
Python module on top of Blender. With Buildee, you can explore a 3D scene and generate realistic RGB, depth, and
semantic segmentation maps. Buildee is also equipped to track 2D / 3D points and perform occlusion checking.



## Installation

Clone the repository with submodules:
```bash
git clone --recursive https://github.com/clementinboittiaux/buildee.git
```

Setup the conda environment:
```bash
conda create -n buildee python=3.11
conda activate buildee
```

Install the numpy version required by blender:
```bash
pip install numpy==1.24.3
```

With the conda environment **activated**,
[build blender](https://developer.blender.org/docs/handbook/building_blender/linux/#__tabbed_2_2)
[as a python module](https://developer.blender.org/docs/handbook/building_blender/python_module/):
```bash
cd buildee/blender
./build_files/build_environment/install_linux_packages.py  # update dependencies
./build_files/utils/make_update.py --use-linux-libraries  # update dependencies
make bpy  # build bpy module
python3 ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/  # create wheel
```

Then, install blender's `bpy` python module:
```bash
pip3 install ../build_linux_bpy/bin/bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl
```

Install your compatible [PyTorch](https://pytorch.org/) version. Then install other requirements:
```bash
cd ..
pip install -r requirements.txt
```

Finally, install Buildee as a package:
```bash
pip install -e .
```

#### Troubleshooting
When installing on a headless server, remember to disable X11 forwarding.


## Usage

With Buildee, you can start your simulation in just a few lines of code. The following example shows how to:
1. Load a Blender scene.
2. Render RGB image, depth map, and segmentation map.
3. Compute the current point cloud of the scene (dynamic points positions may change over time).
4. Step forward the simulation to the next frame.

```py
from buildee import Simulator

sim = Simulator(
    blend_file='file.blend',
    points_density=10.0,
    verbose=True
)

rgb, depth, labels = sim.render()

pcl, pcl_labels, pcl_visibility_mask = sim.compute_point_cloud(imshow=True)

sim.step_frame()
```
