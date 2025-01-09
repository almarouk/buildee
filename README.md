## Installation

Clone the repository with submodules:
```bash
git clone --recursive https://github.com/clementinboittiaux/blender-construction.git
```

Setup the conda environment:
```bash
conda create -n blender python=3.11
conda activate blender
```

Install the numpy version required by blender:
```bash
pip install numpy==1.24.3
```

With the conda environment **activated**,
[build blender](https://developer.blender.org/docs/handbook/building_blender/linux/#__tabbed_2_2)
[as a python module](https://developer.blender.org/docs/handbook/building_blender/python_module/):
```bash
cd blender-construction/blender
./build_files/build_environment/install_linux_packages.py  # update dependencies
./build_files/utils/make_update.py --use-linux-libraries  # update dependencies
make bpy  # build bpy module
python3 ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/  # create wheel
```

Then, install blender's `bpy` python module:
```bash
pip3 install ../build_linux_bpy/bin/bpy-4.4.0a0-cp311-cp311-manylinux_2_31_x86_64.whl
```

Install your compatible [PyTorch](https://pytorch.org/) version. Then install other requirements:
```bash
pip install -r requirements.txt
```

#### Troubleshooting
When installing on a headless server, remember to disable X11 forwarding.


## Training

We provide a toy reinforcement learning application, showing how to use our blender wrapper to train a next-best-view
agent.

### Additional dependencies
Training this agent requires TimeSformer, an efficient video model that can be easily installed from the
[repository](https://github.com/facebookresearch/TimeSformer).

### Pre-trained weights
You can download pre-trained TimeSformer weights from the
[TimeSformer repository](https://github.com/facebookresearch/TimeSformer).
We use [these weights](https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=0),
pre-trained on the K600 dataset using 8 frames. 
