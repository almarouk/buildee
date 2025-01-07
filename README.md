## Installation

Clone the repository with submodules:
```bash
git clone --recursive https://github.com/clementinboittiaux/blender-construction.git
```

Setup the conda environment:
```bash
conda create -n blender python=3.11
conda activate blender
pip install numpy==1.24.3  # numpy version required by blender
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
pip3 install ../build_linux_bpy/bin/bpy-4.4.0a0-cp311-cp311-manylinux_2_31_x86_64.whl  # install bpy
```

Install module requirements:
```bash
pip install -r requirements.txt
```

#### Troubleshooting
When installing on a headless server, remember to disable X11 forwarding.
