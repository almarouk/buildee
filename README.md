Clone the repository with submodules:
```bash
git clone --recursive https://github.com/clementinboittiaux/blender-construction.git
```

Setup the conda environment:
```bash
conda create -n blender python=3.11
conda activate blender
```

With the conda environment **activated**,
build blender [as a python module](https://developer.blender.org/docs/handbook/building_blender/linux/#__tabbed_2_2):
```bash
cd blender-construction/blender
make bpy -j12  # build bpy module
python3 ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/  # create wheel
pip install numpy==1.24.3  # numpy version required by blender
pip3 install ../build_linux_bpy/bin/bpy-4.4.0a0-cp311-cp311-manylinux_2_31_x86_64.whl  # install bpy
```
