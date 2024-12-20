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

```bash
1151  git clone https://projects.blender.org/blender/blender.git
 1152  ls
 1153  git clone https://projects.blender.org/blender/blender.git --depth 1
 1154  cd blender
 1155  sudo apt update
 1156  sudo apt install build-essential git git-lfs subversion cmake libx11-dev libxxf86vm-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev libegl-dev
 1157  sudo apt install libwayland-dev wayland-protocols libxkbcommon-dev libdbus-1-dev linux-libc-dev
 1158  rm -r build_linux_bpy/
 1159  cd blender/
 1160  make bpy -j12
 1161  cd ../build_linux_bpy/
 1162  ls
 1163  cd bin
 1164  ls
 1165  conda create -n blender python=3.11
 1166  conda activate blender
 1167  cd ../..
 1168  ls
 1169  cd blender
 1170  ls
 1171  python3 ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin
 1172  cd ../build_linux_bpy/bin/
 1173  ls
 1174  pip3 install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1175  python
 1176  pip install pkginfo
 1177  pkginfo -f requires_dist bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1178  conda deactivate
 1179  conda remove -n blender --all
 1180  conda create -n blender python=3.11
 1181  conda activate blender
 1182  pip install numpy==1.24.3
 1183  python
 1184  pip install Cython==0.29.30
 1185  pip install requests==2.27.1
 1186  pip install zstandard==0.16.0
 1187  pip3 install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1188  python
 1189  sudo swapon /swapfile
 1190  htop
 1191  ./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m ./output/gaussian_scale2
 1192  blender
 1193  conda activate blender
 1194  pip install matplotlib
 1195  pip install --force-reinstall -v "matplotlib==2.1.2"
 1196  conda deactivate
 1197  conda remove -n blender --all
 1198  conda create -n blender python=3.11
 1199  conda activate blender
 1200  pip install numpy==1.24.3
 1201  pip install Cython==0.29.30
 1202  pip install requests==2.27.1
 1203  pip install zstandard==0.16.0
 1204  cd Dev/build_linux_bpy/bin/
 1205  ls
 1206  pip3 install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1207  pip install opencv-python
 1208  blender
 1209  python
 1210  python3
 1211  sudo swapon /swapfile
 1212  conda create -n blender2 python=3.11
 1213  conda activate blender2
 1214  ls
 1215  cd Dev/build_linux_bpy/
 1216  cd bin
 1217  pip3 install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1218  python
 1219  conda deactivate
 1220  conda remove -n blender2 --all
 1221  conda create -n blender2 python=3.11
 1222  conda activate blender2
 1223  pip install numpy Cython requests zstandard
 1224  ls
 1225  cd ../..
 1226  cd blender
 1227  ls
 1228  make bpy
 1229  ls
 1230  make bpy -j12
 1231  python ./build_files/utils/make_bpy_wheel.py ../build_linux_bpy/bin/
 1232  cd ../build_linux_bpy/bin/
 1233  pip install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1234  python
 1235  cd ../..
 1236  conda deactivate
 1237  conda remove -n blender2 --all
 1238  rm -rf build_linux_bpy
 1239  ls
 1240  cd Dev
 1241  ls
 1242  git clone git@github.com:clementinboittiaux/blender-construction.git
 1243  cd blender-construction/
 1244  ls
 1245  conda activate gaussian_splatting
 1246  python
 1247  conda create -n blender2 python=3.11
 1248  conda activate blender2
 1249  conda install numpy=1.24.3
 1250  conda install conda-forge::zstandard=0.16.0
 1251  conda install anaconda::requests=2.27.1
 1252  conda install conda-forge::cython=0.29.30
 1253  conda install conda-forge::cython
 1254  conda install conda-forge::zstandard
 1255  cd Dev/build_linux_bpy/bin/
 1256  ls
 1257  pip install bpy-4.4.0a0-cp311-cp311-manylinux_2_35_x86_64.whl 
 1258  python
 1259  conda install conda-forge::matplotlib
 1260  python
 1261  conda activate blender2
 1262  python
 1263  conda remove -n blender --all
 1264  conda rename -n blender2 blender
 1265  nvcc -V
 1266  cd Dev/blender-construction/
 1267  ls
 1268  git mv test.py simulator.py
 1269  conda activate blender
 1270  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
 1271  python
 1272  conda activate blender
 1273  pip install pycolmap
 1274  colmap gui
 1275  conda activate blender
 1276  pip install mathutils
 1277  colmap gui
 1278  conda activate blender
 1279  conda install anaconda::scipy 
 1280  python
 1281  conda activate blender
 1282  conda install conda-forge::opencv
 1283  pip install opencv-python
 1284  blender
```