import bpy
import tqdm
import tempfile
import numpy as np
from pathlib import Path


# Load blender file
bpy.ops.wm.open_mainfile(filepath='/home/clementin/Downloads/test2.blend')

# Get the current scene
scene = bpy.context.scene

# Get os temp dir
temp_dir = Path(tempfile.gettempdir())

# Get temp rgb and depth paths
rgb_path = temp_dir / 'rgb.exr'
depth_path = temp_dir / 'depth.exr'

# Set render settings for rgb output
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.exr_codec = 'NONE'
scene.render.filepath = str(rgb_path)

# Set render settings for rgb output
depth_node = scene.node_tree.nodes["Depth path"]
depth_node.base_path = str(depth_path.parent)
depth_node.file_slots[0].path = str(depth_path.stem)

for _ in tqdm.tqdm(range(100)):
    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Load the image
    image = bpy.data.images.load(str(rgb_path))

    # Load depth
    depth = bpy.data.images.load(
        str(depth_path.parent / f'{depth_path.stem}{scene.frame_current:04d}{depth_path.suffix}')
    )
