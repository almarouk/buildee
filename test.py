import bpy
import tqdm
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Load blender file
bpy.ops.wm.open_mainfile(filepath='/home/clementin/Downloads/test.blend')

# Get the current scene
scene = bpy.context.scene

# Get os temp dir
temp_dir = Path(tempfile.gettempdir())

# Get rendered image path
render_path = temp_dir / 'render.exr'

# Render the depth in alpha channel
scene.use_nodes = True
scene.view_layers["ViewLayer"].use_pass_z = True
render_node = scene.node_tree.nodes["Render Layers"]
output_node = scene.node_tree.nodes["Composite"]
scene.node_tree.links.new(render_node.outputs['Depth'], output_node.inputs['Alpha'])

# Setup color space
color_node = scene.node_tree.nodes.new(type='CompositorNodeConvertColorSpace')
color_node.from_color_space = 'Linear Rec.709'
color_node.to_color_space = 'AgX Base sRGB'
scene.node_tree.links.new(render_node.outputs['Image'], color_node.inputs['Image'])
scene.node_tree.links.new(color_node.outputs['Image'], output_node.inputs['Image'])

# Set render settings for rgb output
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '32'
scene.render.image_settings.exr_codec = 'NONE'
scene.render.filepath = str(render_path)

for _ in tqdm.tqdm(range(100)):
    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Load the image
    image = bpy.data.images.load(str(render_path))
