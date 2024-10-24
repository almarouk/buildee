import tempfile
from pathlib import Path

import cv2
import tqdm
import bpy
import mathutils
import matplotlib.pyplot as plt
import numpy as np


class Simulator:
    def __init__(self, blend_file: str | Path):
        # Load blender file
        bpy.ops.wm.open_mainfile(filepath=str(blend_file))

        # Get the current scene
        self.scene = bpy.context.scene

        # Get the current view layer
        self.view_layer = bpy.context.view_layer

        # Get the current camera
        self.camera = self.scene.camera

        # Get os temp dir
        temp_dir = Path(tempfile.gettempdir())

        # Get rendered image path
        self.render_path = temp_dir / 'render.exr'

        # Render the depth in alpha channel
        self.scene.use_nodes = True
        self.scene.view_layers["ViewLayer"].use_pass_z = True
        self.render_node = self.scene.node_tree.nodes["Render Layers"]
        self.output_node = self.scene.node_tree.nodes["Composite"]
        self.scene.node_tree.links.new(self.render_node.outputs['Depth'], self.output_node.inputs['Alpha'])

        # Setup color space
        self.color_node = self.scene.node_tree.nodes.new(type='CompositorNodeConvertColorSpace')
        self.color_node.from_color_space = 'Linear Rec.709'
        self.color_node.to_color_space = 'AgX Base sRGB'
        self.scene.node_tree.links.new(self.render_node.outputs['Image'], self.color_node.inputs['Image'])
        self.scene.node_tree.links.new(self.color_node.outputs['Image'], self.output_node.inputs['Image'])

        # Set render settings for rgb output
        self.scene.render.image_settings.file_format = 'OPEN_EXR'
        self.scene.render.image_settings.color_mode = 'RGBA'
        self.scene.render.image_settings.color_depth = '32'
        self.scene.render.image_settings.exr_codec = 'NONE'
        self.scene.render.filepath = str(self.render_path)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        # Render the scene
        bpy.ops.render.render(write_still=True)

        # Load the image
        render = bpy.data.images.load(str(self.render_path))

        # Read image data
        rgbd = np.empty(len(render.pixels), dtype=np.float32)
        render.pixels.foreach_get(rgbd)
        rgbd = np.flip(rgbd.reshape(
            self.scene.render.resolution_y, self.scene.render.resolution_x, 4
        ), axis=0)

        return rgbd[:, :, :3], rgbd[:, :, 3]

    def set_world_from_camera(self, world_from_camera: np.ndarray, check_collisions: bool = True):
        # Check that there is no collision between current camera and new camera pose
        if check_collisions:
            origin = self.camera.location
            direction = world_from_camera[:3, 3] - np.array(origin)
            distance = float(np.linalg.norm(direction))

            if distance > 1e-8:  # do not move camera if we check for collisions but the distance is too small
                collided, collision_location, _, _, _, _ = self.scene.ray_cast(
                    depsgraph=self.view_layer.depsgraph,
                    origin=origin,
                    direction=mathutils.Vector(direction / distance),
                    distance=distance
                )
                if collided:
                    print('Collision detected, not moving the camera.')
                    return

        # Copy the camera pose
        world_from_camera = world_from_camera.copy()

        # Rotate the camera 180 degrees around the x-axis to fit blender's camera coordinate system
        world_from_camera[:3, 1:3] *= -1  # same as doing world_from_camera @ R(180, x)

        # Set the camera pose
        self.camera.matrix_world = mathutils.Matrix(world_from_camera)

    def get_world_from_camera(self) -> np.ndarray:
        # Read blender camera pose
        world_from_camera = np.array(self.camera.matrix_world)

        # Rotate the camera 180 degrees around the x-axis to fit blender's camera coordinate system
        world_from_camera[:, 1:3] *= -1  # same as doing world_from_camera @ R(180, x)

        return world_from_camera

    def set_camera_from_next_camera(self, camera_from_next_camera: np.ndarray):
        world_from_next_camera = self.get_world_from_camera() @ camera_from_next_camera
        self.set_world_from_camera(world_from_next_camera)

    def move_camera_forward(self, distance: float):
        # Move camera along the z-axis
        self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def rotate_camera_yaw(self, angle: float, degrees: bool = True):
        # Convert angle to radians if needed
        if degrees:
            angle = np.radians(angle)

        # Rotate camera around the y-axis
        self.set_camera_from_next_camera(np.array([
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))


if __name__ == '__main__':
    # Create a simulator
    simulator = Simulator('test.blend')

    depth_color_map = plt.get_cmap('magma')
    max_depth_distance_display = 10.0

    for _ in tqdm.tqdm(range(999999999)):

        key = cv2.waitKeyEx(7)

        if key == ord('q'):
            simulator.rotate_camera_yaw(-4)
        elif key == ord('d'):
            simulator.rotate_camera_yaw(4)
        elif key == ord('z'):
            simulator.move_camera_forward(0.3)
        elif key == ord('s'):
            simulator.move_camera_forward(-0.3)
        elif key == 27:  # escape key
            break

        rgb, depth = simulator.render()

        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
