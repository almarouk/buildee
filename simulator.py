import tempfile
from collections import OrderedDict
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

        # Initialize point cloud geometry node
        self.pcl_node = bpy.data.node_groups.new(name="Pointcloud", type='GeometryNodeTree')
        pcl_input_node = self.pcl_node.nodes.new(type='NodeGroupInput')
        pcl_output_node = self.pcl_node.nodes.new(type='NodeGroupOutput')
        pcl_points_node = self.pcl_node.nodes.new(type="GeometryNodeDistributePointsOnFaces")
        pcl_vertices_node = self.pcl_node.nodes.new(type="GeometryNodePointsToVertices")
        pcl_realize_node = self.pcl_node.nodes.new(type="GeometryNodeRealizeInstances")
        self.pcl_node.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        self.pcl_node.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        self.pcl_node.links.new(pcl_input_node.outputs["Geometry"], pcl_points_node.inputs["Mesh"])
        self.pcl_node.links.new(pcl_points_node.outputs["Points"], pcl_vertices_node.inputs["Points"])
        self.pcl_node.links.new(pcl_vertices_node.outputs["Mesh"], pcl_realize_node.inputs["Geometry"])
        self.pcl_node.links.new(pcl_realize_node.outputs["Geometry"], pcl_output_node.inputs["Geometry"])

        # Get a point cloud representation of every object in the scene
        self.point_clouds = OrderedDict()
        for obj in self.scene.objects:
            if obj.type == 'MESH':
                obj_modifier = obj.modifiers.new('GeometryNodes', type='NODES')
                obj_modifier.node_group = self.pcl_node
                evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                obj_vertices = np.empty(len(evaluated_obj.data.vertices) * 3)
                evaluated_obj.data.vertices.foreach_get('co', obj_vertices)
                obj.modifiers.remove(obj_modifier)
                self.point_clouds[obj] = obj_vertices.reshape(-1, 3)

        # Store the total number of points in the point clouds
        self.n_points = sum(len(vertices) for vertices in self.point_clouds.values())

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
        self.set_world_from_camera(self.get_world_from_camera() @ camera_from_next_camera)

    def get_point_cloud(self) -> np.ndarray:
        # Store all the points in a single point cloud
        point_cloud = np.empty((self.n_points, 3))

        # Set current point cloud index
        point_cloud_idx = 0

        # Iterate over all objects and store their transformed vertices in the point cloud
        for obj, vertices in self.point_clouds.items():
            # Get the object pose
            world_from_obj = np.array(obj.matrix_world)

            # Transform the vertices to world coordinates
            world_vertices = world_from_obj[:3, :3] @ vertices.T + world_from_obj[:3, 3:]

            # Transform the vertices to world coordinates
            point_cloud[point_cloud_idx:point_cloud_idx + len(vertices)] = world_vertices.T

            # Update the point cloud index
            point_cloud_idx += len(vertices)

        return point_cloud

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

    simulator.get_point_cloud()

    depth_color_map = plt.get_cmap('magma')
    max_depth_distance_display = 10.0

    for _ in tqdm.tqdm(range(999999999)):

        key = cv2.waitKeyEx(7)

        if key == ord('q'):
            simulator.rotate_camera_yaw(-15)
        elif key == ord('d'):
            simulator.rotate_camera_yaw(15)
        elif key == ord('z'):
            simulator.move_camera_forward(1)
        elif key == ord('s'):
            simulator.move_camera_forward(-1)
        elif key == 27:  # escape key
            break

        rgb, depth = simulator.render()

        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
