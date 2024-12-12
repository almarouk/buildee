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

        # Get camera matrix
        self.camera_matrix = self.get_camera_matrix()

        # Get a point cloud representation of every object in the scene
        self.object_point_clouds = self.setup_object_point_clouds()
        # Store the total number of points in the point clouds
        self.n_points = sum(len(vertices) for vertices in self.object_point_clouds.values())
        # Setup colors for rendering the point cloud
        self.point_cloud_colors = np.random.randint(0, 256, (self.n_points, 3), dtype=np.uint8)

        pcl, _ = self.get_point_cloud()
        self.point_cloud_colors[np.abs(pcl[:, 0] - 1.0) < 1e-6] = (255, 0, 0)
        self.point_cloud_colors[np.abs(pcl[:, 0] + 1.0) < 1e-6] = (0, 255, 0)
        self.point_cloud_colors[np.abs(pcl[:, 1] - 1.0) < 1e-6] = (0, 0, 255)
        self.point_cloud_colors[np.abs(pcl[:, 1] + 1.0) < 1e-6] = (255, 100, 255)

    def setup_object_point_clouds(self) -> OrderedDict[bpy.types.Object, np.ndarray]:
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
        object_point_clouds = OrderedDict()
        for obj in self.scene.objects:
            if obj.type == 'MESH':
                obj_modifier = obj.modifiers.new('GeometryNodes', type='NODES')
                obj_modifier.node_group = self.pcl_node
                evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
                obj_vertices = np.empty(len(evaluated_obj.data.vertices) * 3)
                evaluated_obj.data.vertices.foreach_get('co', obj_vertices)
                obj.modifiers.remove(obj_modifier)
                object_point_clouds[obj] = obj_vertices.reshape(-1, 3)

        return object_point_clouds

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

    def get_camera_matrix(self) -> np.ndarray:
        image_width = self.scene.render.resolution_x
        image_height = self.scene.render.resolution_y
        fixed_size = image_height if self.camera.data.sensor_fit == 'VERTICAL' else image_width
        f = fixed_size / np.tan(self.camera.data.angle / 2) / 2
        cx = image_width / 2
        cy = image_height / 2
        return np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])

    def set_world_from_camera(self, world_from_camera: np.ndarray, check_collisions: bool = True):
        # Check that there is no collision between current camera and new camera pose
        if check_collisions:
            origin = self.camera.location
            direction = mathutils.Vector(world_from_camera[:3, 3]) - origin

            if direction.length > 1e-8:  # do not move camera if we check for collisions but the distance is too small
                collided, collision_location, _, _, _, _ = self.scene.ray_cast(
                    depsgraph=self.view_layer.depsgraph,
                    origin=origin,
                    direction=direction.normalized(),
                    distance=direction.length + 1e-3  # add a small collision offset
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

    def get_point_cloud(self, imshow: bool = False) -> (np.ndarray, np.ndarray):
        # Store all the points in a single point cloud
        point_cloud = np.empty((self.n_points, 3))

        # Set all points as visible
        mask = np.ones(self.n_points, dtype=bool)

        # Get camera center
        origin = self.camera.location

        # Get image width and height
        image_width = self.scene.render.resolution_x
        image_height = self.scene.render.resolution_y

        # Set current point cloud index
        point_cloud_idx = 0

        # Iterate over all objects and store their transformed vertices in the point cloud
        for obj, vertices in self.object_point_clouds.items():
            # Get the object pose
            world_from_obj = np.array(obj.matrix_world)

            # Transform the vertices to world coordinates and store them in the point cloud
            world_vertices = world_from_obj[:3, :3] @ vertices.T + world_from_obj[:3, 3:]
            point_cloud[point_cloud_idx:point_cloud_idx + len(vertices)] = world_vertices.T

            # Update the point cloud index
            point_cloud_idx += len(vertices)

        # Get camera from world transformation
        camera_from_world = np.linalg.inv(self.get_world_from_camera())

        # Transform the point cloud to camera coordinates
        camera_point_cloud = camera_from_world[:3, :3] @ point_cloud.T + camera_from_world[:3, 3:]

        # Project the point cloud to the image plane
        camera_point_cloud_px = self.camera_matrix @ camera_point_cloud
        camera_point_cloud_px = camera_point_cloud_px[:2] / camera_point_cloud_px[2]

        # Filter points that are outside the image plane
        mask &= (
            (camera_point_cloud[2] > 0) &
            (0 <= camera_point_cloud_px[0]) & (camera_point_cloud_px[0] < image_width) &
            (0 <= camera_point_cloud_px[1]) & (camera_point_cloud_px[1] < image_height)
        )

        # Check occlusion for each valid point
        for idx in np.where(mask)[0]:

            point = mathutils.Vector(point_cloud[idx])

            # Compute ray direction, from the camera center to the point
            ray_direction = point - origin

            # Racyast from the camera center to the point
            collided, collision_location, _, _, _, _ = self.scene.ray_cast(
                depsgraph=self.view_layer.depsgraph,
                origin=origin,
                direction=ray_direction.normalized(),
                distance=ray_direction.length
            )

            # Filter point if the raycast hit somewhere different from the vertice
            if collided and ((point - collision_location).length > 0.01):
                mask[idx] = False

        # Show the point cloud if needed
        if imshow:
            point_cloud_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            point_cloud_image[
                np.int64(camera_point_cloud_px[1, mask]),
                np.int64(camera_point_cloud_px[0, mask])
            ] = self.point_cloud_colors[mask]
            cv2.imshow(f'points', point_cloud_image)

        return point_cloud, mask

    def set_points_density(self, density: float):
        self.pcl_node.nodes['Distribute Points on Faces'].inputs['Density'].default_value = density

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
        simulator.get_point_cloud(imshow=True)

    cv2.destroyAllWindows()
