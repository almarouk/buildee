import tempfile
from collections import OrderedDict
from pathlib import Path

import bpy
import cv2
import mathutils
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from blender_utils import is_animated, get_visible_objects, compute_bvh_tree
from os_utils import redirect_output_to_null, restore_output


class Simulator:
    def __init__(
            self,
            blend_file: str | Path,
            points_density: float = 1.0,
            filter_object_names: list[str] = ('CameraBounds', 'CameraSpawn'),
            verbose: bool = False
    ):
        # Set verbose mode
        self.verbose = verbose

        # Load blender file
        if not self.verbose:
            devnull, original_stdout, original_stderr = redirect_output_to_null()  # redirect print output
            bpy.ops.wm.open_mainfile(filepath=str(blend_file))
            restore_output(devnull, original_stdout, original_stderr)  # restore print output
        else:
            bpy.ops.wm.open_mainfile(filepath=str(blend_file))

        # Get the current scene
        self.scene = bpy.context.scene

        # Get the current camera
        self.camera = self.scene.camera

        # Set camera clip start
        self.camera.data.clip_start = 0.01

        # Get the current view layer
        self.view_layer = bpy.context.view_layer

        # Get os temp dir
        temp_dir = Path(tempfile.gettempdir())

        # Get rendered image path
        self.render_path = temp_dir / 'render.exr'

        # Get camera matrix
        self.camera_matrix = self.get_camera_matrix()

        # Get camera spawn points:
        self.spawn_points = self.init_camera_spawn()

        # Get all visible objects
        self.objects = OrderedDict([
            (obj, is_animated(obj)) for obj in get_visible_objects(
                self.view_layer.layer_collection,
                filter_names=filter_object_names
            )
        ])

        # Get all vertices and polygons for BVH tree for fast point cloud occlusion checking
        self.static_verts_polys, self.dynamic_verts_polys = self.init_vertices_polygons()
        self.n_static_vertices = sum(len(vertices) for vertices, _ in self.static_verts_polys.values())
        self.n_dynamic_vertices = sum(len(vertices) for vertices, _ in self.dynamic_verts_polys.values())
        self.n_vertices = self.n_static_vertices + self.n_dynamic_vertices
        if self.verbose:
            print(f'Found {self.n_static_vertices} static vertices and {self.n_dynamic_vertices} dynamic vertices')

        # Get a point cloud representation of every static and dynamic object in the scene
        self.static_point_clouds, self.dynamic_point_clouds = self.init_object_point_clouds(
            points_density=points_density
        )
        self.n_static_points = sum(len(vertices) for vertices in self.static_point_clouds.values())
        self.n_dynamic_points = sum(len(vertices) for vertices in self.dynamic_point_clouds.values())
        self.n_points = self.n_static_points + self.n_dynamic_points
        self.point_cloud_colors = np.random.randint(0, 256, (self.n_points, 3), dtype=np.uint8)  # rendering colors
        if self.verbose:
            print(f'Extracted {self.n_static_points} static points and {self.n_dynamic_points} dynamic points')

        # Compute the static BVH tree for fast point cloud occlusion checking
        self.static_bvh = compute_bvh_tree(self.static_verts_polys)

        # Store a mask of the observed 3d points
        self.observed_points_mask = np.zeros(self.n_points, dtype=bool)

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

    def init_vertices_polygons(self) -> tuple[
        OrderedDict[bpy.types.Object, tuple[np.ndarray, list[list[int]]]],
        OrderedDict[bpy.types.Object, tuple[np.ndarray, list[list[int]]]]
    ]:
        # Get vertices and polygons indices for every object in the scene
        static_verts_polys, dynamic_verts_polys = OrderedDict(), OrderedDict()
        progress_bar = tqdm.tqdm(total=len(self.objects), desc='BVH', disable=not self.verbose)

        # Initialize static and dynamic vertex offset
        static_vertex_offset, dynamic_vertex_offset = 0, 0

        for obj, dynamic in self.objects.items():

            # Convert to mesh to apply modifiers
            evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated_obj.to_mesh()

            # Get the vertices in object coordinates
            mesh_vertices = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get('co', mesh_vertices)
            mesh_vertices = mesh_vertices.reshape(-1, 3)

            if dynamic:
                polygons = [
                    list(map(lambda vertex: vertex + dynamic_vertex_offset, poly.vertices)) for poly in mesh.polygons
                ]
                dynamic_vertex_offset += len(mesh_vertices)
                dynamic_verts_polys[obj] = (mesh_vertices, polygons)
            else:
                polygons = [
                    list(map(lambda vertex: vertex + static_vertex_offset, poly.vertices)) for poly in mesh.polygons
                ]
                static_vertex_offset += len(mesh_vertices)
                static_verts_polys[obj] = (mesh_vertices, polygons)

            # Free temporary mesh data
            evaluated_obj.to_mesh_clear()

            # Update progress bar
            progress_bar.set_postfix_str(obj.name)
            progress_bar.update()

        progress_bar.close()

        return static_verts_polys, dynamic_verts_polys

    def init_object_point_clouds(
            self, points_density: float
    ) -> tuple[OrderedDict[bpy.types.Object, np.ndarray], OrderedDict[bpy.types.Object, np.ndarray]]:
        # Initialize point cloud geometry node
        pcl_node = bpy.data.node_groups.new(name="Pointcloud", type='GeometryNodeTree')
        pcl_input_node = pcl_node.nodes.new(type='NodeGroupInput')
        pcl_output_node = pcl_node.nodes.new(type='NodeGroupOutput')
        pcl_points_node = pcl_node.nodes.new(type="GeometryNodeDistributePointsOnFaces")
        pcl_vertices_node = pcl_node.nodes.new(type="GeometryNodePointsToVertices")
        pcl_realize_node = pcl_node.nodes.new(type="GeometryNodeRealizeInstances")
        pcl_node.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
        pcl_node.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
        pcl_node.links.new(pcl_input_node.outputs["Geometry"], pcl_points_node.inputs["Mesh"])
        pcl_node.links.new(pcl_points_node.outputs["Points"], pcl_vertices_node.inputs["Points"])
        pcl_node.links.new(pcl_vertices_node.outputs["Mesh"], pcl_realize_node.inputs["Geometry"])
        pcl_node.links.new(pcl_realize_node.outputs["Geometry"], pcl_output_node.inputs["Geometry"])
        pcl_points_node.inputs['Density'].default_value = points_density

        # Get a point cloud representation of every object in the scene
        # Static point clouds are expressed in world coordinates
        # Dynamic point clouds are expressed in object coordinates
        static_point_clouds, dynamic_point_clouds = OrderedDict(), OrderedDict()
        progress_bar = tqdm.tqdm(total=len(self.objects), desc='Point clouds', disable=not self.verbose)

        for obj, dynamic in self.objects.items():

            # Apply Pointcloud modifier to object
            obj_modifier = obj.modifiers.new('GeometryNodes', type='NODES')
            obj_modifier.node_group = pcl_node

            # Convert to mesh to apply modifiers
            evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated_obj.to_mesh()

            # Get vertices in object coordinates
            mesh_vertices = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get('co', mesh_vertices)
            mesh_vertices = mesh_vertices.reshape(-1, 3)

            if dynamic:  # store dynamic point clouds in object coordinates
                dynamic_point_clouds[obj] = mesh_vertices
            else:  # store static point clouds in world coordinates
                world_from_obj = np.array(obj.matrix_world)
                static_point_clouds[obj] = (world_from_obj[:3, :3] @ mesh_vertices.T + world_from_obj[:3, 3:]).T

            # Clear memory and restore object
            evaluated_obj.to_mesh_clear()
            obj.modifiers.remove(obj_modifier)

            # Update progress bar
            progress_bar.set_postfix_str(obj.name)
            progress_bar.update()

        progress_bar.close()

        return static_point_clouds, dynamic_point_clouds

    def init_camera_spawn(self) -> np.ndarray:
        # The camera spawn is the camera's initial position
        world_vertices = np.array(self.camera.location)[None]

        # If there is a CameraSpawn object, use it to sample camera spawn points
        if 'CameraSpawn' in self.scene.objects:
            # Create volume sampling geometry node
            spawn_node = bpy.data.node_groups.new(name="Spawn", type='GeometryNodeTree')
            spawn_input_node = spawn_node.nodes.new(type='NodeGroupInput')
            spawn_output_node = spawn_node.nodes.new(type='NodeGroupOutput')
            spawn_volume_node = spawn_node.nodes.new(type='GeometryNodeMeshToVolume')
            spawn_points_node = spawn_node.nodes.new(type='GeometryNodeDistributePointsInVolume')
            spawn_vertices_node = spawn_node.nodes.new(type="GeometryNodePointsToVertices")
            spawn_realize_node = spawn_node.nodes.new(type="GeometryNodeRealizeInstances")
            spawn_node.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            spawn_node.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
            spawn_node.links.new(spawn_input_node.outputs["Geometry"], spawn_volume_node.inputs["Mesh"])
            spawn_node.links.new(spawn_volume_node.outputs["Volume"], spawn_points_node.inputs["Volume"])
            spawn_node.links.new(spawn_points_node.outputs["Points"], spawn_vertices_node.inputs["Points"])
            spawn_node.links.new(spawn_vertices_node.outputs["Mesh"], spawn_realize_node.inputs["Geometry"])
            spawn_node.links.new(spawn_realize_node.outputs["Geometry"], spawn_output_node.inputs["Geometry"])
            spawn_volume_node.inputs['Voxel Amount'].default_value = 256.0
            spawn_points_node.inputs['Density'].default_value = 1000.0

            # Get CameraSpawn object
            obj = self.scene.objects['CameraSpawn']

            # Apply Spawn modifier to object
            obj_modifier = obj.modifiers.new('GeometryNodes', type='NODES')
            obj_modifier.node_group = spawn_node

            # Convert to mesh to apply modifiers
            evaluated_obj = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
            mesh = evaluated_obj.to_mesh()

            # Get vertices in object coordinates
            mesh_vertices = np.empty(len(mesh.vertices) * 3)
            mesh.vertices.foreach_get('co', mesh_vertices)
            mesh_vertices = mesh_vertices.reshape(-1, 3)

            # Store spawn points in world coordinates
            world_from_obj = np.array(obj.matrix_world)
            world_vertices = (world_from_obj[:3, :3] @ mesh_vertices.T + world_from_obj[:3, 3:]).T

            # Clear memory and remove object
            evaluated_obj.to_mesh_clear()
            bpy.data.objects.remove(obj)

        return world_vertices

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        # Mute blender and render the scene
        devnull, original_stdout, original_stderr = redirect_output_to_null()  # redirect print output
        bpy.ops.render.render(write_still=True)  # render the scene
        restore_output(devnull, original_stdout, original_stderr)  # restore print output

        # Load the image
        render = bpy.data.images.load(str(self.render_path))

        # Read image data
        rgbd = np.empty(len(render.pixels), dtype=np.float32)
        render.pixels.foreach_get(rgbd)
        rgbd = np.flip(rgbd.reshape(
            self.scene.render.resolution_y, self.scene.render.resolution_x, 4
        ), axis=0)

        # Clear blender image data
        bpy.data.images.remove(render)

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

    def get_world_from_camera(self) -> np.ndarray:
        # Read blender camera pose
        world_from_camera = np.array(self.camera.matrix_world)

        # Rotate the camera 180 degrees around the x-axis to fit blender's camera coordinate system
        world_from_camera[:, 1:3] *= -1  # same as doing world_from_camera @ R(180, x)

        return world_from_camera

    def get_point_cloud(self, update_mask: bool = True, imshow: bool = False) -> (np.ndarray, np.ndarray):
        # Store all the points in a single point cloud
        point_cloud = np.empty((self.n_points, 3))
        point_cloud_idx = 0

        # Iterate over all objects and store their transformed vertices in the point cloud
        for obj, dynamic in self.objects.items():

            if dynamic:  # only transform points of dynamic objects
                points = self.dynamic_point_clouds[obj]
                world_from_obj = np.array(obj.matrix_world)
                points = (world_from_obj[:3, :3] @ points.T + world_from_obj[:3, 3:]).T
            else:  # simply load vertices
                points = self.static_point_clouds[obj]

            # Store the points in the point cloud and update cloud index
            point_cloud[point_cloud_idx:point_cloud_idx + len(points)] = points
            point_cloud_idx += len(points)

        # Set all points as visible
        mask = np.ones(self.n_points, dtype=bool)

        # Get image width and height
        image_width = self.scene.render.resolution_x
        image_height = self.scene.render.resolution_y

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

        # Get camera center, origin of raycast
        origin = self.camera.location

        # Compute dynamic BVH tree
        dynamic_bvh = compute_bvh_tree(self.dynamic_verts_polys)

        # Check occlusion for each valid point
        for idx in np.where(mask)[0]:

            point = mathutils.Vector(point_cloud[idx])

            # Compute ray direction, from the camera center to the point
            ray_direction = point - origin

            # Check collision with both static and dynamic BVH trees
            for bvh in [self.static_bvh, dynamic_bvh]:

                # Racyast from the camera center to the point
                collision_location, _, _, _ = bvh.ray_cast(
                    origin,
                    ray_direction.normalized(),
                    ray_direction.length
                )

                # Filter point if the raycast hit somewhere different from the vertice
                if (collision_location is not None) and ((point - collision_location).length > 0.01):
                    mask[idx] = False

        # Update the observed points mask
        if update_mask:
            self.observed_points_mask |= mask

        # Show the point cloud if needed
        if imshow:
            point_cloud_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
            point_cloud_image[
                np.int64(camera_point_cloud_px[1, mask]),
                np.int64(camera_point_cloud_px[0, mask])
            ] = self.point_cloud_colors[mask]
            cv2.imshow(f'points', point_cloud_image)

        return point_cloud, mask

    def set_world_from_camera(self, world_from_camera: np.ndarray, check_collisions: bool = True) -> bool:
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
                    if self.verbose:
                        print('Collision detected, not moving the camera.')
                    return True  # there was a collision

        # Copy the camera pose
        world_from_camera = world_from_camera.copy()

        # Rotate the camera 180 degrees around the x-axis to fit blender's camera coordinate system
        world_from_camera[:3, 1:3] *= -1  # same as doing world_from_camera @ R(180, x)

        # Set the camera pose
        self.camera.matrix_world = mathutils.Matrix(world_from_camera)

        # There was no collision
        return False

    def set_camera_from_next_camera(self, camera_from_next_camera: np.ndarray) -> bool:
        return self.set_world_from_camera(self.get_world_from_camera() @ camera_from_next_camera)

    def respawn_camera(self):
        # Respawn the camera at a random spawn point
        x, y, z = self.spawn_points[np.random.randint(len(self.spawn_points))]

        # Set the camera pose
        self.set_world_from_camera(np.array([
            [1, 0, 0, x],
            [0, 0, 1, y],
            [0, -1, 0, z],
            [0, 0, 0, 1]
        ]), check_collisions=False)

        # Randomly rotate the camera around the z-axis
        self.rotate_camera_yaw(np.random.uniform(0, 2 * np.pi), degrees=False)

    def move_camera_forward(self, distance: float) -> bool:
        # Move camera along the z-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def move_camera_right(self, distance: float) -> bool:
        # Move camera along the x-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, distance],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def move_camera_down(self, distance: float) -> bool:
        # Move camera along the y-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, distance],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def rotate_camera_yaw(self, angle: float, degrees: bool = True) -> bool:
        # Convert angle to radians if needed
        if degrees:
            angle = np.radians(angle)

        # Rotate camera around the y-axis
        return self.set_camera_from_next_camera(np.array([
            [np.cos(angle), 0.0, np.sin(angle), 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle), 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def step_frame(self):
        self.scene.frame_set(self.scene.frame_current + 1)


if __name__ == '__main__':
    # Create a simulator
    simulator = Simulator('/home/clementin/Data/blendernbv/liberty.blend', points_density=100.0, verbose=True)

    depth_color_map = plt.get_cmap('magma')
    max_depth_distance_display = 10.0

    for _ in tqdm.tqdm(range(999999999)):

        key = cv2.waitKeyEx(7)

        if key == ord('q'):
            simulator.move_camera_right(-1)
        elif key == ord('d'):
            simulator.move_camera_right(1)
        elif key == ord('z'):
            simulator.move_camera_forward(1)
        elif key == ord('s'):
            simulator.move_camera_forward(-1)
        elif key == 32:
            simulator.move_camera_down(-1)
        elif key == ord('c'):
            simulator.move_camera_down(1)
        elif key == ord('a'):
            simulator.rotate_camera_yaw(-22.5, degrees=True)
        elif key == ord('e'):
            simulator.rotate_camera_yaw(22.5, degrees=True)
        elif key == 27:  # escape key
            break

        rgb, depth = simulator.render()

        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))
        simulator.get_point_cloud(imshow=True)

        # simulator.step_frame()

    cv2.destroyAllWindows()
