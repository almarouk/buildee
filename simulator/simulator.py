import argparse
import tempfile
from collections import OrderedDict
from pathlib import Path

import bpy
import cv2
import mathutils
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from .blender_utils import (
    is_animated,
    get_visible_objects,
    compute_bvh_tree,
    get_label,
    deselect_all,
    select_object
)
from .os_utils import suppress_output


class Simulator:
    def __init__(
            self,
            blend_file: str | Path,
            points_density: float = 1.0,
            segmentation_sensitivity: float = 0.1,
            geometry_nodes_objects: list[str] = ('Generator',),
            filter_object_names: list[str] = ('CameraBounds',),
            verbose: bool = False
    ):
        """Initialize the simulator.

        :param blend_file: path to the blender file
        :param points_density: density of 3D points to sample per unit area
        :param segmentation_sensitivity: border sensitivity for segmentation masks, higher values discard aliased edges
        :param filter_object_names: list of object names to ignore for 3D points sampling and semantic segmentation
        :param verbose: if True, print debug information
        """
        # Set verbose mode
        self.verbose = verbose

        # Load blender file
        with suppress_output(self.verbose):
            bpy.ops.wm.open_mainfile(filepath=str(blend_file))

        # Get the current scene
        self.scene = bpy.context.scene

        # Get the current camera
        self.camera = self.scene.camera

        # Set camera clip start and end
        self.camera.data.clip_start = 0.01
        self.camera.data.clip_end = 1000.0

        # Get the current view layer
        self.view_layer = bpy.context.view_layer

        # Get os temp dir
        self.temp_dir = Path(tempfile.gettempdir())

        # Get rendered image path
        self.render_path = self.temp_dir / 'render.exr'

        # Get camera matrix
        self.camera_matrix = self.get_camera_matrix()

        # Instantiate geometry nodes
        self.init_geometry_nodes(geometry_nodes_objects)

        # Get camera spawn points:
        self.spawn_points = self.init_camera_spawn()

        # Get all visible objects
        self.objects = OrderedDict([
            (obj, is_animated(obj)) for obj in get_visible_objects(
                self.view_layer.layer_collection,
                filter_names=filter_object_names
            )
        ])

        # Get vertices and polygons of each object for point cloud occlusion checking
        # Static objects and dynamic objects have their own set of vertices and polygons
        self.static_verts_polys, self.dynamic_verts_polys = self.init_vertices_polygons()

        # Get the number of static and dynamic vertices
        self.n_static_vertices = sum(len(vertices) for vertices, _ in self.static_verts_polys.values())
        self.n_dynamic_vertices = sum(len(vertices) for vertices, _ in self.dynamic_verts_polys.values())
        self.n_vertices = self.n_static_vertices + self.n_dynamic_vertices
        if self.verbose:
            print(f'Found {self.n_static_vertices} static vertices and {self.n_dynamic_vertices} dynamic vertices')

        # Sample 3D points for every object in the scene
        # Static objects and dynamic objects have their own set of point clouds
        self.static_point_clouds, self.dynamic_point_clouds = self.init_object_point_clouds(
            points_density=points_density
        )

        # Get the number of static and dynamic points
        self.n_static_points = sum(len(vertices) for vertices in self.static_point_clouds.values())
        self.n_dynamic_points = sum(len(vertices) for vertices in self.dynamic_point_clouds.values())
        self.n_points = self.n_static_points + self.n_dynamic_points
        # self.point_cloud_colors = np.random.randint(0, 256, (self.n_points, 3), dtype=np.uint8)  # rendering colors
        if self.verbose:
            print(f'Extracted {self.n_static_points} static points and {self.n_dynamic_points} dynamic points')

        # Compute a BVH tree for static objects for fast point cloud occlusion checking
        # Dynamic objects have their own BVH tree that will be computed on the fly
        self.static_bvh = compute_bvh_tree(self.static_verts_polys)

        # Store a mask of the observed 3D points
        self.observed_points_mask = np.zeros(self.n_points, dtype=bool)

        # Setup nodes to render depth in the alpha channel
        self.scene.use_nodes = True
        self.scene.view_layers["ViewLayer"].use_pass_z = True
        self.render_node = self.scene.node_tree.nodes["Render Layers"]
        self.output_node = self.scene.node_tree.nodes["Composite"]
        self.scene.node_tree.links.new(self.render_node.outputs['Depth'], self.output_node.inputs['Alpha'])

        # Setup rendering color space
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

        # Setup semantic segmentation
        self.view_layer.use_pass_cryptomatte_object = True
        self.view_layer.pass_cryptomatte_depth = 2
        self.matte_output_node = self.scene.node_tree.nodes.new(type='CompositorNodeOutputFile')
        self.matte_output_node.base_path = str(self.temp_dir)
        self.matte_output_node.format.file_format = 'OPEN_EXR'
        self.matte_output_node.format.color_mode = 'BW'
        self.matte_output_node.format.color_depth = '32'
        self.matte_output_node.format.exr_codec = 'NONE'
        self.matte_output_node.format.color_management = 'OVERRIDE'
        self.matte_output_node.format.linear_colorspace_settings.name = 'Non-Color'
        self.matte_output_node.file_slots[0].path = 'matte'

        # Get object labels based on object names
        self.object_labels = OrderedDict(
            (obj, get_label(obj)) for obj in self.objects
        )

        # Get list of labels from all objects
        self.labels = list(set(self.object_labels.values()))

        # Get number of labels
        self.n_labels = len(self.labels)
        if self.verbose:
            print(f'Found {self.n_labels} labels: {self.labels}')

        # Setup semantic segmentation nodes
        self.init_semantic_segmentation(segmentation_sensitivity=segmentation_sensitivity)

        # Setup colors for point cloud rendering
        self.point_cloud_colors, _, _ = self.compute_point_cloud(update_mask=False)
        self.point_cloud_colors -= self.point_cloud_colors.min(axis=0)
        self.point_cloud_colors /= self.point_cloud_colors.max(axis=0)
        self.point_cloud_colors = np.uint8(self.point_cloud_colors * 255)

    def init_vertices_polygons(self) -> tuple[
        OrderedDict[bpy.types.Object, tuple[np.ndarray, list[list[int]]]],
        OrderedDict[bpy.types.Object, tuple[np.ndarray, list[list[int]]]]
    ]:
        """Get vertices and polygons indices for every object in the scene."""
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

            if dynamic:  # store in dynamic vertices dict
                polygons = [
                    list(map(lambda vertex: vertex + dynamic_vertex_offset, poly.vertices)) for poly in mesh.polygons
                ]
                dynamic_vertex_offset += len(mesh_vertices)
                dynamic_verts_polys[obj] = (mesh_vertices, polygons)
            else:  # store in static vertices dict
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
        """Sample 3D points for every object in the scene.

        :param points_density: density of points to sample per unit area
        """
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

    def init_geometry_nodes(self, geometry_nodes_objects: list[str]):
        """Initialize geometry nodes for the scene."""
        print('Convert geometry nodes to meshes...')
        for obj_name in geometry_nodes_objects:
            if obj_name in self.scene.objects:
                obj = self.scene.objects[obj_name]
                select_object(obj, self.view_layer)
                bpy.ops.object.duplicates_make_real()
                select_object(obj, self.view_layer)
                bpy.ops.object.convert(target='MESH')
                deselect_all(self.view_layer)

    def init_camera_spawn(self) -> np.ndarray:
        """Sample camera spawn points from the CameraSpawn object or use the camera's initial position."""
        # The camera spawn is the camera's initial position
        world_vertices = np.array(self.camera.matrix_world.translation)[None]

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

    def init_semantic_segmentation(self, segmentation_sensitivity: float):
        """Setup semantic segmentation nodes for the scene.

        :param segmentation_sensitivity: border sensitivity for segmentation masks, higher values discard aliased edges
        """
        # Create a Cryptomatte node for each object in the scene
        # Combine each Cryptomatte node based on depth
        last_zcombine_node = self.scene.node_tree.nodes.new(type='CompositorNodeZcombine')
        last_zcombine_node.use_antialias_z = False

        # Create a progress bar
        progress_bar = tqdm.tqdm(total=len(self.object_labels), desc='Cryptomattes', disable=not self.verbose)

        # For each object, add a Cryptomatte node and merge it with the Cryptomatte node of the previous object
        for obj_id, (obj, obj_label) in enumerate(self.object_labels.items()):
            # Get label id
            label_id = self.labels.index(obj_label)

            # Create a Cryptomatte node for the current object
            matte_node = self.scene.node_tree.nodes.new(type='CompositorNodeCryptomatteV2')
            matte_node.matte_id = obj.name

            # Cryptomattes are anti-aliased, so we need to threshold them
            math_node = self.scene.node_tree.nodes.new(type='CompositorNodeMath')
            math_node.operation = 'GREATER_THAN'
            math_node.inputs[1].default_value = segmentation_sensitivity

            # Apply its id to the cryptomatte mask
            mix_matte_node = self.scene.node_tree.nodes.new(type='CompositorNodeMath')
            mix_matte_node.operation = 'MULTIPLY'
            mix_matte_node.inputs[1].default_value = label_id + 1

            # We are going to merge cryptomattes based on depth,
            # so we set the depth to a high value where the mask is zero
            mix_depth_node = self.scene.node_tree.nodes.new(type='CompositorNodeMixRGB')
            mix_depth_node.inputs[1].default_value = (
                self.camera.data.clip_end, self.camera.data.clip_end, self.camera.data.clip_end, 1
            )

            # Link the nodes
            self.scene.node_tree.links.new(matte_node.outputs['Matte'], math_node.inputs[0])
            self.scene.node_tree.links.new(math_node.outputs['Value'], mix_matte_node.inputs[0])
            self.scene.node_tree.links.new(math_node.outputs['Value'], mix_depth_node.inputs[0])
            self.scene.node_tree.links.new(self.render_node.outputs['Depth'], mix_depth_node.inputs[2])

            if obj_id == 0:  # identity ZCombine node
                self.scene.node_tree.links.new(mix_matte_node.outputs['Value'], last_zcombine_node.inputs[0])
                self.scene.node_tree.links.new(mix_depth_node.outputs['Image'], last_zcombine_node.inputs[1])
                self.scene.node_tree.links.new(mix_matte_node.outputs['Value'], last_zcombine_node.inputs[2])
                self.scene.node_tree.links.new(mix_depth_node.outputs['Image'], last_zcombine_node.inputs[3])
            else:  # merge with previous ZCombine node
                zcombine_node = self.scene.node_tree.nodes.new(type='CompositorNodeZcombine')
                zcombine_node.use_antialias_z = False
                self.scene.node_tree.links.new(last_zcombine_node.outputs['Image'], zcombine_node.inputs[0])
                self.scene.node_tree.links.new(last_zcombine_node.outputs['Z'], zcombine_node.inputs[1])
                self.scene.node_tree.links.new(mix_matte_node.outputs['Value'], zcombine_node.inputs[2])
                self.scene.node_tree.links.new(mix_depth_node.outputs['Image'], zcombine_node.inputs[3])
                last_zcombine_node = zcombine_node

            # Update progress bar
            progress_bar.set_postfix_str(obj.name)
            progress_bar.update()

        # Connect the final ZCombine node to the matte output node
        self.scene.node_tree.links.new(last_zcombine_node.outputs['Image'], self.matte_output_node.inputs['Image'])

    def get_camera_matrix(self) -> np.ndarray:
        """Get the camera matrix from the current camera parameters."""
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
        """Get the transformation matrix from world to camera coordinates."""
        # Read blender camera pose
        world_from_camera = np.array(self.camera.matrix_world)

        # Rotate the camera 180 degrees around the x-axis to fit blender's camera coordinate system
        world_from_camera[:, 1:3] *= -1  # same as doing world_from_camera @ R(180, x)

        return world_from_camera

    def set_world_from_camera(self, world_from_camera: np.ndarray, check_collisions: bool = True) -> bool:
        """Set the camera pose from a transformation matrix.

        :param world_from_camera: transformation matrix from world to camera coordinates
        :param check_collisions: check for collisions with objects in the scene
        :return: True if ``check_collisions`` were enabled and there was a collision, False otherwise
        """
        # Check that there is no collision between current camera and new camera pose
        if check_collisions:
            origin = self.camera.matrix_world.translation
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
        """Set the camera pose from a transformation matrix relative to the current camera pose. Also checks
        for collisions with objects in the scene. Camera does not move if there is a collision.

        :param camera_from_next_camera: transformation matrix from camera to next camera coordinates
        :return: True if there was a collision, False otherwise
        """
        return self.set_world_from_camera(self.get_world_from_camera() @ camera_from_next_camera)

    def move_camera_forward(self, distance: float) -> bool:
        """Move the camera along its z-axis and check for collisions.

        :param distance: distance to move the camera
        :return: True if there was a collision, False otherwise
        """
        # Move camera along the z-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, distance],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def move_camera_down(self, distance: float) -> bool:
        """Move the camera along its y-axis and check for collisions.

        :param distance: distance to move the camera
        :return: True if there was a collision, False otherwise
        """
        # Move camera along the y-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, distance],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def move_camera_right(self, distance: float) -> bool:
        """Move the camera along its x-axis and check for collisions.

        :param distance: distance to move the camera
        :return: True if there was a collision, False otherwise
        """
        # Move camera along the x-axis
        return self.set_camera_from_next_camera(np.array([
            [1.0, 0.0, 0.0, distance],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]))

    def turn_camera_right(self, angle: float, degrees: bool = True) -> bool:
        """Rotate the camera along its y-axis and check for collisions.

        :param angle: angle to rotate the camera
        :param degrees: if True, angle is in degrees, otherwise in radians
        :return: True if there was a collision, False otherwise
        """
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
        """Increment simulation frame."""
        self.scene.frame_set(self.scene.frame_current + 1)

    def respawn_camera(self):
        """Respawn the camera at a random spawn point and randomly rotate the camera around its y-axis."""
        # Get random spawn point
        x, y, z = self.spawn_points[np.random.randint(len(self.spawn_points))]

        # Set the camera pose
        self.set_world_from_camera(np.array([
            [1, 0, 0, x],
            [0, 0, 1, y],
            [0, -1, 0, z],
            [0, 0, 0, 1]
        ]), check_collisions=False)

        # Randomly rotate the camera around the z-axis
        self.turn_camera_right(np.random.uniform(0, 2 * np.pi), degrees=False)

    def depth_to_world_points(self, depth: np.ndarray) -> np.ndarray:
        """Unproject depth map to world points given the current camera pose.

        :param depth: depth map with shape (h, w)
        :return: 3D points in world coordinates with shape (h, w, 3)
        """
        world_from_cam = self.get_world_from_camera()  # get world from camera transformation matrix
        v, u = np.where((0 < depth) & (depth < self.camera.data.clip_end - 1))  # get pixel coordinates at valid depths
        uvws = np.stack([u + 0.5, v + 0.5, np.ones_like(u)])  # homogeneous pixel coordinates
        cam_points = np.linalg.inv(self.camera_matrix) @ (depth[v, u] * uvws)  # unproject in camera space
        world_points = world_from_cam[:3, :3] @ cam_points + world_from_cam[:3, 3:]  # transform to world space
        points = np.full((*depth.shape, 3), np.nan)  # setup (h, w, 3) array, invalid depths are nan
        points[v, u] = world_points.T  # store points where depth is valid
        return points

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render the scene given current render settings and scene parameters. Returns the rendered RGB image,
        the corresponding depth map and the semantic segmentation map. The semantic segmentation map indicates
        the label id of each pixel in the image. Label ids correspond to their position in ``Simulator.labels``.
        Pixels with no labels are assigned -1.

        :return: RGB image, depth map, semantic segmentation map
        """
        # Mute blender and render the scene
        with suppress_output():
            bpy.ops.render.render(write_still=True)

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

        # Load semantic segmentation data
        matte_path = self.temp_dir / f'matte{self.scene.frame_current:04d}.exr'
        render = bpy.data.images.load(str(matte_path))

        # Read semantic segmentation data
        matte = np.empty(len(render.pixels), dtype=np.float32)
        render.pixels.foreach_get(matte)
        matte = np.flip(matte.reshape(
            self.scene.render.resolution_y, self.scene.render.resolution_x, 4
        ), axis=0)[:, :, 0]

        # Assign label ids to matte image, -1 for no labels
        matte = np.int32(matte + 0.5) - 1

        # Clear semantic segmentation image data
        bpy.data.images.remove(render)

        # Remove matte file
        matte_path.unlink()

        return rgbd[:, :, :3], rgbd[:, :, 3], matte

    def compute_point_cloud(
            self,
            update_mask: bool = True,
            imshow: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the sampled point cloud of the scene at the current frame. Returns the point cloud in world coordinates,
        the corresponding labels for each point and a mask indicating which points are visible in the rendered image.

        :param update_mask: whether to update the simulator's observed points mask
        :param imshow: whether to show an image of the point cloud in the current view
        :return: point cloud, point cloud labels, mask indicating visible points
        """
        # Store all the points in a single point cloud
        point_cloud = np.empty((self.n_points, 3))
        point_cloud_labels = np.empty(self.n_points, dtype=np.int32)
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
            point_cloud_labels[point_cloud_idx:point_cloud_idx + len(points)] = self.labels.index(
                self.object_labels[obj]
            )
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
        origin = self.camera.matrix_world.translation

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

        return point_cloud, point_cloud_labels, mask


def explore(
        blend_file: str | Path,
        points_density: float = 1.0,
        segmentation_sensitivity: float = 0.1,
        geometry_nodes_objects: list[str] = ('Generator',),
        filter_object_names: list[str] = ('CameraBounds',),
        max_depth_distance_display: float = 10.0,
        verbose: bool = False
):
    """Explore the 3D scene."""
    # Create a simulator
    simulator = Simulator(
        blend_file=blend_file,
        points_density=points_density,
        segmentation_sensitivity=segmentation_sensitivity,
        geometry_nodes_objects=geometry_nodes_objects,
        filter_object_names=filter_object_names,
        verbose=verbose
    )

    # Setup depth colormap
    depth_color_map = plt.get_cmap('magma')

    # Setup segmentation colormap
    seg_color_map = plt.get_cmap('jet')

    # Main loop
    for _ in tqdm.tqdm(range(999999999)):

        # Get key pressed
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
            simulator.turn_camera_right(-22.5, degrees=True)
        elif key == ord('e'):
            simulator.turn_camera_right(22.5, degrees=True)
        elif key == 27:  # escape key
            break

        # Render image
        rgb, depth, seg = simulator.render()

        # Setup depth for display
        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        # Setup segmentation for display
        seg = seg_color_map((seg + 1) / simulator.n_labels)[:, :, :3]

        # Show rgb, depth and point cloud
        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'segmentation', cv2.cvtColor(np.uint8(seg * 255), cv2.COLOR_RGB2BGR))
        simulator.compute_point_cloud(imshow=True)

        # Step to next frame (update animations)
        simulator.step_frame()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore a Blender 3D scene.')
    parser.add_argument('--blend-file', type=str, required=True, help='path to the blender file')
    parser.add_argument(
        '--points-density', type=float, default=1.0, help='density of 3D points to sample per unit area'
    )
    parser.add_argument(
        '--segmentation-sensitivity', type=float, default=0.1,
        help='border sensitivity for segmentation masks, higher values discard aliased edges'
    )
    parser.add_argument(
        '--generator-objects', type=str, nargs='+', default=('Generator',),
        help='list of generator objects, such as geometry nodes or particles, to convert to meshes'
    )
    parser.add_argument(
        '--filter-object-names', type=str, nargs='+', default=('CameraBounds', 'CameraSpawn'),
        help='list of object names to ignore for 3D points sampling and semantic segmentation'
    )
    parser.add_argument(
        '--max-depth-distance_display', type=float, default=10.0,
        help='maximum depth distance to display'
    )
    parser.add_argument('--verbose', action='store_true', help='print debug information')
    args = parser.parse_args()
    explore(
        blend_file=args.blend_file,
        points_density=args.points_density,
        segmentation_sensitivity=args.segmentation_sensitivity,
        geometry_nodes_objects=args.generator_objects,
        filter_object_names=args.filter_object_names,
        max_depth_distance_display=args.max_depth_distance_display,
        verbose=args.verbose
    )
