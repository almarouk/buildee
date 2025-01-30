import re
from collections import OrderedDict

import bpy
import mathutils
import numpy as np


def has_animation_data(obj: bpy.types.Object) -> bool:
    # Returns True if the object has animation data
    if obj.animation_data is not None:
        return (obj.animation_data.action is not None) or (obj.animation_data.drivers is not None)
    return False


def is_animated(obj: bpy.types.Object) -> bool:
    # Returns True if the object or any of its parents have animation data
    if has_animation_data(obj):
        return True
    elif obj.parent:
        return is_animated(obj.parent)
    return False


def get_visible_collections(layer_collection: bpy.types.Collection) -> list[bpy.types.Collection]:
    # Recursively get all visible collections
    visible_collections = []
    if layer_collection.is_visible:
        visible_collections.append(layer_collection)
        for child in layer_collection.children:
            visible_collections.extend(get_visible_collections(child))
    return visible_collections


def get_visible_objects(collection: bpy.types.Collection, filter_names: list[str] = None) -> list[bpy.types.Object]:
    # Get all render-visible objects in all visible collections
    collections = get_visible_collections(collection)
    return [
        obj for col in collections for obj in col.collection.objects
        if (obj.type == 'MESH') and (not obj.hide_render) and (obj.name not in filter_names)
    ]


def get_label(obj: bpy.types.Object) -> str:
    # The label of an object is its name without the Blender duplicate number, e.g., 'Cube.001' has class 'Cube'
    root_name = re.findall(r'(.+)\.\d+$', obj.name)
    if len(root_name) > 0:
        return root_name[0]
    return obj.name


def compute_bvh_tree(
        object_verts_polys: OrderedDict[bpy.types.Object, tuple[np.ndarray, list[list[int]]]]
) -> mathutils.bvhtree.BVHTree:
    # Initialize vertices and polygons
    vertices, polygons = [], []

    # Iterate over all objects and store their vertices and polygons for BVH tree
    for obj, (verts, polys) in object_verts_polys.items():
        # Transform vertices to world coordinates
        world_from_obj = np.array(obj.matrix_world)
        world_verts = world_from_obj[:3, :3] @ verts.T + world_from_obj[:3, 3:]

        # Extend vertices and polygons
        vertices.extend(list(map(mathutils.Vector, world_verts.T)))
        polygons.extend(polys)

    return mathutils.bvhtree.BVHTree.FromPolygons(vertices, polygons, epsilon=0.0)


def batch_raycast(
        origin: mathutils.Vector,
        points: np.ndarray,
        ray_directions: np.ndarray,
        ray_lengths: np.ndarray,
        static_bvh: mathutils.bvhtree.BVHTree,
        dynamic_bvh: mathutils.bvhtree.BVHTree
):
    """Optimized function to check occlusion using BVH raycasting."""
    n_pts = len(points)
    occluded = np.zeros(n_pts, dtype=bool)

    for i in range(n_pts):
        point = mathutils.Vector(points[i])
        ray_direction = mathutils.Vector(ray_directions[i])
        ray_length = ray_lengths[i].item()

        # Check collision with static BVH
        collision_location, _, _, _ = static_bvh.ray_cast(origin, ray_direction, ray_length)
        if collision_location and (point - collision_location).length > 0.01:
            occluded[i] = True
            continue  # Skip dynamic check if already occluded

        # Check collision with dynamic BVH
        collision_location, _, _, _ = dynamic_bvh.ray_cast(origin, ray_direction, ray_length)
        if collision_location and (point - collision_location).length > 0.01:
            occluded[i] = True

    return occluded
