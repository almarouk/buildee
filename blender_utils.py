from collections import OrderedDict

import bpy
import mathutils
import numpy as np
import re


def has_animation_data(obj: bpy.types.Object) -> bool:
    if obj.animation_data is not None:
        return (obj.animation_data.action is not None) or (obj.animation_data.drivers is not None)
    return False


def is_animated(obj: bpy.types.Object) -> bool:
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


def random_colors_without_replacement(num_colors: int) -> np.ndarray:
    # Generate random colors without replacement
    for _ in range(100):
        colors = np.random.randint(0, 256, (num_colors, 3), dtype=np.uint8)
        if len(np.unique(colors, axis=0)) == num_colors:
            return colors
    raise ValueError('Could not generate unique random colors')
