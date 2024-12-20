import bpy
import pycolmap
import mathutils


if __name__ == '__main__':
    reconstruction = pycolmap.Reconstruction('/home/clementin/Data/mipnerf360/bicycle/sparse/0/')

    # Get current scene
    scene = bpy.context.scene

    # Delete all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    sensor_height = 36  # Typical sensor width for a full-frame camera in mm

    for colmap_image in reconstruction.images.values():

        # Get image pose
        w_T_c = colmap_image.cam_from_world.inverse().matrix()
        w_T_c[:, 1:3] *= -1  # rotate 180 degrees around x-axis

        # Create camera object
        bpy.ops.object.camera_add()
        camera = bpy.context.selected_objects[0]

        # Set camera pose
        camera.matrix_world = mathutils.Matrix(w_T_c).to_4x4()

        # Set camera parameters
        colmap_camera = reconstruction.cameras[colmap_image.camera_id]
        sensor_width = sensor_height * colmap_camera.width / colmap_camera.height
        fx, fy, cx, cy = colmap_camera.params
        focal_length_mm_x = (fx * sensor_width) / colmap_camera.width
        focal_length_mm_y = (fy * sensor_height) / colmap_camera.height
        camera.data.lens = focal_length_mm_x
        camera.data.sensor_fit = 'VERTICAL'
        camera.data.sensor_width = sensor_width
        camera.data.sensor_height = sensor_height
        camera.data.shift_x = (cx - colmap_camera.width / 2) / colmap_camera.width
        camera.data.shift_y = (cy - colmap_camera.height / 2) / colmap_camera.height

        # Set camera name
        camera.name = colmap_image.name

        # Set resolution
        scene.render.resolution_x = colmap_camera.width
        scene.render.resolution_y = colmap_camera.height

    # Load point cloud
    mesh = bpy.data.meshes.new(name="PointCloud")

    # Create a new object with this mesh
    obj = bpy.data.objects.new("PointCloud", mesh)

    # Link the object to the current collection
    bpy.context.collection.objects.link(obj)

    # Create vertices from the numpy array
    vertices = [tuple(pt.xyz) for pt in reconstruction.points3D.values()]

    # Add vertices to the mesh
    mesh.from_pydata(vertices, [], [])  # No edges or faces, only vertices

    # Update the mesh
    mesh.update()

    # Save the blender file
    bpy.ops.wm.save_mainfile(filepath='colmap.blend')
