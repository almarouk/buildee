import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from simulator import Simulator


if __name__ == '__main__':
    # Create a simulator
    simulator = Simulator(
        'construction.blend',
        points_density=0.0,
        segmentation_sensitivity=0.99,
        verbose=True
    )

    # Get camera matrix
    camera_matrix = simulator.get_camera_matrix()

    # Setup voxel size
    voxel_size = 0.1

    # Initialize voxel grid and labels
    voxels = np.zeros((0, 3), dtype=np.int32)
    voxel_labels = np.zeros(0, dtype=np.int32)

    # Setup label colors
    label_colors = np.random.rand(len(simulator.labels), 3)

    # Setup depth colormap
    depth_color_map = plt.get_cmap('magma')
    max_depth_distance_display = 10.0

    # Setup segmentation colormap
    seg_color_map = plt.get_cmap('jet')

    for _ in range(100):
        # Render image
        rgb, depth, seg = simulator.render()

        # Get valid depth values
        valid = (depth < (simulator.camera.data.clip_end - 1)) & (seg > -1)

        # Unproject depth to point cloud
        world_from_cam = simulator.get_world_from_camera()
        v, u = np.where(valid)
        uvws = np.stack([u + 0.5, v + 0.5, np.ones_like(u)])
        cam_points = np.linalg.inv(camera_matrix) @ (depth[v, u] * uvws)  # unproject in camera view
        world_points = world_from_cam[:3, :3] @ cam_points + world_from_cam[:3, 3:]  # transform to world view
        voxels, idxs = np.unique(np.vstack([
            voxels, np.int32(np.floor(world_points.T / voxel_size))
        ]), return_index=True, axis=0)
        voxel_labels = np.hstack([voxel_labels, seg[v, u]])[idxs]

        # Setup depth for display
        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        # Setup segmentation for display
        seg = seg_color_map((seg + 1) / len(simulator.labels))

        # Show rgb, depth and point cloud
        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'segmentation', cv2.cvtColor(np.uint8(seg * 255), cv2.COLOR_RGB2BGR))
        cv2.waitKey(7)

        # Move camera randomly
        action = np.random.randint(8)
        match action:
            case 0:
                simulator.move_camera_forward(1)
            case 1:
                simulator.move_camera_forward(-1)
            case 2:
                simulator.move_camera_down(1)
            case 3:
                simulator.move_camera_down(-1)
            case 4:
                simulator.move_camera_right(1)
            case 5:
                simulator.move_camera_right(-1)
            case 6:
                simulator.rotate_camera_yaw(22.5, degrees=True)
            case 7:
                simulator.rotate_camera_yaw(-22.5, degrees=True)

    # Destroy opencv windows
    cv2.destroyAllWindows()

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(voxels * voxel_size)
    point_cloud.colors = o3d.utility.Vector3dVector(label_colors[voxel_labels])

    # Visualize point cloud
    o3d.visualization.draw_geometries([point_cloud])
