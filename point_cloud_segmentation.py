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

    # Setup depth colormap for display
    depth_color_map = plt.get_cmap('magma')
    max_depth_distance_display = 10.0

    # Setup labels colormap for display
    seg_color_map = plt.get_cmap('jet')

    # Setup voxel size
    voxel_size = 0.1

    # Initialize voxel grid and labels
    voxels = np.zeros((0, 3), dtype=np.int32)
    voxel_labels = np.zeros(0, dtype=np.int32)

    for _ in range(100):
        # Render image
        rgb, depth, labels = simulator.render()

        # Unproject depth to world points
        world_points = simulator.depth_to_world_points(depth=depth)

        # Get indices of valid points and known labels
        valid = ~np.any(np.isnan(world_points), axis=2) & (labels > -1)

        # Transform world points to voxels
        view_voxels = np.int32(np.floor(world_points[valid] / voxel_size))

        # Add view voxels and labels to voxel grid
        voxels = np.vstack([voxels, view_voxels])
        voxel_labels = np.hstack([voxel_labels, labels[valid]])

        # Remove duplicates
        voxels, idxs = np.unique(voxels, return_index=True, axis=0)
        voxel_labels = voxel_labels[idxs]

        # Setup depth for display
        depth = depth_color_map(
            depth.clip(0, max_depth_distance_display) / max_depth_distance_display
        )

        # Setup labels for display
        labels = seg_color_map((labels + 1) / len(simulator.labels))

        # Show rgb, depth and point cloud
        cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'depth', cv2.cvtColor(np.uint8(depth * 255), cv2.COLOR_RGB2BGR))
        cv2.imshow(f'segmentation', cv2.cvtColor(np.uint8(labels * 255), cv2.COLOR_RGB2BGR))
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
    point_cloud.colors = o3d.utility.Vector3dVector(seg_color_map((voxel_labels + 1) / len(simulator.labels))[:, :3])

    # Visualize point cloud
    o3d.visualization.draw_geometries([point_cloud])
