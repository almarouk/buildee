import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from simulator import Simulator


def unilateral_chamfer_score(
        pts1: np.ndarray,
        pts2: np.ndarray,
        pts1_labels: np.ndarray,
        pts2_labels: np.ndarray,
        dist_thresh: float,
        visualize: bool = False
) -> float:
    kd = KDTree(pts1)
    dists, nearest_indices = kd.query(pts2)
    nearest_labels = pts1_labels[nearest_indices]
    matching_dists = dists < dist_thresh
    matching_labels = pts2_labels == nearest_labels
    matching_pts = matching_dists & matching_labels
    chamfer_score = np.mean(matching_pts)

    if visualize:
        vis_labels = np.zeros_like(pts2_labels)
        vis_labels[~matching_dists] = 1
        vis_labels[~matching_labels] = 2
        visualize_point_cloud(pts2, vis_labels - 1, 2)

    return chamfer_score.item()


def visualize_point_cloud(pts: np.ndarray, pts_labels: np.ndarray, num_labels: int, cmap: str = 'jet'):
    pcl = o3d.geometry.PointCloud()
    cmap = plt.get_cmap(cmap)
    pcl.points = o3d.utility.Vector3dVector(pts)
    pcl.colors = o3d.utility.Vector3dVector(cmap((pts_labels + 1) / num_labels)[:, :3])
    o3d.visualization.draw_geometries([pcl])


if __name__ == '__main__':
    # Create a simulator
    simulator = Simulator(
        'test.blend',
        points_density=100.0,
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
        labels = seg_color_map((labels + 1) / simulator.n_labels)[:, :, :3]

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

    # Compute estimated point cloud from voxel grid
    estimated_point_cloud = voxels * voxel_size

    # Visualize estimated point cloud
    visualize_point_cloud(estimated_point_cloud, voxel_labels, simulator.n_labels, seg_color_map.name)

    # Get simulator ground truth point cloud
    point_cloud, point_cloud_labels, _ = simulator.get_point_cloud()

    # Visualize ground truth point cloud
    visualize_point_cloud(point_cloud, point_cloud_labels, simulator.n_labels, seg_color_map.name)
