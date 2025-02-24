import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tqdm
from scipy.spatial import KDTree

from ..simulator import Simulator


def unilateral_chamfer_score(
        pts1: np.ndarray,
        pts2: np.ndarray,
        pts1_labels: np.ndarray,
        pts2_labels: np.ndarray,
        dist_thresh: float,
        visualize: bool = False
) -> float:
    """Compute the unilateral chamfer score between two point clouds.
    The chamfer score is the percentage of points in ``pts2`` that:
    1) are within ``dist_thresh`` of a point in ``pts1``;
    2) have the same label as the nearest point in ``pts1``.

    :param pts1: reference point cloud, shape (N, 3)
    :param pts2: point cloud to evaluate, shape (M, 3)
    :param pts1_labels: labels of ``pts1``, shape (N,)
    :param pts2_labels: labels of ``pts2``, shape (M,)
    :param dist_thresh: a point in ``pts2`` is considered valid if it is within ``dist_thresh`` of a point in ``pts1``
    :param visualize: 3D visualization of ``pts2`` matching points
    :return: chamfer score of ``pts2`` w.r.t. ``pts1``
    """
    kd = KDTree(pts1)
    dists, nearest_indices = kd.query(pts2)
    nearest_labels = pts1_labels[nearest_indices]
    matching_dists = dists < dist_thresh
    matching_labels = pts2_labels == nearest_labels
    matching_pts = matching_dists & matching_labels
    chamfer_score = np.mean(matching_pts)

    if visualize:
        vis_labels = np.zeros_like(pts2_labels)
        vis_labels[~matching_labels] = 1
        vis_labels[~matching_dists] = 2
        visualize_point_cloud(pts2, vis_labels - 1, 2)

    return chamfer_score.item()


def visualize_point_cloud(pts: np.ndarray, pts_labels: np.ndarray, num_labels: int, cmap: str = 'jet'):
    """Visualize a point cloud with labels using Open3D.

    :param pts: point cloud, shape (N, 3)
    :param pts_labels: labels of points, shape (N,)
    :param num_labels: number of unique labels
    :param cmap: colormap to use for labels
    """
    pcl = o3d.geometry.PointCloud()
    cmap = plt.get_cmap(cmap)
    pcl.points = o3d.utility.Vector3dVector(pts)
    pcl.colors = o3d.utility.Vector3dVector(cmap((pts_labels + 1) / num_labels)[:, :3])
    o3d.visualization.draw_geometries([pcl])


def random_walk(
        blend_file: str,
        points_density: float = 100.0,
        voxel_size: float = 0.1,
        num_steps: int = 100,
        visualize: bool = False
):
    # Create a simulator
    simulator = Simulator(
        blend_file,
        points_density=points_density,
        segmentation_sensitivity=0.99,
        verbose=True
    )

    # Setup labels colormap for display
    seg_color_map = plt.get_cmap('jet')

    # Initialize voxel grid and labels
    voxels = np.zeros((0, 3), dtype=np.int32)
    voxel_labels = np.zeros(0, dtype=np.int32)
    voxel_colors = np.zeros((0, 3), dtype=np.float32)

    print('Start random walk')

    for _ in tqdm.tqdm(range(num_steps)):
        # Render image
        rgb, depth, labels = simulator.render()

        # Unproject depth to 3D world points
        world_points = simulator.depth_to_world_points(depth=depth)

        # Get indices of valid points and known labels
        valid = ~np.any(np.isnan(world_points), axis=2) & (labels > -1)

        # Transform world points to voxels
        view_voxels = np.int32(np.floor(world_points[valid] / voxel_size))

        # Add view voxels and labels to voxel grid
        voxels = np.vstack([voxels, view_voxels])
        voxel_labels = np.hstack([voxel_labels, labels[valid]])
        voxel_colors = np.vstack([voxel_colors, rgb[valid]])

        # Remove duplicates
        voxels, idxs = np.unique(voxels, return_index=True, axis=0)
        voxel_labels = voxel_labels[idxs]
        voxel_colors = voxel_colors[idxs]

        # Visualize render
        if visualize:
            # Setup labels for display
            labels = seg_color_map((labels + 1) / simulator.n_labels)[:, :, :3]

            # Show rgb and segmentation map
            cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
            cv2.imshow(f'segmentation', cv2.cvtColor(np.uint8(labels * 255), cv2.COLOR_RGB2BGR))
            cv2.waitKey(7)

        # Move camera in a random direction
        simulator.step_frame()

    # Destroy opencv windows
    if visualize:
        cv2.destroyAllWindows()

    # Compute estimated point cloud from voxel grid
    estimated_point_cloud = voxels * voxel_size

    label_colors = {
        'Building': np.array([90, 90, 90], dtype=np.uint8),  # Neutral gray
        'Scaffolding': np.array([230, 180, 190], dtype=np.uint8),  # Soft red
        'Plastic barrel': np.array([0, 80, 200], dtype=np.uint8),  # Bright blue
        'CardboardBox': np.array([210, 180, 140], dtype=np.uint8),  # Light brown
        'Wheelbarrow': np.array([150, 40, 50], dtype=np.uint8),  # Rusty brown
        'Boulder': np.array([170, 160, 150], dtype=np.uint8),  # Rock gray
        'Grass': np.array([60, 180, 75], dtype=np.uint8),  # Vibrant green
        'Crane': np.array([255, 165, 0], dtype=np.uint8),  # Construction orange
        'Container': np.array([200, 0, 0], dtype=np.uint8),  # Bold red
        'Steel beam': np.array([128, 128, 128], dtype=np.uint8),  # Steel gray
        'Acrow Prop': np.array([140, 160, 60], dtype=np.uint8),  # Yellow-green for visibility
        'PVC': np.array([50, 50, 50], dtype=np.uint8),  # Almost black
        'Palette': np.array([130, 140, 90], dtype=np.uint8),  # Warm wood tone
        'Terrain': np.array([150, 110, 90], dtype=np.uint8),  # Reddish-brown for contrast
        'Bulldozer': np.array([255, 200, 0], dtype=np.uint8),  # High-visibility yellow
        'WoodenBox': np.array([190, 120, 60], dtype=np.uint8),  # Darker wood tone
        'Toilet': np.array([80, 160, 220], dtype=np.uint8),  # Light blue
        'Fence': np.array([30, 150, 120], dtype=np.uint8)  # Teal green
    }

    colors = voxel_colors.copy()
    for label in np.unique(voxel_labels):
        # label_color = np.percentile(voxel_colors[voxel_labels == label], 80, axis=0)
        label_color = label_colors[simulator.labels[label]] / 255
        colors[voxel_labels == label] = label_color

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(estimated_point_cloud)
    pcl.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcl])
    o3d.io.write_point_cloud('pcl.ply', pcl)

    # Visualize estimated point cloud
    if visualize:
        visualize_point_cloud(estimated_point_cloud, voxel_labels, simulator.n_labels, seg_color_map.name)

    # Get simulator ground truth point cloud
    point_cloud, point_cloud_labels, _ = simulator.compute_point_cloud()

    # Visualize ground truth point cloud
    if visualize:
        visualize_point_cloud(point_cloud, point_cloud_labels, simulator.n_labels, seg_color_map.name)

    # Compute chamfer scores
    print('Chamfer score - estimated points to ground truth points:', unilateral_chamfer_score(
        point_cloud, estimated_point_cloud, point_cloud_labels, voxel_labels, 0.2, visualize=visualize
    ))
    print('Chamfer score - ground truth points to estimated points:', unilateral_chamfer_score(
        estimated_point_cloud, point_cloud, voxel_labels, point_cloud_labels, 0.2, visualize=visualize
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Random walk in a Blender scene, estimate a 3D point cloud and semantic labels for each point, '
                    'compute chamfer scores.'
    )
    parser.add_argument('--blend-file', type=str, required=True, help='path to Blender file')
    parser.add_argument(
        '--points-density', type=float, default=100.0,
        help='ground truth point cloud density'
    )
    parser.add_argument(
        '--voxel-size', type=float, default=0.1,
        help='estimated point cloud voxel grid size'
    )
    parser.add_argument('--num-steps', type=int, default=100-30+1, help='number of steps for random walk')
    parser.add_argument(
        '--visualize', action='store_true',
        help='visualize rendered images, estimated and ground truth point clouds, and chamfer scores'
    )
    args = parser.parse_args()
    random_walk(
        blend_file=args.blend_file,
        points_density=args.points_density,
        voxel_size=args.voxel_size,
        num_steps=args.num_steps,
        visualize=args.visualize
    )
