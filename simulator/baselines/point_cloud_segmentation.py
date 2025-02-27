import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tqdm
from scipy.spatial import KDTree

from ..simulator import Simulator


def compute_miou(
        gt_pts: np.ndarray,
        est_pts: np.ndarray,
        gt_labels: np.ndarray,
        est_labels: np.ndarray,
        dist_thresh: float,
) -> float:
    """Compute the mIoU between two point clouds labeled with semantic classes.

    :param gt_pts: ground truth point cloud, shape (N, 3)
    :param est_pts: estimated point cloud, shape (M, 3)
    :param gt_labels: labels of ``gt_pts``, shape (N,)
    :param est_labels: labels of ``est_pts``, shape (M,)
    :param dist_thresh: mIoU only considers estimated points that are within ``dist_thresh`` of a ground truth point
    :return: mIoU of ``est_pts`` w.r.t. ``gt_pts``
    """
    # Build KDTree for nearest neighbor search
    kd = KDTree(gt_pts)

    # Get nearest ground truth neighbor and its label for each estimated point
    dists, nearest_gt_indices = kd.query(est_pts)
    nearest_gt_labels = gt_labels[nearest_gt_indices]

    # Filter out estimated points that are not close to any ground truth point
    valid = dists < dist_thresh
    nearest_gt_labels = nearest_gt_labels[valid]
    est_labels = est_labels[valid]

    # Get all unique labels
    all_labels = np.unique(np.hstack([nearest_gt_labels, est_labels]))

    # Compute mIoU
    miou = 0.0
    for label in all_labels:
        tp = np.sum((nearest_gt_labels == label) & (est_labels == label))
        fp = np.sum((nearest_gt_labels != label) & (est_labels == label))
        fn = np.sum((nearest_gt_labels == label) & (est_labels != label))
        miou += tp / (tp + fp + fn)

    return miou / len(all_labels)


def unilateral_chamfer_l1_distance(
        pts1: np.ndarray,
        pts2: np.ndarray
) -> float:
    """Compute the average L1 distance from each point in ``pts2`` to its nearest neighbor in ``pts1``.

    :param pts1: reference point cloud, shape (N, 3)
    :param pts2: point cloud to evaluate, shape (M, 3)
    :return: Chamfer-L1 distance of ``pts2`` w.r.t. ``pts1``
    """
    kd = KDTree(pts1)
    dists, _ = kd.query(pts2, p=1)
    return dists.mean()


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

        # Remove duplicates
        voxels, idxs = np.unique(voxels, return_index=True, axis=0)
        voxel_labels = voxel_labels[idxs]

        # Visualize render
        if visualize:
            # Setup labels for display
            labels = seg_color_map((labels + 1) / simulator.n_labels)[:, :, :3]

            # Show rgb and segmentation map
            cv2.imshow(f'rgb', cv2.cvtColor(np.uint8(rgb * 255), cv2.COLOR_RGB2BGR))
            cv2.imshow(f'segmentation', cv2.cvtColor(np.uint8(labels * 255), cv2.COLOR_RGB2BGR))
            cv2.waitKey(7)

        # Move camera in a random direction
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
                simulator.turn_camera_right(22.5, degrees=True)
            case 7:
                simulator.turn_camera_right(-22.5, degrees=True)

    # Destroy opencv windows
    if visualize:
        cv2.destroyAllWindows()

    # Compute estimated point cloud from voxel grid
    estimated_point_cloud = voxels * voxel_size

    # Visualize estimated point cloud
    if visualize:
        visualize_point_cloud(estimated_point_cloud, voxel_labels, simulator.n_labels, seg_color_map.name)

    # Get simulator ground truth point cloud
    point_cloud, point_cloud_labels, _ = simulator.compute_point_cloud()

    # Visualize ground truth point cloud
    if visualize:
        visualize_point_cloud(point_cloud, point_cloud_labels, simulator.n_labels, seg_color_map.name)

    # Compute chamfer scores
    print('Chamfer score - estimated points w.r.t. ground truth points:', unilateral_chamfer_score(
        pts1=point_cloud,
        pts2=estimated_point_cloud,
        pts1_labels=point_cloud_labels,
        pts2_labels=voxel_labels,
        dist_thresh=2*voxel_size,
        visualize=visualize
    ))
    print('Chamfer score - ground truth points w.r.t. estimated points:', unilateral_chamfer_score(
        pts1=estimated_point_cloud,
        pts2=point_cloud,
        pts1_labels=voxel_labels,
        pts2_labels=point_cloud_labels,
        dist_thresh=2*voxel_size,
        visualize=visualize
    ))

    # Compute Chamfer-L1 distances
    print('Chamfer-L1 distance - estimated points w.r.t. ground truth points:', unilateral_chamfer_l1_distance(
        pts1=point_cloud,
        pts2=estimated_point_cloud
    ))
    print('Chamfer-L1 distance - ground truth points w.r.t. estimated points:', unilateral_chamfer_l1_distance(
        pts1=estimated_point_cloud,
        pts2=point_cloud
    ))

    # Compute mIoU
    print('mIoU:', compute_miou(
        point_cloud, estimated_point_cloud, point_cloud_labels, voxel_labels, dist_thresh=2*voxel_size
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
    parser.add_argument('--num-steps', type=int, default=100, help='number of steps for random walk')
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
