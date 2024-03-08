import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import treg


def get_open3d_camera_matrix():
    """
        Get intrinsic camera matrix
    :return: camera matrix
    """
    width = 1280
    height = 720
    fx = 607.2491455078125
    fy = 607.165283203125
    cx = 639.1669921875
    cy = 364.76153564453125
    camera_matrix = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    return camera_matrix


def get_rgbd_images(img_path, depth_img_path):
    img = o3d.io.read_image(img_path)
    depth_img = o3d.io.read_image(depth_img_path)
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth_img)

    return rgbd_img


def convert_rgbd_point_cloud(rgbd_img, camera_matrix):
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, camera_matrix)
    return point_cloud



def draw_registration_result(source, target, transformation):
    source.transform(transformation)

    # This is patched version for tutorial rendering.
    # Use `draw` function for you application.
    o3d.visualization.draw_geometries(
        [source, target],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])


if __name__ == '__main__':
    img1_path = 'data/rgb_0/000000.png'
    img1_depth_path = 'data/depth_0/000000.png'

    img2_path = 'data/rgb_0/000001.png'
    img2_depth_path = 'data/depth_0/000001.png'

    camera_matrix = get_open3d_camera_matrix()
    rgbd_img1 = get_rgbd_images(img1_path, img1_depth_path)
    source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img1, camera_matrix)

    rgbd_img2 = get_rgbd_images(img2_path, img2_depth_path)
    target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img2, camera_matrix)

    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 0.07

    # Initial alignment or source to target transform.
    init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                        [-0.139, 0.967, -0.215, 0.7],
                                        [0.487, 0.255, 0.835, -1.4],
                                        [0.0, 0.0, 0.0, 1.0]])

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPlane()

    # Convergence-Criteria for Vanilla ICP
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                           relative_rmse=0.000001,
                                           max_iteration=50)
    # Down-sampling voxel-size.
    voxel_size = 0.025

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True

    s = time.time()

    callback_after_iteration = lambda updated_result_dict: print(
        "Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
            updated_result_dict["iteration_index"].item(),
            updated_result_dict["fitness"].item(),
            updated_result_dict["inlier_rmse"].item()))

    registration_icp = treg.icp(source, target, max_correspondence_distance,
                                init_source_to_target, estimation, criteria,
                                voxel_size, callback_after_iteration)

    icp_time = time.time() - s
    print("Time taken by ICP: ", icp_time)
    print("Inlier Fitness: ", registration_icp.fitness)
    print("Inlier RMSE: ", registration_icp.inlier_rmse)

    draw_registration_result(source, target, registration_icp.transformation)