"""
    pose_10: pose of camera 1 relative to camera 0
    pose_rl: pose of right camera relative to left camera
"""

import os
from tqdm import tqdm
import numpy as np
import cv2 as cv2
from keypoint_matching import sift_keypoint_matching, superglue_keypoint_matching, loftr_keypoint_matching
from metrics import get_similarity
from constants import constants


def get_camera_matrix():
    """
        Get intrinsic camera matrix
    :return: camera matrix
    """
    return np.array([[constants.fx, 0, constants.cx],
                     [0, constants.fy, constants.cy],
                     [0, 0, 1]])


def extract_gt(gt_path):
    """
        Extract ground truth of relative pose of camera 1 and 2 wrt to camera 0
    :param gt_path: ground truth file path
    :return: relative pose of camera 1 and 2 wrt to camera 0
    """
    orig_c0_pose = np.array(np.loadtxt(gt_path + 'camera0.txt'))
    orig_c1_pose = np.array(np.loadtxt(gt_path + 'camera1.txt'))
    orig_c2_pose = np.array(np.loadtxt(gt_path + 'camera2.txt'))

    pose_10 = np.matmul(np.linalg.inv(orig_c0_pose), orig_c1_pose)
    pose_20 = np.matmul(np.linalg.inv(orig_c0_pose), orig_c2_pose)

    return pose_10, pose_20


def charuco_keypoint_detection(img_path):
    """
        Finding keypoints in a black and white board using charucoboard and charucodetection
    :param img_path: path of the image that contains a black and a white board
    :return: Return keypoints in the board, their ids and image size
    """
    allCorners = []
    allIds = []
    decimator = 0

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((5, 4), 70, 54, aruco_dict)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

    if len(corners) > 0:
        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(gray, corner,
                             winSize=(3, 3),
                             zeroZone=(-1, -1),
                             criteria=criteria)
        res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
            allCorners.append(res2[1])
            allIds.append(res2[2])
        '''
        img = cv2.aruco.drawDetectedCornersCharuco(image=img,
                                                   charucoCorners=np.array(allCorners).squeeze(2),
                                                   charucoIds=np.array(allIds))
        cv2.imshow("Detected Charuco board", img)
        cv2.waitKey(0)
        '''

    decimator += 1
    imgsize = gray.shape

    return allCorners, allIds, imgsize


def camera_calibrate_marker(world_coord_filepath, calibration_dir, calibration_img_id1, calibration_img_id2,
                            camera_matrix):
    """
        Calibrate camera using marker. A marker is typically a black and white board.
        We use chaurco board and detection to find keypoints in the marker. Once keypoints are detected.
        Given the world coordinates, we interpolate pose using stereo calibration.
    :param world_coord_filepath: filpath of world coordinate
    :param calibration_dir: diretory containing calibration images
    :param calibration_img_id1: calibration image 1
    :param calibration_img_id2: calibration image 2
    :param camera_matrix: intrinsic camera matrix
    :return: relative pose
    """
    imgcoords1, ids1, imgsize1 = charuco_keypoint_detection(calibration_dir + calibration_img_id1)

    imgcoords1 = np.array(imgcoords1).squeeze(2)
    imgcoords1 = imgcoords1.astype('float32')

    imgcoords2, ids2, imgsize2 = charuco_keypoint_detection(calibration_dir + calibration_img_id2, )
    imgcoords2 = np.array(imgcoords2).squeeze(2)
    imgcoords2 = imgcoords2.astype('float32')

    world_coords = np.array(np.loadtxt(world_coord_filepath))
    world_coords = np.expand_dims(world_coords, axis=0)
    world_coords = world_coords.astype('float32')

    res = cv2.stereoCalibrate(world_coords, imgcoords1, imgcoords2, camera_matrix, None, camera_matrix, None, imgsize1,
                              flags=cv2.CALIB_FIX_INTRINSIC)
    relative_rotation = res[5]
    relative_translation = res[6]

    pose_lr = np.hstack((relative_rotation, relative_translation))
    pose_lr = np.vstack((pose_lr, np.array([0., 0., 0., 1.])))
    pose_rl = np.linalg.inv(pose_lr)

    return pose_rl


def camera_calibrate_non_marker(img1_path, img2_path, img_size, camera_matrix, keypoint_matching_algo='SIFT'):
    """
        Calibrate camera without marker. We first find corresponding matches between two images.
        We then use epipolar geometry to retrieve relative rotation and translation.
    :param img_path1: path of image 1
    :param img_path2: path of image 1
    :param img_size: size of the image (W X H)
    :param camera_matrix: instrinsic camera matrix
    :param keypoint_matching_algo: SIFT, SUPERGLUE, LOFTR
    :return: relative pose
    """

    # finding corresponding keypoints using keypoint matching algorithm
    if keypoint_matching_algo == 'SIFT':
        pts1, pts2 = sift_keypoint_matching(img1_path, img2_path, img_size)
    elif keypoint_matching_algo == 'SUPERGLUE':
        pts1, pts2 = superglue_keypoint_matching(img1_path, img2_path, None)
    elif keypoint_matching_algo == 'LOFTR':
        pts1, pts2 = loftr_keypoint_matching(img1_path, img2_path, img_size)
    else:
        print('Keypoint matching algorithm is incorrect')
        return None

    # Find essential matrix from the corresponding keypoints
    E, _ = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.RANSAC, 0.99, 1.0)

    # Finding pose from the essential matrix
    _, relative_rotation, relative_translation, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    # Calculate relative pose
    pose_lr = np.hstack((relative_rotation, relative_translation))
    pose_lr = np.vstack((pose_lr, np.array([0., 0., 0., 1.])))
    pose_rl = np.linalg.inv(pose_lr)

    return pose_rl


def get_marker_based_calibration(world_coord_filepath, calibration_dir,
                                 calibration_img_id1, calibration_img_id2):
    """
        Get calibration using marker based method

        :param world_coord_filepath: filepath of world coordinates
        :param calibration_dir: directory path
        :param calibration_img_id1: calibrated image1
        :param calibration_img_id2: calibrated image2

        :return: relative pose, rotation similarity, translation error
        """
    camera_matrix = get_camera_matrix()
    pose_10, pose_20 = extract_gt(calibration_dir)
    pose_marker = camera_calibrate_marker(world_coord_filepath, calibration_dir, calibration_img_id1,
                                          calibration_img_id2, camera_matrix)
    rotation_similarity, translation_error = get_similarity(pose_marker, pose_10)

    return pose_marker, rotation_similarity, translation_error


def run_non_marker_calibration(img_dir1, img_dir2, img_size, calibration_dir, keypoint_matching_algo):
    """

    :param img_dir1: directory containing img for camera1
    :param img_dir2: directory containing img for camera2
    :param img_size: image size
    :param calibration_dir: calibration directory
    :param keypoint_matching_algo: SIFT, LOFTR, GLUE
    :return: average rotation similarity and translation error
    """
    camera_matrix = get_camera_matrix()
    pose_10, pose_20 = extract_gt(calibration_dir)

    img_ids1 = os.listdir(img_dir1)
    img_ids2 = os.listdir(img_dir2)
    rotation_similarity_list, translation_error_list = [], []

    for img_id1, img_id2 in tqdm(zip(img_ids1, img_ids2)):
        try:
            # non marker based calibration
            pose_non_marker = camera_calibrate_non_marker(img_dir1 + img_id1, img_dir2 + img_id2, img_size, camera_matrix,
                                                          keypoint_matching_algo)

            rotation_similarity, translation_error = get_similarity(pose_non_marker, pose_10)

            rotation_similarity_list.append(rotation_similarity)
            translation_error_list.append(translation_error)
        except:
            continue
    return np.average(np.array(rotation_similarity_list)), np.average(np.array(translation_error_list))


if __name__ == '__main__':
    img_dir1 = "./data/rgb_0/"
    img_dir2 = "./data/rgb_1/"
    img_size = (640, 480) 
    calibration_img_id1 = 'camera_0.png'
    calibration_img_id2 = 'camera_1.png'
    calibration_dir = "./data/Calibration/"
    world_coord_filepath = "./data/Calibration/wo.txt"
    keypoint_matching_algo = "LOFTR" # SIFT, SUPERGLUE, LOFTR

    pose_marker, rotation_similarity, translation_error  = (
        get_marker_based_calibration(world_coord_filepath, calibration_dir, calibration_img_id1, calibration_img_id2))
    print('Marker Pose: \n', pose_marker)
    print("Rotation similarity: ", rotation_similarity, " translation error: ", translation_error)

    avg_rotation_similarity, avg_translation_error = (
        run_non_marker_calibration(img_dir1, img_dir2, img_size, calibration_dir, keypoint_matching_algo))
    print("Average rotation similarity = ", avg_rotation_similarity)
    print("Average translation similairty = ", avg_translation_error)

