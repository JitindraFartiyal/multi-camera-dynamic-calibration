import cv2 as cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import kornia as K
import kornia.feature as KF
from SuperGlue.superglue_match import keypoint_match, make_matching_plot


def find_blob(kp, blob_size, img_size):
    '''
        topleft, top right, bottomleft, bottomright
    :param kp:
    :param blob_size:
    :param img_size: W x H; cols x width
    :return:
    '''
    centroid = np.average(kp, 0) # W x H
    centroid = centroid[::-1] # H x W
    half_width = (img_size[0] * blob_size) / 2
    half_height = (img_size[1] * blob_size) / 2

    # rows x cols; H x W
    x1, y1 = int(centroid[0] - half_height), int(centroid[1] - half_width)
    x2, y2 = int(centroid[0] - half_height), int(centroid[1] + half_width)
    x3, y3 = int(centroid[0] + half_height), int(centroid[1] - half_width)
    x4, y4 = int(centroid[0] + half_height), int(centroid[1] + half_width)

    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if x3 > img_size[1]:
        x3 = img_size[1]
    if x4 > img_size[1]:
        x4 = img_size[1]

    if y1 < 0:
        y1 = 0
    if y2 > img_size[0]:
        y2 = img_size[0]
    if y3 < 0:
        y3 = 0
    if y4 > img_size[0]:
        y4 = img_size[0]

    size_of_bbox = [x3-x1, y2-y1]

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] # rows x cols


def extract_blob(img, blob_bbox):
    return img[blob_bbox[0][0]: blob_bbox[2][0], blob_bbox[0][1]: blob_bbox[1][1]] # H X W


def sift_keypoint_matching(img1_path, img2_path, blob_matching=False, blob_size=0.5, viz_blob=False):
    """
        Use SIFT keypoint detection and BF matcher to match
        keypoints in two images
    :param img1_path: image1 path
    :param img2_path: image2 path
    return matched keypoints
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)


    if blob_matching or viz_blob:
        blob_bbox1 = find_blob(pts1, blob_size, img1.shape)
        blob_bbox2 = find_blob(pts2, blob_size, img2.shape)

        if blob_matching:
            pass
        if viz_blob:
            img1 = cv2.rectangle(img1, (blob_bbox1[0]), (blob_bbox1[3]), (0, 255, 0), 3)
            img2 = cv2.rectangle(img2, (blob_bbox2[0]), (blob_bbox2[3]), (0, 255, 0), 3)

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('keypoint_matching_results/matched_images_sift.png', img3)

    return np.array(pts1), np.array(pts2)


def superglue_keypoint_matching(img1_path, img2_path, blob_bbox=None):
    """
        Use Superglue keypoint detection to match
        keypoints in two images
        https://github.com/magicleap/SuperGluePretrainedNetwork
    :param img1_path: image1 path
    :param img2_path: image2 path
    return matched keypoints
    """

    results = keypoint_match(img1_path, img2_path, blob_bbox)
    kpts0 = results['keypoints0']
    kpts1 = results['keypoints1']
    matches = results['matches']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = results['match_confidence'][valid]

    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (640, 480))

    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.resize(img2, (640, 480))

    make_matching_plot(
        img1, img2, kpts0, kpts1, mkpts0, mkpts1, color,
        text, "keypoint_matching_results/matched_images_superglue.png", True,
        True, False, 'Matches', [])

    return mkpts0, mkpts1


def loftr_keypoint_matching(img1_path, img2_path, img_size):
    """
        Use Lofter keypoint detection to match
        keypoints in two images
        https://github.com/zju3dv/LoFTR
    :param img1_path: image1 path
    :param img2_path: image2 path
    :param img_size: size of the image (W X H)
    return matched keypoints
    """
    img1 = K.io.load_image(img1_path, K.io.ImageLoadType.RGB32)[None, ...]
    img2 = K.io.load_image(img2_path, K.io.ImageLoadType.RGB32)[None, ...]

    img1 = K.geometry.resize(img1, (img_size), antialias=True)
    img2 = K.geometry.resize(img2, (img_size), antialias=True)

    matcher = KF.LoFTR(pretrained="indoor_new")

    input_dict = {
        "image0": K.color.rgb_to_grayscale(img1),  # LofTR works on grayscale images only
        "image1": K.color.rgb_to_grayscale(img2),
    }

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()

    img1 = cv2.imread(img1_path)
    img1 = cv2.resize(img1, (640, 480))

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (640, 480))
    img3 = np.concatenate((img1, img2), axis=1)

    set_color = lambda x: x + random.randint(0, 255)
    for (pt0, pt1) in zip(mkpts0, mkpts1):
        r, g, b = 0, 0, 0
        r, g, b = set_color(r), set_color(g), set_color(b)
        x1, y1 = int(pt0[0]), int(pt0[1])
        x2, y2 = int(pt1[0] + img1.shape[1]), int(pt1[1])
        cv2.line(img3, (x1, y1), (x2, y2), (r, g, b), 2)

    cv2.imwrite('keypoint_matching_results/matched_images_lofter.png', img3)

    return mkpts0, mkpts1


if __name__ == '__main__':
    img1_path = './data/rgb_0/000000.png'
    img2_path = './data/rgb_1/000000.png'
    blob_size = 0.5
    img_size = (1280, 720) # W X H; rows = 720, cols = 1280

    # pts0, pts1 = sift_keypoint_matching(img1_path, img2_path, viz_blob=True)
    # pts0, pts1 = superglue_keypoint_matching(img1_path, img2_path)
    pts0, pts1 = loftr_keypoint_matching(img1_path, img2_path, img_size)

    # pts0, pts1 = sift_keypoint_matching(img1_path, img2_path, blob_bbox)
    # pts0, pts1 = superglue_keypoint_matching(img1_path, img2_path, blob_bbox)
    # pts0, pts1 = loftr_keypoint_matching(img1_path, img2_path, img_size, blob_bbox)
