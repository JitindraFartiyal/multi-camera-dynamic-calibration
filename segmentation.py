import sys
import os
import cv2
import numpy as np
from tqdm import tqdm

from metrics import get_similarity

sys.path.append("./FastSAM")
from calibration import get_camera_matrix, extract_gt
from FastSAM import Inference
from image_similarity_measures.quality_metrics import ssim, fsim


def top_bboxes(bbox, object_conf=0.6):
    """

    :param bbox:
    :param object_conf:
    :return:
    """
    bboxes = []
    for bbox in bbox.data:
        if bbox[4] > object_conf:
            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def draw_bboxes(bboxes, img_path):
    img = cv2.imread(img_path)
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
    cv2.imwrite(img_path, img)


def get_centroid(bboxes, img_path):
    centroids = []
    img = cv2.imread(img_path)
    for bbox in bboxes:
        centroid_x = ((bbox[2] - bbox[0]) // 2) + bbox[0]
        centroid_y = ((bbox[3] - bbox[1]) // 2) + bbox[1]
        centroids.append([centroid_x, centroid_y])
        cv2.circle(img, (centroid_x, centroid_y), 2, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(img_path, img)
    return centroids


def get_similar_objects(img1, img2, bboxes1, bboxes2, similarity_score_conf=0.6):
    similar_objects_idxs = []
    for i in range(len(bboxes1)):
        max_conf = 0.0
        pos1, pos2 = None, None
        for j in range(len(bboxes2)):

            cropped_img1 = img1[bboxes1[i][1]: bboxes1[i][3], bboxes1[i][0]: bboxes1[i][2]]
            cropped_img2 = img2[bboxes2[j][1]: bboxes2[j][3], bboxes2[j][0]: bboxes2[j][2]]
            min_x = min(cropped_img1.shape[0], cropped_img2.shape[0])
            min_y = min(cropped_img1.shape[1], cropped_img2.shape[1])

            if min_x > 7 and min_y > 7:
                scaled_cropped_img1 = cv2.resize(cropped_img1, (min_x, min_y))
                scaled_cropped_img2 = cv2.resize(cropped_img2, (min_x, min_y))

                similarity_score = similarity_matches(scaled_cropped_img1, scaled_cropped_img2)
                if similarity_score >= similarity_score_conf and similarity_score > max_conf:
                    max_conf = similarity_score
                    pos1, pos2 = i, j
        if pos1 is not None:
            similar_objects_idxs.append([pos1, pos2, max_conf])

    del_idxs = []
    for i in range(len(similar_objects_idxs)):
        y = similar_objects_idxs[i][1]
        conf = similar_objects_idxs[i][2]
        max_conf = 0
        drop_idxs = []
        for j in range(i+1, len(similar_objects_idxs)):
            if y == similar_objects_idxs[j][1]:
                if (similar_objects_idxs[j][2] > conf and
                        similar_objects_idxs[j][2] > max_conf):
                    max_conf = similar_objects_idxs[j][2]
                else:
                    if j not in del_idxs:
                        drop_idxs.append(j)
        if max_conf > conf:
            if i not in del_idxs:
                del_idxs.append(i)
        for idx in drop_idxs:
            del_idxs.append(idx)
    similar_objects_idxs = np.array(similar_objects_idxs)
    similar_objects_idxs = list(np.delete(similar_objects_idxs, del_idxs, axis=0))
    similar_objects_idxs = [[int(obj[0]), int(obj[1]), round(obj[2], 4)]
                            for obj in similar_objects_idxs]

    return similar_objects_idxs


def get_keypoints(centroids1, centroids2, similar_objects_idxs):
    pts1, pts2 = [], []
    for idx in similar_objects_idxs:
        pts1.append(np.array(centroids1[idx[0]]))
        pts2.append(np.array(centroids2[idx[1]]))

    return np.array(pts1), np.array(pts2)


def get_pose(pts1, pts2):
    # Find essential matrix from the corresponding keypoints
    E, _ = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.RANSAC, 0.99, 1.0)

    # Finding pose from the essential matrix
    _, relative_rotation, relative_translation, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    # Calculate relative pose
    pose_lr = np.hstack((relative_rotation, relative_translation))
    pose_lr = np.vstack((pose_lr, np.array([0., 0., 0., 1.])))
    pose_rl = np.linalg.inv(pose_lr)

    return pose_rl


def similarity_matches(img1, img2, metrics="ssim"):
    if metrics == "ssim":
        return ssim(img1, img2)
    else:
        return fsim(img1, pred_img=img2)


def run_segmentation_calibration(calibration_dir, img_dir1, img_dir2, similar_matches_path,
        segmented_output_path1, segmented_output_path2):

    rotation_similarity_list, translation_error_list = [], []
    camera_matrix = get_camera_matrix()

    img_ids1 = os.listdir(img_dir1)
    img_ids2 = os.listdir(img_dir2)

    for img_id1, img_id2 in tqdm(zip(img_ids1, img_ids2)):
        try:
            bbox1 = Inference.run_segmentation(img_dir1 + img_id1, segmented_output_path1, device="cpu")
            bbox2 = Inference.run_segmentation(img_dir2 + img_id2, segmented_output_path2, device="cpu")

            bboxes1 = top_bboxes(bbox1)
            bboxes2 = top_bboxes(bbox2)

            draw_bboxes(bboxes1, segmented_output_path1 + img_id1)
            draw_bboxes(bboxes2, segmented_output_path2 + img_id2)

            centroids1 = get_centroid(bboxes1, segmented_output_path1 + img_id1)
            centroids2 = get_centroid(bboxes2, segmented_output_path2 + img_id2)

            img1 = cv2.imread(img_dir1 + img_id1)
            img2 = cv2.imread(img_dir2 + img_id2)

            similar_matches_dir = similar_matches_path + img_id1.split(".")[0]
            if not os.path.exists(similar_matches_dir):
                os.makedirs(similar_matches_dir)

            similar_objects_idxs = get_similar_objects(img1, img2, bboxes1, bboxes2)
            for i, idx in enumerate(similar_objects_idxs):
                bbox1 = bboxes1[idx[0]]
                bbox2 = bboxes2[idx[1]]
                matching_img1 = img1.copy()
                matching_img2 = img2.copy()
                cv2.rectangle(matching_img1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]), color=(0, 255, 0), thickness=2)
                cv2.rectangle(matching_img2, (bbox2[0], bbox2[1]), (bbox2[2], bbox2[3]), color=(0, 0, 255), thickness=2)
                concat_matching_imgs = cv2.hconcat([matching_img1, matching_img2])
                cv2.imwrite(similar_matches_dir + "/sm_" + str(i) + ".png",
                            concat_matching_imgs)
            pts1, pts2 = get_keypoints(centroids1, centroids2, similar_objects_idxs)
            if len(pts1) >= 8:
                pose = get_pose(pts1, pts2)
                pose_10, pose_20 = extract_gt(calibration_dir)
                rotation_similarity, translation_error = get_similarity(pose, pose_10)

                rotation_similarity_list.append(rotation_similarity)
                translation_error_list.append(translation_error)
            else:
                print("Similarity matches failed")
        except:
            continue
    return np.average(np.array(rotation_similarity_list)), np.average(np.array(translation_error_list))


if __name__ == "__main__":
    calibration_dir = "./data/Calibration/"
    camera_matrix = get_camera_matrix()
    # compare all 17 X 17 objects using SIFT/SUPERGLUE/LOFTR and then get corresponding pts
    # check if there are high confidence keypoint matches
    img_dir1 = "./data/rgb_0/"
    img_dir2 = "./data/rgb_2/"

    similar_matches_path = "./similar_matches/"

    segmented_output_path1 = "./segmented/rgb_0/"
    segmented_output_path2 = "./segmented/rgb_2/"

    avg_rotation_similarity, avg_translation_error = run_segmentation_calibration(calibration_dir, img_dir1, img_dir2, similar_matches_path,
        segmented_output_path1, segmented_output_path2)

    print("Average rotation similarity = ", avg_rotation_similarity)
    print("Average translation similairty = ", avg_translation_error)
