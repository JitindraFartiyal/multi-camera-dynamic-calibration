import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(pose1, pose2):
    rotation_similarity = 0.
    translation_error = 0.

    for i in range(3):
        x1 = pose1[:, i:i + 1].squeeze(1)
        x2 = pose2[:, i:i + 1].squeeze(1)
        rotation_similarity += np.dot(x1, x2)/(norm(x1)*norm(x2))

    rotation_similarity /= 3
    translation_error = np.linalg.norm(pose1[:, 2:3] - pose2[:, 2:3])

    return rotation_similarity, translation_error
