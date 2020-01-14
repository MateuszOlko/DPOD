import numpy as np
from DPOD.models_handler import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(666)

models_handler = ModelsHandler('data/kaggle')
model_ids = [5, 5, 9]
translation_vectors = [
    np.array([5, 1, 25]),
    np.array([4, 1, 10]),
    np.array([-2, 1, 8]),
]
rotation_matrices = [
    euler_to_Rot(0, 0, 0),
    euler_to_Rot(0, 0, 0),
    euler_to_Rot(0, 0, 0)
]

img = np.zeros((2710, 3384, 3), dtype=np.uint8)
for model_id, translation_vector, rotation_matrix in zip(model_ids, translation_vectors, rotation_matrices):
    img = models_handler.draw_model(img, model_id, translation_vector, rotation_matrix)


result = pnp_ransac_multiple_instances(img[..., 1], img[..., 2], img[..., 0], models_handler, 0, min_inliers=1000)
from pprint import pprint
pprint(result)