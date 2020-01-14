import numpy as np
from DPOD.models_handler import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(666)

models_handler = ModelsHandler('data/kaggle')
model_ids = [5, 5, 5]
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

plt.imshow(img); plt.show()

result = models_handler.pnp_ransac_single_instance(img[..., 1], img[..., 2], img[..., 0]!=0, 5)
success, ransac_rotation_matrix, ransac_translation_vector, inliers = result

img[inliers[:, 0], inliers[:, 1]] = np.array([255, 0, 0])
plt.imshow(img); plt.show()

renderer_instance = models_handler.draw_model(np.zeros_like(img)-1, 5, ransac_translation_vector, ransac_rotation_matrix)
instance_present_mask = np.nonzero(renderer_instance[..., 0] == 5)
img[instance_present_mask[0], instance_present_mask[1], 0] = 0

result = models_handler.pnp_ransac_single_instance(img[..., 1], img[..., 2], img[..., 0]==5, 5)
success, ransac_rotation_matrix, ransac_translation_vector, inliers = result

img[inliers[:, 0], inliers[:, 1]] = np.array([255, 0, 255])
plt.imshow(img); plt.show()

renderer_instance = models_handler.draw_model(np.zeros_like(img)-1, 5, ransac_translation_vector, ransac_rotation_matrix)
instance_present_mask = np.nonzero(renderer_instance[..., 0] == 5)
img[instance_present_mask[0], instance_present_mask[1], 0] = 0

result = models_handler.pnp_ransac_single_instance(img[..., 1], img[..., 2], img[..., 0]==5, 5)
success, ransac_rotation_matrix, ransac_translation_vector, inliers = result

img[inliers[:, 0], inliers[:, 1]] = np.array([255, 0, 255])
plt.imshow(img); plt.show()


print(ransac_translation_vector, sep='\n')
print(ransac_rotation_matrix, sep='\n')