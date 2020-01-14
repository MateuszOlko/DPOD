import numpy as np
from DPOD.models_handler import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(666)

models_handler = ModelsHandler('data/kaggle')
model_id = 5

points, _ = models_handler.model_id_to_vertices_and_triangles(model_id)
translation_vector = np.array([-3, -2, 15])
rotation_matrix = np.diag([1, 1, 1]).astype(float)
rotation_matrix = euler_to_Rot(0, 0, 0)
rotation_rodrigues_vector = cv2.Rodrigues(rotation_matrix)[0]

img = np.zeros((2710, 3384, 3), dtype=np.uint8)
img = models_handler.draw_model(img, model_id, translation_vector, rotation_matrix)
plt.imshow(img); plt.show()

pixels_to_consider = np.where(img[..., 0] == model_id)
pixels_to_consider = pixels_to_consider[0], pixels_to_consider[1]
observed_colors = img[pixels_to_consider][:, 1:]
points_implied = models_handler.get_color_to_3dpoints_arrays(model_id)[observed_colors[:, 0], observed_colors[:, 1]]
points_projected = np.stack([pixels_to_consider[1], pixels_to_consider[0]]).T.astype(float)
points = points_implied


result = cv2.solvePnPRansac(points, points_projected, models_handler.camera_matrix, None)
success, ransac_rotataton_rodrigues_vector, ransac_translation_vector, inliers = result
ransac_rotataton_rodrigues_vector = ransac_rotataton_rodrigues_vector.flatten()
ransac_rotation_matrix = cv2.Rodrigues(ransac_rotataton_rodrigues_vector)[0].T
ransac_translation_vector = ransac_translation_vector.flatten()
inliers = inliers.flatten()

print(translation_vector, ransac_translation_vector, sep='\n')
print(rotation_matrix, ransac_rotation_matrix, sep='\n')

img = models_handler.draw_model(img, model_id, ransac_translation_vector, ransac_rotation_matrix)
plt.imshow(img); plt.show()

result = models_handler.pnp_ransac_single_instance2(img[..., 1], img[..., 2], img[..., 0] == model_id, model_id)
success, rotation_matrix, translation_vector, inliers = result
img = np.zeros_like(img)
img = models_handler.draw_model(img, model_id, translation_vector, rotation_matrix)
plt.imshow(img); plt.show()


