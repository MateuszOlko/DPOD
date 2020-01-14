import numpy as np
from DPOD.models_handler import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(666)

models_handler = ModelsHandler('data/kaggle')
model_id = 5

points, _ = models_handler.model_id_to_vertices_and_triangles(model_id)
translation_vector = np.array([-3, -2, 15])
rotation_matrix = euler_to_Rot(0, 0, 0.7)
rotation_rodrigues_vector = cv2.Rodrigues(rotation_matrix)[0]

downscaling = 8
img = np.zeros((2710, 3384, 3), dtype=np.uint8)
img = models_handler.draw_model(img, model_id, translation_vector, rotation_matrix, 1)
img = cv2.resize(img, tuple(x//downscaling for x in reversed(img.shape[:2])), cv2.INTER_NEAREST)

plt.imshow(img); plt.show()

result = models_handler.pnp_ransac_single_instance(img[..., 1], img[..., 2], img[..., 0] == model_id, model_id, downscaling)
success, ransac_rotation_matrix, ransac_translation_vector, inliers = result

img = np.zeros_like(img)
img = models_handler.draw_model(img, model_id, ransac_translation_vector, ransac_rotation_matrix, downscaling)
plt.imshow(img); plt.show()

print(translation_vector, ransac_translation_vector, sep='\n')
print(rotation_matrix, ransac_rotation_matrix, sep='\n')
