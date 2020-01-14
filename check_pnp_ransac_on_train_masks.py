from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from DPOD.models_handler import ModelsHandler
import numpy as np
import cv2
import matplotlib.pyplot as plt

models_handler = ModelsHandler('data/kaggle')

fav_n = 2526
dataset = KaggleImageMaskDataset('data/kaggle')
image, (cls_mask, h_mask, angle_mask), predstr = dataset[fav_n]
print(predstr)
singlestr = ' '.join(predstr.split(' ')[:7])
print(singlestr)

image = image.numpy().transpose(1, 2, 0)
image = (image - image.min()) / (image.max() - image.min())
image = (256*image).astype(np.uint8)

puste = np.zeros((2710, 3384, 3), dtype=np.uint8)
models_handler.draw_kaggle_models_from_kaggle_string(puste, singlestr)
plt.imshow(puste); plt.show()

cls_mask = cls_mask.numpy()
h_mask = h_mask.numpy()
angle_mask = angle_mask.numpy()

mask = cls_mask == 70
image[mask] = np.zeros(3)
image[mask, 2] = h_mask[mask]
image[mask, 0] = angle_mask[mask]

plt.imshow(image)
plt.show()

converged, inliers_positions, trans, rot = \
    models_handler.pnp_ransac_single_instance2(h_mask, angle_mask, mask, 70)

print(trans)

models_handler.draw_model(puste, 70, trans[0], cv2.Rodrigues(rot)[0])
plt.imshow(puste); plt.show()

exit(0)
