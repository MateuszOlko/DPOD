import os
from glob import glob
import json
import numpy as np
import cv2
from DPOD.apolloscape_specs import car_id2name, car_name2id
from math import sin, cos
import matplotlib.pyplot as plt


class ModelsHandler:
    def __init__(self, kaggle_dataset_dir_path):
        self.raw_models = {
            os.path.split(model_path)[-1][:-5]: json.load(open(model_path))
            for model_path in glob(f'{kaggle_dataset_dir_path}/car_models_json/*.json')
        }

        # k is camera instrinsic matrix
        self.k = np.array([
            [2304.5479, 0, 1686.2379],
            [0, 2305.8757, 1354.9849],
            [0, 0, 1]
        ], dtype=np.float32)

    def model_id_to_vertices_and_triangles(self, model_id):
        if model_id in self.raw_models:
            data = self.raw_models[model_id]
        elif model_id in car_id2name:
            data = self.raw_models[car_id2name[model_id]]
        else:
            raise KeyError

        vertices = np.array(data['vertices'])
        vertices[:, 1] = -vertices[:, 1]
        triangles = np.array(data['faces']) - 1
        return vertices, triangles

    @staticmethod
    def normalize_to_256(x):
        return 256 * (x - x.min()) / (x.max() - x.min()) % 256

    @staticmethod
    def _draw_obj(image, vertices, triangles, color=(0, 0, 255), colors=None):
        if colors is not None:
            for n, t in enumerate(triangles):
                coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
                cv2.fillConvexPoly(image, coord, colors[n])
        else:
            for t in triangles:
                coord = np.array([vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]], dtype=np.int32)
                cv2.fillConvexPoly(image, coord, color)

    def project_points(self, points, rodrigues_rotation_vector, translation_vector):
        pass

    def draw_model(
            self, img, model_id, translation_vector, rotation_matrix,
            skip_class_mask=False,
            skip_uv_mask=False
    ):
        '''
        draw model identified by model_in (string or int) onto img using coloring
        (class_mask, height_mask, angle_mask)
        '''
        x, y, z = translation_vector

        Rt = np.eye(4)
        t = np.array([x, y, z]).astype(float)
        Rt[:3, 3] = t
        Rt[:3, :3] = rotation_matrix.T  # transposition is due to coordinate system change
        Rt = Rt[:3, :]
        vertices, triangles = self.model_id_to_vertices_and_triangles(model_id)
        P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
        P[:, :-1] = vertices
        P = P.T

        img_cor_points = np.dot(Rt, P)
        points3d = img_cor_points.T
        img_cor_points = np.dot(self.k, img_cor_points)
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]

        img_cor_points = cv2.projectPoints(
            vertices,
            cv2.Rodrigues(rotation_matrix.T)[0],
            np.array(translation_vector),
            self.k,
            None
        )[0][:, 0, :]

        # todo: jescze points3d opencv a nie kagglowym bodgem

        if not skip_class_mask:
            color = model_id if isinstance(model_id, int) else car_name2id[model_id]
            self._draw_obj(img, img_cor_points, triangles, color=(color, 0, 0))

        if not skip_uv_mask:
            faces_mid_points_on_model = np.array([
                (vertices[t1] + vertices[t2] + vertices[t3]) / 3 for t1, t2, t3 in triangles])
            faces_mid_points_in_reality = np.array(
                [(points3d[t1] + points3d[t2] + points3d[t3]) / 3 for t1, t2, t3 in triangles])

            # face_ordering = np.argsort(-faces_mid_points_in_reality[:, 1]) # draw faces on each model from bottom
            face_ordering = np.argsort(-faces_mid_points_in_reality[:, 2])  # draw faces on each model from front

            # these are faces colours

            h_colors = faces_mid_points_on_model[:, 1]
            r_colors = np.arctan2(*faces_mid_points_on_model[:, [0, 2]].T)
            h_colors = self.normalize_to_256(h_colors)
            r_colors = self.normalize_to_256(r_colors)
            colors = np.array([(0, hc, rc) for hc, rc in zip(h_colors, r_colors)]).astype(int)

            triangles = triangles[face_ordering]
            colors = colors[face_ordering].tolist()

            self._draw_obj(img, img_cor_points, triangles, colors=colors)

        return img

    @staticmethod
    def _euler_to_Rot(yaw, pitch, roll):
        Y = np.array([
            [cos(yaw), 0, sin(yaw)],
            [0, 1, 0],
            [-sin(yaw), 0, cos(yaw)]
        ])
        P = np.array([
            [1, 0, 0],
            [0, cos(pitch), -sin(pitch)],
            [0, sin(pitch), cos(pitch)]
        ])
        R = np.array([
            [cos(roll), -sin(roll), 0],
            [sin(roll), cos(roll), 0],
            [0, 0, 1]
        ])
        return np.dot(Y, np.dot(P, R))

    def draw_kaggle_model(self, img, model_type, kaggle_yaw, kaggle_pitch, kaggle_roll, x, y, z, **kwargs):
        model_type = int(model_type)
        kaggle_yaw   = float(kaggle_yaw)
        kaggle_pitch = float(kaggle_pitch)
        kaggle_roll  = float(kaggle_roll)
        x, y, z = float(x), float(y), float(z)

        yaw, pitch, roll = -kaggle_pitch, -kaggle_yaw, -kaggle_roll
        rotation_matrix = self._euler_to_Rot(yaw, pitch, roll)
        self.draw_model(img, model_type, (x, y, z), rotation_matrix, **kwargs)

    def draw_kaggle_models(self, img, model_types, kaggle_yaws, kaggle_pitches, kaggle_rolls, xs, ys, zs, **kwargs):
        for foo in sorted(zip(model_types, kaggle_yaws, kaggle_pitches, kaggle_rolls, xs, ys, zs), key=lambda foo: foo[-1], reverse=True):
            self.draw_kaggle_model(img, *foo, **kwargs)

    def draw_kaggle_models_from_kaggle_string(self, img, kaggle_string, **kwargs):
        items = kaggle_string.split(' ')
        model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
        self.draw_kaggle_models(img, model_types, yaws, pitches, rolls, xs, ys, zs, **kwargs)

    def make_mask_from_kaggle_string(self, kaggle_string, img=None):
        # img is only for copying resolution
        retval = np.zeros(img.shape[:2]+[3]) if img else np.zeros([2710, 3384, 3])
        retval[:, :, 0] = -1
        self.draw_kaggle_models_from_kaggle_string(retval, kaggle_string)
        return retval

    def show_visualizations(self, img, mask):
        model_type_mask = mask[:, :, 0]
        height_mask     = mask[:, :, 1]
        angle_mask      = mask[:, :, 2]

        no_car_black_mask = np.zeros(model_type_mask.shape + (4,), dtype=np.uint8)  # this is RGBA
        no_car_black_mask[:, :, 3] = (model_type_mask == -1)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].imshow(img)
        # axs[0, 0].set_title('Axis [0,0]')

        axs[0, 1].imshow(model_type_mask)
        axs[0, 1].imshow(no_car_black_mask)
        # axs[0, 1].set_title('Axis [0,1]')

        axs[1, 0].imshow(height_mask)
        axs[1, 0].imshow(no_car_black_mask)
        # axs[1, 0].set_title('Axis [1,0]')

        axs[1, 1].imshow(angle_mask, cmap='hsv')
        axs[1, 1].imshow(no_car_black_mask)
        # axs[1, 1].set_title('Axis [1,1]')

        plt.show()
