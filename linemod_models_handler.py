import numpy as np
import cv2
from glob import glob
import os
from functools import lru_cache
from tqdm import tqdm


def read_obj_file(path):
    """
    reads .obj file from provided dataset

    vertices: (n_vertices, 3) float array with 3d coordinates of vertices
    faces:    (n_faces, 3)    int   array with indices of model faces
    """
    with open(path) as file:
        vertices = []
        faces = []
        for line in file:
            if line[0] == 'v':
                x, y, z = map(float, line.split(' ')[1:4])
                vertices.append((x, y, z))
            if line[0] == 'f':
                v1, v2, v3 = map(int, map(lambda foo: foo.split('//')[0], line.split(' ')[1:]))
                faces.append((v1-1, v2-1, v3-1))

        vertices = np.array(vertices)
        faces = np.array(faces)

        return vertices, faces


def read_position_file(path):
    """
    reads position (poses/<model_name>/info_<image_id>.txt) file from provided dataset

    image_size: (h, w) int tuple
    model_id: str - this is sometimes number and sometimes name
    rotation_matrix: (3, 3) float array
    center: (3,) float array - position of model center in meters
    extend: (3,) float array - I don't know what it is
    """
    with open(path) as file:
        lines = file.readlines()
        image_size = tuple(map(int, lines[1].split(' ')))
        model_id = lines[2]
        if 'Holepuncher' in path:
            rotation_matrix = np.array([[float(x) for x in line.split(' ')] for line in lines[6:9]])
            center = np.array(list(map(float, lines[10].split(' '))))
            extend = np.array(list(map(float, lines[12].split(' '))))
        else:
            rotation_matrix = np.array([[float(x) for x in line.split(' ')] for line in lines[4:7]])
            center = np.array(list(map(float, lines[8].split(' '))))
            extend = np.array(list(map(float, lines[10].split(' '))))

        return image_size, model_id, rotation_matrix, center, extend


def draw_poly(image, vertices, color):
    """
    image: (h, w, c) int array
    vertices: (n_vertices, 2) vertices coordinates
    color: (c,) int array
    """
    cv2.fillConvexPoly(image, vertices.astype(int), color.tolist())
    return image


def transform_points(points, rotation_matrix, translation_vector):
    return points@rotation_matrix + translation_vector


def project_points(points, camera_matrix):
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera_matrix, None)[0][:, 0, :]


class ModelsHandler:
    def __init__(self, linemod_dataset_dir_path, color_resolution=256):
        self.camera_matrix = np.array([
            [572.41140, 0, 325.26110],
            [0, 573.57043, 242.04899],
            [0, 0, 1]
        ])
        self.color_resolution = int(color_resolution)
        self._model_name_to_model_file_path = dict()
        for model_dir in glob(f'{linemod_dataset_dir_path}/models/*'):
            model_name = os.path.split(model_dir)[1]
            if 'Holepuncher' in model_dir:
                continue
            model_filepath = glob(f'{model_dir}/*.obj')[0]

            self._model_name_to_model_file_path[model_name] = model_filepath
        self.model_name_to_model_id = {name: n + 1 for n, name in enumerate(sorted(self._model_name_to_model_file_path.keys()))}
        self.model_id_to_model_name = {v: k for k, v in self.model_name_to_model_id.items()}

    @property
    def model_names(self):
        return list(self.model_name_to_model_id.keys())

    @lru_cache()
    def get_vertices(self, model_name):
        return read_obj_file(self._model_name_to_model_file_path[model_name])[0]

    @lru_cache()
    def get_faces(self, model_name):
        return read_obj_file(self._model_name_to_model_file_path[model_name])[1]

    @lru_cache()
    def get_faces_midpoints(self, model_name):
        vertices = self.get_vertices(model_name)
        faces = self.get_faces(model_name)
        faces_mid_points = (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3
        return faces_mid_points

    @lru_cache()
    def get_faces_uv_colors(self, model_name):
        """
        return (n_faces, 2) float array with
            first  column being height scaled to [0, 1] and
            second column being angle  scaled to [0, 1]
        """
        vertices = self.get_vertices(model_name)
        faces = self.get_faces(model_name)
        faces_mid_points = self.get_faces_midpoints(model_name)
        max_height = vertices[:, 1].max()
        min_height = vertices[:, 1].min()
        height_colors = (faces_mid_points[:, 1] - min_height) / (max_height - min_height)
        angle_colors  = np.arctan2(*faces_mid_points[:, [0, 2]].T)
        angle_colors  = (angle_colors + np.pi) / (2*np.pi)
        return np.stack([height_colors, angle_colors], axis=-1)

    def draw_color_mask(self, image, model_name, rotation_matrix, center):
        """
        image: (h, w, 2) int valued array to draw onto
        rotation_matrix: (3,3) float array
        center: (3, ) float array - translation vector
        """
        translation_vector = center
        points3d_on_model = self.get_vertices(model_name)
        points3d_in_reality = transform_points(points3d_on_model, rotation_matrix, translation_vector)
        points2d_on_image = project_points(points3d_in_reality, self.camera_matrix)

        faces = self.get_faces(model_name)

        faces_mid_points3d_in_reality = (
            points3d_in_reality[faces[:, 0]] +
            points3d_in_reality[faces[:, 1]] +
            points3d_in_reality[faces[:, 2]]
        ) / 3

        # face_ordering = np.argsort(-faces_mid_points3d_in_reality[:, 1]) # draw faces on each model from bottom
        face_ordering = np.argsort(-faces_mid_points3d_in_reality[:, 2])  # draw faces on each model from front

        faces = faces[face_ordering]                                  # this changes order
        colors = self.get_faces_uv_colors(model_name)[face_ordering]  # this changes order

        for vertices, color in tqdm(list(zip(faces, colors))):
            draw_poly(image, np.stack([points2d_on_image[v] for v in vertices]), (self.color_resolution*color).astype(int))

        return image

    def draw_class_mask(self, image, model_name, rotation_matrix, center):
        """
        Fills area occupied by model accordingly to model_id->model_name mapping in self.model_name_to_model_id

        image: (h, w) int valued array to draw onto
        rotation_matrix: (3,3) float array
        center: (3, ) float array - translation vector
        """

        translation_vector = center
        points3d_on_model = self.get_vertices(model_name)
        points3d_in_reality = transform_points(points3d_on_model, rotation_matrix, translation_vector)
        points2d_on_image = project_points(points3d_in_reality, self.camera_matrix)

        faces = self.get_faces(model_name)
        color = np.array([self.model_name_to_model_id[model_name]]).astype(int)

        for vertices in tqdm(faces):
            draw_poly(image, np.stack([points2d_on_image[v] for v in vertices]), color)

        return image

import matplotlib.pyplot as plt

def generate_masks(linemod_dir_path, target_dir_path, parallel=False, debug=False, color_resolution=256):
    models_handler = ModelsHandler(linemod_dir_path, color_resolution=color_resolution)
    os.makedirs(target_dir_path, exist_ok=True)

    def target(image_id):
        data = []
        for model_name in models_handler.model_names:
            pose_file_path = f'{linemod_dir_path}/poses/{model_name}/info_{image_id}.txt'
            if not os.path.exists(pose_file_path):
                continue
            print(f'reading {pose_file_path}')
            _, _, rotation_matrix, center, _ = read_position_file(pose_file_path)
            data.append((model_name, rotation_matrix, center))

        correspondence_mask = np.zeros((480, 640, 2)).astype(int)
        class_mask = np.zeros((480, 640)).astype(int)
        for model_name, rotation_matrix, center in sorted(data, key=lambda x: x[2][2], reverse=True):
            correspondence_mask = models_handler.draw_color_mask(correspondence_mask, model_name, rotation_matrix, center)
            class_mask          = models_handler.draw_class_mask(class_mask,          model_name, rotation_matrix, center)

        plt.imshow(correspondence_mask[..., 0]);plt.show()
        plt.imshow(correspondence_mask[..., 1]);plt.show()
        plt.imshow(class_mask);plt.show()

    target('00000')


dir = '/home/maciej/Downloads/OcclusionChallengeICCV2015'

generate_masks(dir, '/')
5/0


mh = ModelsHandler(dir)
uv_colors = mh.get_faces_uv_colors('Cat')
img = np.zeros([480, 640, 2]) - 1
_, _, rotation_matrix, center, _ = read_position_file(f'{dir}/poses/Cat/info_00000.txt')
img = mh.draw_color_mask(img, 'Cat', rotation_matrix, center)

import matplotlib.pyplot as plt
plt.imshow(img[..., 0]);plt.show()
class_mask = np.zeros([480, 640])
class_mask = mh.draw_class_mask(class_mask, 'Cat', rotation_matrix, center)
plt.imshow(img[..., 0]); plt.show()
plt.imshow(img[..., 1]); plt.show()
plt.imshow(class_mask) ; plt.show()