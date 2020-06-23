import numpy as np
import cv2
from glob import glob
import os
from functools import lru_cache
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


from concurrent.futures import ThreadPoolExecutor


def get_reversed_mapping(mh, model_name):
    def compute_threaded(ids, colors, ver, t):
        color_to_ver = np.empty((len(ids), 255, 3))
        for num, i in enumerate(ids):
            if i >= 255:
                 continue
            a = colors - np.array([i, 0])
            a_sq = (a ** 2).sum(axis=1)
            for j in range(255):
                diff_sq = a_sq - 2 * a[:, 1] * j + j ** 2
                min_ = np.argmin(diff_sq)
                color_to_ver[num, j, :] = ver[min_]
                t.update()
        return ids, color_to_ver

    ver = mh.get_vertices(model_name)
    colors = mh.color_uv_model(ver, model_name) * 255
    color_to_vers = np.empty((255, 255, 3))
    t = tqdm(total=255*255, smoothing=0.05)
    with ThreadPoolExecutor() as executor:
        for ids, rows in executor.map(
            compute_threaded,
            np.arange(256).reshape(8, -1),
            [colors] * 255,
            [ver] * 255,
            [t] * 255,
        ):
            proper_ones = ids < 255
            ids = ids[proper_ones]
            color_to_vers[ids] = rows[proper_ones]
#     for i, j in tqdm(product(range(255), range(255)), total=255):
#         min_ = np.argmin(((colors - np.array([i, j]))**2).sum(axis=1))
#         color_to_vers[i, j, :] = ver[min_]
    return color_to_vers

def read_obj_file(path):
    """
    reads <something>_small.obj file

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
                v1, v2, v3 = map(int, line.split(' ')[1:])
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
    try:
        with open(path) as file:
            lines = file.readlines()
            if len(lines) == 2:
                # no object
                return None
            image_size = tuple(map(int, lines[1].split(' ')))
            model_id = lines[2]

            rotation_matrix = np.array([[float(x) for x in line.split(' ')] for line in lines[4:7]])
            center = np.array(list(map(float, lines[8].split(' '))))
            extend = np.array(list(map(float, lines[10].split(' '))))

            return image_size, model_id, rotation_matrix, center, extend

    except Exception as e:
        print(e)
        print('crashed on', path)
        print(open(path).readlines())
        raise e


def draw_poly(image, vertices, color):
    """
    image: (h, w, c) int array
    vertices: (n_vertices, 2) vertices coordinates
    color: (c,) int array
    """
    cv2.fillConvexPoly(image, vertices.astype(int), color.tolist())
    return image


def transform_points(points, rotation_matrix, translation_vector):
    points = points@rotation_matrix.T + translation_vector
    points = points@np.diag([1, -1, -1])
    return points


def project_points(points, camera_matrix):
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera_matrix, None)[0][:, 0, :]


class ModelsHandler:
    def __init__(self, models_dir_path='models_big', color_resolution=255):
        self.camera_matrix = np.array([
            [572.41140, 0, 325.26110],
            [0, 573.57043, 242.04899],
            [0, 0, 1]
        ])
        self.color_resolution = color_resolution
        self._model_name_to_model_file_path = dict()
        self._model_color_mapping = dict()
        for model_dir in glob(f'{models_dir_path}/*'):
            model_name = os.path.split(model_dir)[1]
            if 'Holepuncher' in model_dir:
                continue
            model_filepath = glob(f'{model_dir}/*.obj')[0]

            self._model_name_to_model_file_path[model_name] = model_filepath
            self._model_color_mapping[model_name] = np.load(f"{model_dir}/mapping.npy")
        self.model_name_to_model_id = {name: n + 1 for n, name in enumerate(sorted(self._model_name_to_model_file_path.keys()))}
        self.model_id_to_model_name = {v: k for k, v in self.model_name_to_model_id.items()}
        self.model_h_min_max = {model_name: (self.get_vertices(model_name)[:, 1].min(), self.get_vertices(model_name)[:, 1].max())
                                for model_name in self.model_names}
        print(self.model_name_to_model_id)

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

    def color_uv(self, points):
        # to [0, 1] range
        max_height = points[:, 1].max()
        min_height = points[:, 1].min()
        height_colors = (points[:, 1] - min_height) / (max_height - min_height)
        angle_colors  = np.arctan2(*points[:, [2, 0]].T)
        angle_colors  = (angle_colors + np.pi) / (2*np.pi)
        return np.stack([height_colors, angle_colors], axis=-1)

    def color_uv_model(self, points, model_name):
        # to [0, 1] range
        min_height, max_height = self.model_h_min_max[model_name]
        height_colors = (points[:, 1] - min_height) / (max_height - min_height)
        angle_colors  = np.arctan2(*points[:, [2, 0]].T)
        angle_colors  = (angle_colors + np.pi) / (2*np.pi)
        return np.stack([height_colors, angle_colors], axis=-1)

    def draw_face(self, image, vertices, model_name, rotation_matrix, translation_vector):
        """
        image: (h, w, c) int array
        vertices: (3, 3) vertices coordinates
        colors: (3, 2) int array
        """
        h_diff = vertices[:, 0].max() - vertices[:, 0].min()
        w_diff = vertices[:, 1].max() - vertices[:, 1].min()
        z_diff = vertices[:, 2].max() - vertices[:, 2].min()
        diff = max(h_diff, w_diff, z_diff)

        if diff > 0.005:
            mids = np.stack([vertices[[2, 1]].mean(axis=0), vertices[[0, 2]].mean(axis=0), vertices[[0, 1]].mean(axis=0)])
            for i in range(3):
                image = self.draw_face(
                    image,
                    np.stack([vertices[i], mids[(i + 1) % 3], mids[(i + 2) % 3]]),
                    model_name,
                    rotation_matrix,
                    translation_vector
                )
            image = self.draw_face(
                image,
                mids,
                model_name,
                rotation_matrix,
                translation_vector
            )
        else:
            mid_point = vertices.mean(axis=0)
            mid_color = self.color_uv_model(np.array([mid_point]), model_name) * self.color_resolution
            reality = transform_points(vertices, rotation_matrix, translation_vector)
            projected = project_points(reality, self.camera_matrix)
            image = draw_poly(image, projected, mid_color[0].astype(np.int))

        return image


    @lru_cache()
    def get_faces_uv_colors(self, model_name):
        """
        return (n_faces, 2) float array with
            first  column being height scaled to [0, 1] and
            second column being angle  scaled to [0, 1]
        """
        vertices = self.get_vertices(model_name)
        faces_mid_points = self.get_faces_midpoints(model_name)  # self.color_uv can be used here
        max_height = vertices[:, 1].max()
        min_height = vertices[:, 1].min()
        height_colors = (faces_mid_points[:, 1] - min_height) / (max_height - min_height)
        angle_colors  = np.arctan2(*faces_mid_points[:, [2, 0]].T)
        angle_colors  = (angle_colors + np.pi) / (2*np.pi)
        return np.stack([height_colors, angle_colors], axis=-1)

    @lru_cache()
    def get_faces_vertices_uv_colors(self, model_name):
        """
        return (n_faces, 3, 2) float array with
            first  column being height scaled to [0, 1] and
            second column being angle  scaled to [0, 1]
        """
        vertices = self.get_vertices(model_name)
        faces = self.get_faces(model_name)
        faces = vertices[faces]
        max_height = vertices[:, 1].max()
        min_height = vertices[:, 1].min()
        height_colors = (faces[:, :, 1] - min_height) / (max_height - min_height)
        angle_colors  = np.arctan2(*faces[:, :, [2, 0]].T).T
        angle_colors  = (angle_colors + np.pi) / (2*np.pi)
        return np.stack([height_colors, angle_colors], axis=-1)

    @lru_cache()
    def get_color_to_3dpoints_arrays(self, model_name):
        # #todo oddzielnie wysokość i kąty lol
        # vertices = self.get_vertices(model_name)#@np.diag([1, -1, -1])
        # colors = (self.color_uv(vertices)*self.color_resolution).astype(int)
        # points_for_griddata = colors
        # values_for_griddata = vertices
        # grid1, grid2 = np.mgrid[0:self.color_resolution, 0:self.color_resolution]
        #
        # def interpolate(method):
        #     return griddata(
        #         points=points_for_griddata,
        #         values=values_for_griddata,
        #         xi=(grid1, grid2),
        #         method=method
        #     )
        #
        # interpolated = interpolate('cubic')
        # missing_mask = np.isnan(interpolated)
        # interpolated[missing_mask] = interpolate('nearest')[missing_mask]
        # #interpolated = interpolate('nearest')
        return self._model_color_mapping[model_name]


    @lru_cache()
    def get_color_to_y(self, model_name):
        vertices = self.get_vertices(model_name)#@np.diag([1, -1, -1])
        colors = (self.color_uv(vertices)*self.color_resolution).astype(int)
        points_for_griddata = colors[:, 0]
        values_for_griddata = vertices[:, 1]
        grid = np.mgrid[0:self.color_resolution]

        def interpolate(method):
            return griddata(
                points=points_for_griddata,
                values=values_for_griddata,
                xi=grid,
                method=method
            )

        interpolated = interpolate('linear')
        missing_mask = np.isnan(interpolated)
        interpolated[missing_mask] = interpolate('nearest')[missing_mask]
        #interpolated = interpolate('nearest')
        return interpolated

    @lru_cache()
    def get_color_to_xz(self, model_name):
        vertices = self.get_vertices(model_name)#@np.diag([1, -1, -1])
        colors = (self.color_uv(vertices)*self.color_resolution).astype(int)
        points_for_griddata = colors[:, 1]
        values_for_griddata = np.stack([vertices[:, 0], vertices[:, 2]], axis=-1)
        # print("xz val", np.stack([vertices[:, 0], vertices[:, 2]], axis=-1).shape)
        grid = np.mgrid[0:self.color_resolution]

        def interpolate(method):
            return griddata(
                points=points_for_griddata,
                values=values_for_griddata,
                xi=grid,
                method=method
            )

        interpolated = interpolate('linear')
        missing_mask = np.isnan(interpolated)
        interpolated[missing_mask] = interpolate('nearest')[missing_mask]
        #interpolated = interpolate('nearest')
        return interpolated

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

        faces = faces[face_ordering, :]                                  # this changes order
        colors = self.get_faces_uv_colors(model_name)[face_ordering] * self.color_resolution  # this changes order
        # for vertices in points3d_on_model[faces]:
        #     self.draw_face(image, vertices, model_name, rotation_matrix, translation_vector)
        for vertices, color in zip(points2d_on_image[faces], colors):
            draw_poly(image, vertices, color.astype(int))

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
        # print(points2d_on_image[:4])
        faces = self.get_faces(model_name)
        print("ver, faces", points3d_on_model.shape, faces.shape)
        print("f idx stat", faces.min(), faces.max())
        color = np.array([self.model_name_to_model_id[model_name]]).astype(int)

        for vertices in faces:
            draw_poly(image, np.stack([points2d_on_image[v] for v in vertices]), color)

        return image
