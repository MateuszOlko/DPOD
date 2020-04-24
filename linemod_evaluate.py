from DPOD.datasets.linemod.models_handler import ModelsHandler, read_position_file
import numpy as np
import pickle
import os
import argparse
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
import cv2

from DPOD.datasets.linemod.models_handler import project_points

def error_on_instance(
    model_vertices,
    rotation_matrix,
    translation_vector,
    rotation_matrix_ground_truth,
    translation_vector_ground_truth):

    # pprint([translation_vector, translation_vector_ground_truth])
    pprint([rotation_matrix, rotation_matrix_ground_truth])
    # print('\n\n\n\n')

    vertices_predicted = model_vertices@rotation_matrix + translation_vector
    vertices_ground_truth = model_vertices@rotation_matrix_ground_truth + translation_vector_ground_truth

    size0 = vertices_predicted[:, 0].max() - vertices_predicted[:, 0].min()
    size1 = vertices_predicted[:, 1].max() - vertices_predicted[:, 1].min()
    size2 = vertices_predicted[:, 2].max() - vertices_predicted[:, 2].min()

    size = (size0**2 + size1**2 + size2**2) ** (1/2)

    print(size)


    size0 = vertices_ground_truth[:, 0].max() - vertices_ground_truth[:, 0].min()
    size1 = vertices_ground_truth[:, 1].max() - vertices_ground_truth[:, 1].min()
    size2 = vertices_ground_truth[:, 2].max() - vertices_ground_truth[:, 2].min()

    size = (size0**2 + size1**2 + size2**2) ** (1/2)
    #print(model_vertices.shape, vertices_ground_truth.shape, vertices_predicted.shape)
    #print(((vertices_predicted - vertices_ground_truth)**2).sum(axis=1).shape)
    # print(vertices_ground_truth.mean(axis=0), vertices_predicted.mean(axis=0))
    # mean_shift = (vertices_predicted - vertices_ground_truth).mean(axis=0)
    # shift_variance = (((vertices_predicted - vertices_ground_truth) - mean_shift) ** 2).mean(axis=0)
    # print(mean_shift, shift_variance**(1/2))
    # print(translation_vector - translation_vector_ground_truth)

    # err = np.empty(vertices_ground_truth.shape)
    # for i, v_gt in enumerate(vertices_ground_truth):
    #     err[i] = (((vertices_predicted - v_gt)**2).sum(axis=1)).min()

    err_sq = ((vertices_predicted - vertices_ground_truth)**2)
    err = (err_sq.sum(axis=1)**(1/2))
    err = (err**(1/2)).mean()

    hit = err < (size * 0.1)
    print(err, size * 0.1, hit)
    return err, hit


def evaluate(path_to_linemod_dir, path_to_instances_dir):
    models_handler = ModelsHandler()
    instances_paths = glob(f'{path_to_instances_dir}/*.pkl')
    acc_on_images = []


    translations = []
    translations_gt = []
    rotations = []
    rotations_gt = []
    for instances_path in instances_paths:
        print('evaluating', instances_path)

        with open(instances_path, 'rb') as file:
            instances = pickle.load(file)

        image_id = os.path.split(instances_path)[1][:-len('_instances.pkl')]

        hits = []

        image = np.zeros((480, 640))
        image_gt = np.zeros((480, 640))

        for model_name, translation_vector, rotation_matrix in instances:
            position_file_path = f'{path_to_linemod_dir}/poses/{model_name}/info_{image_id}.txt'
            # translation_vector = translation_vector * np.array([1, 1, -1])
            # rotation_matrix = rotation_matrix @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
            # rotation_matrix = rotation_matrix * np.array([[1, 1, 1], [1, -1, -1], [-1, -1, -1]])
            if not os.path.exists(position_file_path):
                hits.append(0)
                print('wrongly predicted', model_name, position_file_path)
                continue
                
        
            position_ground_truth = read_position_file(position_file_path)
            if position_ground_truth is None:
                hits.append(0)
                print('wrongly predicted', model_name, position_file_path)
                continue

            _, _, rotation_matrix_gt, translation_vector_gt, _ = position_ground_truth
            err, hit = error_on_instance(
                models_handler.get_vertices(model_name),
                rotation_matrix,
                translation_vector,
                rotation_matrix_gt,
                translation_vector_gt
            )
            hits.append(hit)
            # translations.append(translation_vector)
            # translations_gt.append(translation_vector_gt)
            # for i in range(3):
            #     rotations.append(rotation_matrix[i])
            #     rotations_gt.append(rotation_matrix_gt[i])

            models_handler.draw_class_mask(image, model_name, rotation_matrix, translation_vector)
            models_handler.draw_class_mask(image_gt, model_name, rotation_matrix_gt, translation_vector_gt)


        # p = project_points(np.array([[0,0,0], [0.1,0,0]]), models_handler.camera_matrix)
        # p = [(int(e[0]), int(e[1])) for e in p]
        # print(p)
        # cv2.line(image, p[0], p[1], (1, 1, 1))
        # p = project_points(np.array([[0,0,0], [0,0.1,0]]), models_handler.camera_matrix)
        # p = [(int(e[0]), int(e[1])) for e in p]
        # print(p)
        # cv2.line(image, p[0], p[1], (1, 1, 1))
        # p = project_points(np.array([[0,0,0], [0,0,0.1]]), models_handler.camera_matrix)
        # p = [(int(e[0]), int(e[1])) for e in p]
        # print(p)
        # cv2.line(image, p[0], p[1], (1, 1, 1))
        plt.imsave("render.png", image)
        plt.imsave("render_gt.png", image_gt)

        masks = np.load(f"../experiments/repro/Apr-15-16:06/infered_masks/{image_id}_masks.npy")
        plt.imsave("class.png", masks[0])
        break

    # print("Solution")
    # print(len(translations))
    # print(np.linalg.lstsq(np.array(translations), np.array(translations_gt))[0])
    # print("Rotations")
    # print(len(translations))
    # print(np.linalg.lstsq(np.array(rotations), np.array(rotations_gt))[0])
    
    acc_on_images.append(np.mean(hits))

    return np.mean(acc_on_images)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path_to_instances_dir')
    argparser.add_argument('--path_to_linemod_dir', default='/mnt/bigdisk/datasets/linemod')
    args = argparser.parse_args()

    acc = evaluate(args.path_to_linemod_dir, args.path_to_instances_dir)
    print(acc)

