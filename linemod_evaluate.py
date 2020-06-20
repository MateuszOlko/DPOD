from DPOD.datasets.linemod.models_handler import ModelsHandler, read_position_file
import numpy as np
import pickle
import os
import argparse
from glob import glob
from pprint import pprint
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from DPOD.datasets.linemod.models_handler import project_points
from DPOD.datasets import PATHS
from DPOD.datasets.linemod.models_handler import transform_points

def error_on_instance(
    model_vertices,
    rotation_matrix,
    translation_vector,
    rotation_matrix_ground_truth,
    translation_vector_ground_truth):

    # pprint([translation_vector, translation_vector_ground_truth])
    # pprint([rotation_matrix, rotation_matrix_ground_truth])
    # print('\n\n\n\n')

    vertices_predicted = model_vertices@rotation_matrix + translation_vector
    # vertices_predicted = transform_points(model_vertices, rotation_matrix, translation_vector)
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

    size0 = model_vertices[:, 0].max() - model_vertices[:, 0].min()
    size1 = model_vertices[:, 1].max() - model_vertices[:, 1].min()
    size2 = model_vertices[:, 2].max() - model_vertices[:, 2].min()

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
    err = err.mean()

    hit = err < (size * 0.1)
    print(err, size * 0.1, hit)
    return err, hit


def evaluate(path_to_linemod_dir, path_to_instances_dir):
    models_handler = ModelsHandler()
    instances_paths = glob(f'{path_to_instances_dir}/*.pkl')
    acc_on_images = []
    retrived = 0
    relevant = 0
    retrived_relevant = 0
    errors = []

    idx= 0
    for instances_path in tqdm(instances_paths):
        print('evaluating', instances_path)

        with open(instances_path, 'rb') as file:
            instances = pickle.load(file)

        instances_dict = {name: (vec, matrix) for name, vec, matrix in instances}

        image_id = os.path.split(instances_path)[1][:-len('_instances.pkl')]

        hits = []

        for model_name in models_handler.model_names:
            position_file_path = f'{path_to_linemod_dir}/poses/{model_name}/info_{image_id}.txt'
            if model_name in instances_dict.keys():
                retrived += 1
                translation_vector, rotation_matrix = instances_dict[model_name]
                translation_vector = translation_vector * np.array([1, -1, -1])
                rotation_matrix = np.diag([1, -1, -1])@rotation_matrix
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
                errors.append(err)
                retrived_relevant += hit
                relevant += 1
                hits.append(hit)
                # models_handler.draw_class_mask(image, model_name, rotation_matrix, translation_vector)
                # models_handler.draw_class_mask(image_gt, model_name, rotation_matrix_gt, translation_vector_gt)
            elif os.path.exists(position_file_path):
                position_ground_truth = read_position_file(position_file_path)
                if position_ground_truth is not None:
                    hits.append(0)
                    relevant += 1
                    # _, _, rotation_matrix_gt, translation_vector_gt, _ = position_ground_truth
                    # models_handler.draw_class_mask(image_gt, model_name, rotation_matrix_gt, translation_vector_gt)


        # plt.imsave(f"render{idx}.png", image)
        # plt.imsave(f"render_gt{idx}.png", image_gt)
        # idx += 1
        #
        # masks = np.load(f"{PATHS['linemod']}/masks/{image_id}_masks.npy")
        # plt.imsave(f"class{idx}.png", masks[2])


    # print("Solution")
    # print(len(translations))
    # print(np.linalg.lstsq(np.array(translations), np.array(translations_gt))[0])
    # print("Rotations")
    # print(len(translations))
    # print(np.linalg.lstsq(np.array(rotations), np.array(rotations_gt))[0])
        m = np.mean(hits) if hits else 0
        acc_on_images.append(m)

    return np.mean(acc_on_images), np.mean(errors)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path_to_instances_dir')
    argparser.add_argument('--path_to_linemod_dir', default=PATHS["linemod"])
    args = argparser.parse_args()

    acc, mean_error = evaluate(args.path_to_linemod_dir, args.path_to_instances_dir)
    print(f"Accuracy {acc}; Mean error {mean_error}")

