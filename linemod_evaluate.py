from linemod_models_handler import ModelsHandler, read_position_file
import numpy as np
import pickle
import os
import argparse
from glob import glob
from pprint import pprint


def error_on_instance(
    model_vertices,
    rotation_matrix,
    translation_vector,
    rotation_matrix_ground_truth,
    translation_vector_ground_truth):

    #pprint([translation_vector, translation_vector_ground_truth])
    #pprint([rotation_matrix, rotation_matrix_ground_truth])
    #print('\n\n\n\n')

    size0 = model_vertices[:, 0].max() - model_vertices[:, 0].min()
    size1 = model_vertices[:, 1].max() - model_vertices[:, 1].min()
    size2 = model_vertices[:, 2].max() - model_vertices[:, 2].min()

    size = max(size0, size1, size2)

    vertices_predicted = model_vertices@rotation_matrix + translation_vector
    vertices_ground_truth = model_vertices@rotation_matrix_ground_truth + translation_vector_ground_truth

    #print(model_vertices.shape, vertices_ground_truth.shape, vertices_predicted.shape)
    #print(((vertices_predicted - vertices_ground_truth)**2).sum(axis=1).shape)

    err_sq = ((vertices_predicted - vertices_ground_truth)**2).sum(axis=1)
    err = (err_sq**(1/2)).mean()

    hit = err < (size * 0.10)
    
    return err, hit


def evaluate(path_to_linemod_dir, path_to_instances_dir):
    models_handler = ModelsHandler()
    instances_paths = glob(f'{path_to_instances_dir}/*.pkl')
    acc_on_images = []

    for instances_path in instances_paths:
        print('evaluating', instances_path)

        with open(instances_path, 'rb') as file:
            instances = pickle.load(file)

        image_id = os.path.split(instances_path)[1][:-len('_instances.pkl')]

        hits = []

        for model_name, translation_vector, rotation_matrix in instances:
            position_file_path = f'{path_to_linemod_dir}/poses/{model_name}/info_{image_id}.txt'
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
            print(hit, err)
            hits.append(hit)
        
        acc_on_images.append(np.mean(hits))

    return np.mean(acc_on_images)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('path_to_instances_dir')
    argparser.add_argument('--path_to_linemod_dir', default='/mnt/bigdisk/datasets/linemod')
    args = argparser.parse_args()

    acc = evaluate(args.path_to_linemod_dir, args.path_to_instances_dir)
    print(acc)

