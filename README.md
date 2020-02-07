# LINEMOD dataset
is available [here](https://hci.iwr.uni-heidelberg.de/vislearn/iccv2015-occlusion-challenge/).
We are not using `Holepuncher` as its 3D model seems corrupted. We are using downsampled 3D models
(present in `models_small`) as it speeds up rendering process. They can be produced using MeshLab 
and `meshlab_script.mlx`  
# Pipeline

## Generating masks
```
usage: linemod_generate_masks.py [-h] [--linemod_dir_path LINEMOD_DIR_PATH]
                                 [--models_dir_path MODELS_DIR_PATH]
                                 [--target_dir_path TARGET_DIR_PATH] [--show]
                                 [--debug] [--save]

optional arguments:
  -h, --help            show this help message and exit
  --linemod_dir_path LINEMOD_DIR_PATH path to directory containing linemod dataset
  --models_dir_path MODELS_DIR_PATH   path to directory containing 3D models
  --target_dir_path TARGET_DIR_PATH   path to directory to save generated masks to
  --show                show generated images on the go
  --debug, -d           process only 20 images
  --save                save humanreadable masks
```

## Training
```
TODO
```

## Inferring masks
```
TODO
```

## Applying PnP+RANSAC
```
usage: linemod_ransac.py [-h] [-d] [-v] [--min_inliers MIN_INLIERS]
                         [--iterationsCount ITERATIONSCOUNT]
                         [--reprojectionError REPROJECTIONERROR]
                         [--confidence CONFIDENCE] [--flags FLAGS]
                         path_to_masks_dir path_to_output_dir

Applies ransac with specified parameters to all (3,h,w) np.uint8
path_to_masks_dir/<ImageId>_masks.npy masks with u, v, class channels and
saves 
[ 
  [ 
    model_id, int 
    ransac_translation_vector, (3) float np.array
    ransac_rotation_matrix, (3,3) float np.array 
  ] for each instance found 
] 
to path_to_output_dir/<ImageId>_instances.pkl with pickle.dump skipping
calculating already saved outputs

positional arguments:
  path_to_masks_dir
  path_to_output_dir

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           process only 20 images
  -v, --verbose         print predicted poses
  --min_inliers MIN_INLIERS  handcrafted RANSAC parameter
  --iterationsCount ITERATIONSCOUNT      RANSAC parameter
  --reprojectionError REPROJECTIONERROR  RANSAC parameter
  --confidence CONFIDENCE                RANSAC parameter
  --flags FLAGS                          RANSAC parameter

```