## inferring masks only
```
usage: infer_masks.py [-h] [-d] path_to_model path_to_output_dir

infers mask using provided model and saves them as
path_to_output_dir/ImageId.npy as (3,h,w) np.uint8 numpy arrays,
ignores already present masks

positional arguments:
  path_to_model
  path_to_output_dir

optional arguments:
  -h, --help          show this help message and exit
  -d, --debug         process only 20 images

```
## applying PnPRansac only
```
usage: apply_ransac.py [-h] [-d] [--min_inliers MIN_INLIERS] [--no_class]
                       [--iterationsCount ITERATIONSCOUNT]
                       [--reprojectionError REPROJECTIONERROR]
                       [--confidence CONFIDENCE] [--flags FLAGS]
                       path_to_masks_dir path_to_output_dir

Applies ransac with specified parameters to all (3,h,w) np.uint8
path_to_masks_dir/<ImageId>.npy masks with class, u, v channels and saves
(output of PoseBlock) [ [ model_id, int ransac_translation_vector, (3) float
np.array ransac_rotation_matrix, (3,3) float np.array ] for each instance
found ] to path_to_output_dir/<ImageId>_instances.pkl with pickle.dump
skipping calculating already saved outputs

positional arguments:
  path_to_masks_dir
  path_to_output_dir

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           process only 20 images
  --min_inliers MIN_INLIERS
  --no_class
  --iterationsCount ITERATIONSCOUNT
  --reprojectionError REPROJECTIONERROR
  --confidence CONFIDENCE
  --flags FLAGS

```



## Visualization (not working)
```
usage: visualize.py [-h] [-d]
                    path_to_submission_file path_to_kaggle_dataset_dir
                    path_to_output_dir

positional arguments:
  path_to_submission_file
  path_to_kaggle_dataset_dir
  path_to_output_dir

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           do only 20 images

```




```
usage: generate_masks.py [-h] [-f] [-d] [-p]
                         kaggle_dataset_path mask_folder_path

positional arguments:
  kaggle_dataset_path
  mask_folder_path

optional arguments:
  -h, --help           show this help message and exit
  -f, --force          force calculating masks again
  -d, --debug          calculate only 10 masks, for debugging
  -p, --parallel       use all cores
```

### Sanity checks
Masks and interpretation
```
usage: check_generated_masks.py [-h] [--show] [--save] kaggle_dataset_dir_path

run to check whether mask are read and interpreted properly

positional arguments:
  kaggle_dataset_dir_path

optional arguments:
  -h, --help            show this help message and exit
  --show                show masks visualization with matplotlib (blocks)
  --save                save produced visualizations in sanity_checks
```

Dataset
```
usage: check_dataset_class_train.py [-h] [--show] [--save]
                                    kaggle_dataset_dir_path

run to check whether dataset return what it should

positional arguments:
  kaggle_dataset_dir_path

optional arguments:
  -h, --help            show this help message and exit
  --show                show masks visualization with matplotlib (blocks)
  --save                save produced visualizations in sanity_checks
```