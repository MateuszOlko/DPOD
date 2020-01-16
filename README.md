First predicions are in  `experiments/DPOD/Jan-15-15:20/output`. use `rotation_matrix_to_kaggle_yaw_pitch_roll` function from models handler to make submission

```
pip3 install -r requirements.txt
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