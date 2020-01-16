For prediction use
```usage: test.py [-h] [-d]
               path_to_model path_to_output_file path_to_kaggle_dataset_folder

positional arguments:
  path_to_model
  path_to_output_file
  path_to_kaggle_dataset_folder

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           do only 5 batches
```

For example
```
python DPOD/test.py experiments/DPOD/Jan-15-15\:20//final-model.pt debug_submission.csv /mnt/bigdisk/datasets/kaggle/ -d
```


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