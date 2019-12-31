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