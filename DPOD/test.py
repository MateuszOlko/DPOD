from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock, Translator
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset
from argparse import ArgumentParser
import pandas as pd


def main(path_to_model, path_to_kaggle_folder, path_to_output_file, number_of_batches_to_process):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = KaggleImageMaskDataset(path_to_kaggle_folder, is_train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1
    )

    # load correspondence block
    model1 = DPOD(image_size=(2710 // 8, 3384 // 8))
    model1.load_state_dict(torch.load(path_to_model))
    model1.to(device)

    # load pose block
    model2 = PoseBlock(path_to_kaggle_folder)

    # load 'translator' to prediction strings
    model3 = Translator()

    list_of_prediction_strings = []

    with torch.no_grad():
        for n_batch, images in enumerate(tqdm(data_loader, desc='(batches of images)')):
            
            if n_batch >= number_of_batches_to_process:
                break
            
            batch_of_images = images.to(device)
            batch_of_classes, batch_of_u_channel, batch_of_v_channel = model1(batch_of_images)
            batch_of_instances = model2(batch_of_classes, batch_of_u_channel, batch_of_v_channel)

            batch_of_prediction_strings = model3(batch_of_instances)
            list_of_prediction_strings.extend(batch_of_prediction_strings)

    print(dataset.get_IDs()[:number_of_batches_to_process])
    print(list_of_prediction_strings)

    submission_dataframe = pd.DataFrame({
        'ImageId': dataset.get_IDs()[:number_of_batches_to_process],
        'PredictionString': list_of_prediction_strings
    })

    submission_dataframe.to_csv(path_to_output_file, index=False)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('path_to_model')
    arg_parser.add_argument('path_to_output_file')
    arg_parser.add_argument('path_to_kaggle_dataset_folder')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='do only 5 batches')

    args = arg_parser.parse_args()

    number_of_batches_to_process = 5 if args.debug else 10000

    list_of_prediction_strings = main(
        args.path_to_model, 
        args.path_to_kaggle_dataset_folder, 
        args.path_to_output_file,
        number_of_batches_to_process
    )
    print(list_of_prediction_strings, sep='\n')
