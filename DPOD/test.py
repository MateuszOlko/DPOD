from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from DPOD.model import DPOD, PoseBlock, Translator
from DPOD.datasets.kaggle_dataset import KaggleImageMaskDataset


def main(path_to_model, path_to_kaggle_folder, render_visualizations=False):

    # determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    dataset = KaggleImageMaskDataset(path_to_kaggle_folder, is_train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=10,
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
        for n_image, images in enumerate(tqdm(data_loader)):

            batch_of_images = images.to(device)
            batch_of_classes, batch_of_u_channel, batch_of_v_channel = model1(batch_of_images)
            batch_of_instances = model2(batch_of_classes, batch_of_u_channel, batch_of_v_channel)

            batch_of_prediction_strings = model3(batch_of_instances)
            list_of_prediction_strings.extend(batch_of_prediction_strings)

            if n_image >= 3:
                break

    print(list_of_prediction_strings)
    return list_of_prediction_strings


if __name__ == "__main__":
    main('experiments/DPOD/Jan-15-15:20/final-model.pt', '../datasets/kaggle', False)
