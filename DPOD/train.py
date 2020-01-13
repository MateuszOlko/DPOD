import argparse
import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from datasets import make_dataset
from model import DPOD
from utils import get_mask, get_experiment_directory


def parse_args():
    parser = argparse.ArgumentParser(description='Train DPOD model')
    parser.add_argument('--dataset', dest="dataset", default="kaggle",
                        help='Name of dataset to train on.')
    parser.add_argument('--batch-size', default=2, dest="batch_size",
                        type=int)
    parser.add_argument('--workers', default=1, type=int, help='Number of workers to use while loading data.')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, dest="learning_rate", help='Learning rate of optimizer.')
    parser.add_argument('--val_size', type=float, default=0.25,
                        help='Validation size as percentage of dataset.')
    parser.add_argument('--epochs', help="Number of epochs of training", default=20, type=int)
    parser.add_argument('--name', help="Name of the experiment", default="DPOD", type=str)

    args = parser.parse_args()
    args.exp_name = get_experiment_directory(args)
    return args


def wise_loss(preds, targets):
    class_pred = get_mask(preds[0])
    class_loss = (class_pred == targets[0]).type(torch.FloatTensor)

    u_pred = get_mask(preds[1])
    u_target = targets[1]
    u_loss = torch.min((u_pred - u_target) % 256, (u_target - u_pred) % 256).type(torch.FloatTensor)

    v_pred = get_mask(preds[2])
    v_target = targets[2]
    v_loss = torch.min((v_pred - v_target) % 256, (v_target - v_pred) % 256).type(torch.FloatTensor)

    return class_loss, u_loss, v_loss


def train(args, model, device):
    model_path = os.path.join(args.exp_name, "saved-model.pt") 
    train_set, val_set = make_dataset(args, name=args.dataset)

    train_data = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        drop_last=True,
    )

    val_data = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        drop_last=True,
    )

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    for e in range(args.epochs):
        mean_loss = 0
        for images, targets, _ in tqdm(train_data):
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds[0], targets[0])
            loss += criterion(preds[1], targets[1])
            loss += criterion(preds[2], targets[2])
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()
        print(f"Epoch {e}: Train loss {mean_loss/len(train_data)}")

        with torch.no_grad():
            mean_loss = 0
            mean_class_loss = 0
            mean_u_loss = 0
            mean_v_loss = 0
            for images, targets, _ in tqdm(val_data):
                images = images.to(device)
                targets = [t.to(device) for t in targets]
                preds = model(images)
                class_loss, u_loss, v_loss = wise_loss(preds, targets)
                loss = criterion(preds[0], targets[0])
                loss += criterion(preds[1], targets[1])
                loss += criterion(preds[2], targets[2])
                mean_loss += loss.item()
                mean_class_loss += class_loss.mean().item()
                mean_u_loss += u_loss.mean().item()
                mean_v_loss += v_loss.mean().item()

            mean_loss /= len(val_data)
            mean_class_loss /= len(val_data)
            mean_u_loss /= len(val_data)
            mean_v_loss /= len(val_data)
            print(f"Epoch {e} Eval scores - CELoss: {mean_loss},"
                f" Classification accuracy: {mean_class_loss},"
                f" Our loss (u channel): {mean_u_loss},"
                f" Our loss (v channel): {mean_v_loss}"
                )

    torch.save(model.state_dict(), model_path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPOD(image_size=(3384//8, 2710//8)).to(device)
    train(args, model, device)


if __name__ == "__main__":
    main()


