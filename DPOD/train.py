import argparse

import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
from tqdm import tqdm

from datasets import make_dataset
from model import DPOD
from utils import get_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Train DPOD model')
    parser.add_argument('--dataset', dest="dataset", default="kaggle",
                        help='Name of dataset to train on.')
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int)
    parser.add_argument('--workers', help='Number of workers to use while loading data.')
    parser.add_argument('-lr', '--learning_rate', dest="learning_rate",
                        help='Learning rate of optimizer.')
    parser.add_argument('--val_size', type=float, default=0.25,
                        help='Validation size as percentage of dataset.')

    return parser.parse_args()


def wise_loss(preds, targets):
    class_pred = get_mask(preds[0])
    class_target = get_mask(targets[0])
    class_loss = (class_pred == class_target)

    u_pred = get_mask(preds[1])
    u_target = get_mask(targets[1])
    u_loss = ((u_pred - u_target) % 256, (u_target - u_pred) % 256).min()

    return class_loss, u_loss


def train(args, model):
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

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    for e in range(args.epochs):
        mean_loss = 0
        for images, targets, _ in tqdm(train_data):
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds[0], targets[0]).mean()
            loss += criterion(preds[1], targets[1]).mean()
            loss += criterion(preds[2], targets[2]).mean()
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()
        print(f"Epoch {e}: Train loss {mean_loss/len(train_data)}")

    with torch.no_grad():
        mean_loss = 0
        mean_class_loss = 0
        mean_u_loss = 0
        for images, targets, _ in val_data:
            preds = model(images)
            class_loss, u_loss = wise_loss(preds, targets)
            loss = criterion(preds[0], targets[0]).mean()
            loss += criterion(preds[1], targets[1]).mean()
            loss += criterion(preds[2], targets[2]).mean()
            mean_loss += loss.item()
            mean_class_loss += class_loss.mean().item()
            mean_u_loss += class_loss.mean().item()

        mean_loss /= len(val_data)
        mean_class_loss /= len(val_data)
        mean_u_loss /= len(val_data)
        print(f"Epoch {e} Eval scores - BCELoss: {mean_loss},"
              f" Classification accuracy: {mean_class_loss},"
              f" Our loss (u channel): {mean_u_loss}"
              )


def main():
    torch.random.seed(42)
    np.random.seed(42)
    args = parse_args()
    model = DPOD()
    train(args, model)


if __name__ == "__main__":
    main()


