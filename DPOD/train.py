import argparse
import os

import torch
import numpy as np
import pandas as pd

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
    parser.add_argument('--checkpoint', help="Path to model to resume training from")

    args = parser.parse_args()
    args.exp_dir = get_experiment_directory(args)
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
    train_set, val_set, whole_dataset = make_dataset(args, name=args.dataset)
    class_weights, height_weights, angle_weights = [a.to(device) for a in whole_dataset.get_class_weights()]

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

    scores = pd.DataFrame(columns="epoch train_loss val_loss val_accuracy val_u_channel val_v_channel".split())
    class_criterion = CrossEntropyLoss(weight=class_weights)
    height_criterion = CrossEntropyLoss(weight=height_weights)
    angle_criterion = CrossEntropyLoss(weight=angle_weights)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for e in range(args.epochs):
        mean_loss = 0
        for images, targets in tqdm(train_data):
            images = images.to(device)
            targets = [t.type(torch.LongTensor).to(device) for t in targets]
            optimizer.zero_grad()
            preds = model(images)
            loss = class_criterion(preds[0], targets[0])
            loss += height_criterion(preds[1], targets[1])
            loss += angle_criterion(preds[2], targets[2])
            loss.backward()
            mean_loss += loss.item()
            optimizer.step()
        mean_loss /= len(train_data)
        print(f"Epoch {e}: Train loss {mean_loss}")

        model.eval()
        with torch.no_grad():
            mean_val_loss = 0
            mean_class_loss = 0
            mean_u_loss = 0
            mean_v_loss = 0
            for images, targets in tqdm(val_data):
                images = images.to(device)
                targets = [t.type(torch.LongTensor).to(device) for t in targets]
                preds = model(images)
                class_loss, u_loss, v_loss = wise_loss(preds, targets)
                loss = class_criterion(preds[0], targets[0])
                loss += height_criterion(preds[1], targets[1])
                loss += angle_criterion(preds[2], targets[2])
                mean_val_loss += loss.item()
                mean_class_loss += class_loss.mean().item()
                mean_u_loss += u_loss.mean().item()
                mean_v_loss += v_loss.mean().item()

            mean_val_loss /= len(val_data)
            mean_class_loss /= len(val_data)
            mean_u_loss /= len(val_data)
            mean_v_loss /= len(val_data)
            print(f"Epoch {e} Eval scores - CELoss: {mean_val_loss},"
                f" Classification accuracy: {mean_class_loss},"
                f" Our loss (u channel): {mean_u_loss},"
                f" Our loss (v channel): {mean_v_loss}"
                )
        
        scores = scores.append([{
            'epoch': e,
            'train_loss': mean_loss,
            'val_loss': mean_val_loss,
            'val_accuracy': mean_class_loss,
            'val_u_channel': mean_u_loss,
            'val_v_channel': mean_v_loss,
        }])

        os.makedirs(os.path.join(args.exp_dir, f"epoch-{e}"))
        model_path = os.path.join(args.exp_dir, f"epoch-{e}", "model.pt") 
        csv_path = os.path.join(args.exp_dir, f"epoch-{e}", "scores.csv")
        
        torch.save(model.state_dict(), model_path)
        scores.to_csv(csv_path)

    model_path = os.path.join(args.exp_dir, "final-model.pt") 
    csv_path = os.path.join(args.exp_dir, "scores.csv")
    torch.save(model.state_dict(), model_path)
    scores.to_csv(csv_path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPOD(image_size=(480, 640), num_classes=7+1)
    if args.checkpoint:
        print("Loading model from checkpoint:", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    train(args, model, device)


if __name__ == "__main__":
    main()


