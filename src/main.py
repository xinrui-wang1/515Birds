import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import Net
from train import train_model
from predict import predict_image
from api import app

def parse_args():
    parser = argparse.ArgumentParser(description='Train and serve an image classification model')
    parser.add_argument('--data-dir', default='data', help='path to the training data')
    parser.add_argument('--model-path', default='model.pt', help='path to save the trained model')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer')
    parser.add_argument('--log-interval', type=int, default=10, help='log progress every n batches')
    parser.add_argument('--serve', action='store_true', help='serve the model through a Flask API')
    parser.add_argument('--host', default='localhost', help='host to serve the API on')
    parser.add_argument('--port', type=int, default=5000, help='port to serve the API on')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define the data transformations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset and create data loaders
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize the model, loss function, and optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, args.log_interval)

    # Save the trained model
    torch.save(model.state_dict(), args.model_path)

    # Serve the model through a Flask API
    if args.serve:
        app.config['MODEL_PATH'] = args.model_path
        app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
