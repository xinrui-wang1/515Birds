import torch
import torchvision.transforms as transforms
from PIL import Image
from models import get_model
from config import Config
import argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(image_path, model_path):
    # Load the model
    model = torch.load(model_path)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load and preprocess the image
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        _, preds = torch.max(output, 1)

    # Get the class label
    with open('class_labels.txt') as f:
        class_labels = [line.strip() for line in f.readlines()]

    predicted_label = class_labels[preds.item()]
    confidence = probs[0][preds.item()].item()

    return predicted_label, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classification Predictions')
    parser.add_argument('image_path', type=str, help='path to image file')
    parser.add_argument('model_path', type=str, help='path to model file')
    args = parser.parse_args()

    label, confidence = predict(args.image_path, args.model_path)
    print(f'Prediction: {label}, Confidence: {confidence:.2f}')