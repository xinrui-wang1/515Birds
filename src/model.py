import torch
import torchvision.transforms as transforms
from PIL import Image

def main(image_path):
    # Load the saved model
    model = YourModelClass()
    model.load_state_dict(torch.load('model.ckpt'))

    # Preprocess the input image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Return the predicted species
    return predicted.item()