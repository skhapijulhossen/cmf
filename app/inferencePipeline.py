import torch
import numpy as np
import cv2
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Function to convert image to PyTorch tensor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to fit model input size
        transforms.ToTensor(),           # Convert image to PyTorch tensor
        transforms.Normalize(            # Normalize image with mean and standard deviation
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to Inference 
def inference(image, model,  device="cpu"):
    """Inferece forged regions in a given image.
    Args:
        loader (torch.utils.data.DataLoader): Data loader for the image.
        model (torch.nn.Module): Model for the inference.
        device (str): Device to use for inference.
    Returns:
        torch.Tensor: Inference result.
    """
    try:
        # Move the model to the desired device
        device = torch.device("cpu")  # or torch.device("cuda") or torch.device("cuda:0")
        model.to(device)
        # model.eval()
        with torch.no_grad():
                image = image.to(device=torch.device('cpu'))
                preds = torch.sigmoid(model(image))
                preds = (preds > 0.5).float()
        return preds
    except Exception as e:
        logging.error(e)

