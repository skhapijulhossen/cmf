import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, "src"))

import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from inferencePipeline import preprocess_image, inference
import time
from src.model import UNET

# Streamlit app
def main():
    st.title("Upload Image")

    # Upload image via Streamlit
    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        if st.button("Analyze"):
            with st.spinner("Analyzing"):
                time.sleep(3)
            # Display uploaded image
            st.subheader("Uploaded Image")
            st.image(uploaded_image, caption="Uploaded Image",
                     use_column_width=True)

            # Convert uploaded image to PyTorch tensor
            image = Image.open(uploaded_image)
            image_tensor = preprocess_image(image)

            # Display PyTorch tensor shape
            st.write("PyTorch Tensor Shape:", image_tensor.shape)

            # Generate a new image (placeholder)
            # Replace this with your actual image generation code
            model = UNET(in_channels=3, out_channels=1)
            checkpoint = torch.load("artifacts/unet.pth.tar", map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["state_dict"])
            
            # Inference
            preds = inference(model, image_tensor)
            preds = preds.squeeze(0)
            preds = preds.cpu().detach().numpy()
            new_image = Image.fromarray(preds)

            # Display new generated image
            st.subheader("Generated Image")
            st.image(new_image, caption="Generated Image",
                     use_column_width=True)


if __name__ == '__main__':
    main()