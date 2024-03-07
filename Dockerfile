# Use official PyTorch image as base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire app directory into the container
COPY . /app

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
# CMD ["streamlit", "run", "app.py"]
# torch @ https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=0d4e8c52a1fcf5ed6cfc256d9a370fcf4360958fc79d0b08a51d55e70914df46
# torchaudio @ https://download.pytorch.org/whl/cu121/torchaudio-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=676bda4042734eda99bc59b2d7f761f345d3cde0cad492ad34e3aefde688c6d8
# torchdata==0.7.0
# torchsummary==1.5.1
# torchtext==0.16.0
# torchvision @ https://download.pytorch.org/whl/cu121/torchvision-0.16.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=e76e78d0ad43636c9884b3084ffaea8a8b61f21129fbfa456a5fe734f0affea9
