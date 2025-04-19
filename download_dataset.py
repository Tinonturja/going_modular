"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""
import os
import requests
import zipfile
from pathlib import Path

# setup path to data folder
data_path = Path("data")
image_path = data_path / 'pizza_steak_sushi'

if image_path.is_dir():
    print(f"{image_path} already exists")
else:
    image_path.mkdir(parents=True,
                     exist_ok=True)

    # Download the data file
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    with open(image_path / 'pizza_steak_sushi.zip', 'wb') as f:
        print("Downloading the file")
        f.write(request.content)

    # Unzip the file
    with zipfile.ZipFile(image_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        print("Unzipping the file")
        zip_ref.extractall(image_path)

    # Delete the zip file
    os.remove(image_path / 'pizza_steak_sushi.zip')
