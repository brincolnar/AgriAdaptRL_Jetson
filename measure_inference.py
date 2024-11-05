import os
from pathlib import Path
from random import randint, seed
from time import time

import torch
from torchvision import transforms
from PIL import Image

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.models.UNet import UNet  # Assuming UNet is defined in segmentation.models


class BatchInference:
    def __init__(self, image_resolution, model, dataset, image_paths):
        self.project_path = Path(settings.PROJECT_DIR)
        self.image_dir = "./data/ordered_train_test/all/images"
        self.image_resolution = image_resolution
        self.model_key = f"{dataset}_{model}_{image_resolution[0]}"
        self.image_paths = image_paths

        self.model = UNet(out_channels=2).to("cuda")
        self.model.load_state_dict(torch.load(f"segmentation/training/garage/{model}"))
        self.model.eval()
        self.tensor_to_image = ImageImporter(dataset).tensor_to_image

    def _load_image(self, file_name):
        img = Image.open(self.project_path / self.image_dir / file_name)
        img = transforms.Resize(self.image_resolution)(img)
        img_tensor = transforms.ToTensor()(img).to("cuda")
        return img_tensor[None, :]  # Add batch dimension

    def infer(self, batch_size):
        images = torch.cat([self._load_image(file_name) for file_name in self.image_paths[:batch_size]], dim=0)
        
        # Warm-up run
        with torch.no_grad():
            _ = self.model.forward(images)
        
        # Actual timed inference
        with torch.no_grad():
            start_time = time()
            y_pred = self.model.forward(images)
            inference_time = time() - start_time
        return inference_time


if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8, 16, 32]
    dataset = "geok"
    image_resolution = (256, 256)

    models = [
        'unet_512_pruned_00_iterative_1.pt', 
        'unet_512_pruned_05_iterative_1.pt',
        'unet_512_pruned_025_iterative_1.pt',
        'unet_512_pruned_075_iterative_1.pt'
    ]

    # Set a seed for reproducibility in selecting random images
    seed(42)

    # Load the available images and select a subset to use for consistent testing
    project_path = Path(settings.PROJECT_DIR)
    image_dir = "./data/ordered_train_test/all/images"
    image_paths = os.listdir(project_path / image_dir)
    selected_images = [image_paths[randint(0, len(image_paths) - 1)] for _ in range(max(batch_sizes))]

    # Loop through each model and each batch size, measuring the average inference time
    for model in models:
        print(f"Evaluating model: {model}")
        for batch_size in batch_sizes:
            bi = BatchInference(image_resolution=image_resolution, model=model, dataset=dataset, image_paths=selected_images)

            # Perform inference multiple times to get an average time
            times = [bi.infer(batch_size) for _ in range(100)]
            avg_time = sum(times) / len(times)

            print(f"Model: {model}, Batch size: {batch_size}, Average inference time: {avg_time:.8f} seconds")
