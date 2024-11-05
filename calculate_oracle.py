import os
from pathlib import Path
from random import randint

from PIL import Image
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from adaptation.inference import AdaptiveWidth
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNetCofly, SlimSqueezeUNet
from segmentation.models.slim_unet import SlimUNet
from segmentation.helpers.masking import get_binary_masks_infest

import numpy as np # for softmax

from spectral_features import SpectralFeatures
from texture_features import TextureFeatures
from vegetation_features import VegetationIndices

import csv

class OracleComputation:
    def __init__(
        self,
        image_resolution,
        model_architecture,
        dataset,
        preformance_analysis,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        self.project_path = Path(settings.PROJECT_DIR)
        print(f"project_path", self.project_path)
        self.image_dir = "./all/images/"
        assert model_architecture in ["slim", "squeeze"]
        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
        self.dataset = dataset
        self.preformance_analysis = preformance_analysis
        model_key = f"{dataset}_{model_architecture}_{image_resolution[0]}"
        if is_trans:
            model_key += "_trans"
        if is_best_fitting:
            model_key += "_opt"
        if model_architecture == "slim":
            self.model = SlimUNet(out_channels=2)
        elif dataset == "cofly" or is_trans:
            self.model = SlimSqueezeUNetCofly(out_channels=2)
        elif dataset == "geok":
            self.model = SlimSqueezeUNet(out_channels=2)
        print(f"model_key: {model_key}")
        self.model.load_state_dict(
            torch.load(
                Path(settings.PROJECT_DIR)
                / f"segmentation/training/garage/{model_key}.pt"
            )
        )
        self.save_image = save_image
        self.adaptive_width = AdaptiveWidth(model_key)
        self.tensor_to_image = ImageImporter(dataset).tensor_to_image
        self.random_image_index = -1

    def _yolov7_label(self, label, image_width, image_height):
        """
        Implement an image mask generation according to this:
        https://roboflow.com/formats/yolov7-pytorch-txt
        """
        #print("label: ")
        #print(label)
        # Deconstruct a row
        class_id, center_x, center_y, width, height = [
            float(x) for x in label.split(" ")
        ]

        # Get center pixel
        center_x = center_x * image_width
        center_y = center_y * image_height

        # Get border pixels
        top_border = int(center_x - (width / 2 * image_width))
        bottom_border = int(center_x + (width / 2 * image_width))
        left_border = int(center_y - (height / 2 * image_height))
        right_border = int(center_y + (height / 2 * image_height))

        # Generate pixels
        pixels = []
        for x in range(left_border, right_border):
            for y in range(top_border, bottom_border):
                pixels.append((x, y))

        return int(class_id), pixels

    def _get_single_image(self, file_name):
        img = Image.open(self.project_path / self.image_dir / file_name)
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.image_resolution)

        img = smaller(img)
        img = create_tensor(img)

        image_width = img.shape[1]
        image_height = img.shape[2]

        # Constructing the segmentation mask
        # We init the whole tensor as the background
        mask = torch.cat(
            (
                torch.ones(1, image_width, image_height),
                torch.zeros(1, image_width, image_height),
            ),
            0,
        )


        # Then, label by label, add to other classes and remove from background.
        label_file_name = file_name[:-3] + "txt"
        with open(
            self.project_path / self.image_dir.replace("images", "labels") / label_file_name
        ) as rows:
            labels = [row.rstrip() for row in rows]
            for label in labels:
                class_id, pixels = self._yolov7_label(label, image_width, image_height)
                if class_id != 1:
                    continue
                # Change values based on received pixels
                for pixel in pixels:
                    mask[0][pixel[0]][pixel[1]] = 0
                    mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to("cuda:0")
        mask = mask.to("cuda:0")
        img = img[None, :]
        mask = mask[None, :]

        return img, mask, file_name

    def _generate_images(self, X, y, y_pred):
        if not os.path.exists("results"):
            os.mkdir("results")
        # Generate an original rgb image with predicted mask overlay.
        x_mask = torch.tensor(
            torch.mul(X.clone().detach().cpu(), 255), dtype=torch.uint8
        )
        x_mask = x_mask[0]

        # Draw predictions
        y_pred = y_pred[0]
        mask = torch.argmax(y_pred.clone().detach(), dim=0)
        weed_mask = torch.where(mask == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)

        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_pred.jpg")

        # Draw ground truth
        mask = y.clone().detach()[0]
        weed_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
        # lettuce_mask = torch.where(mask[2] == 1, True, False)[None, :, :]
        # mask = torch.cat((weed_mask, lettuce_mask), 0)
        image = draw_segmentation_masks(x_mask, weed_mask, colors=["red"], alpha=0.5)
        plt.imshow(image.permute(1, 2, 0))
        plt.savefig(f"results/{self.random_image_index}_true.jpg")

    def infer(self, file_name, fixed=-1):
        image, mask, filename = self._get_single_image(file_name)

        # Convert the image from Tensor to a numpy array for processing with OpenCV
        np_image = self.tensor_to_image(image.cpu())[0]

        # Ensure the image has the correct shape (H, W, C) where C should be 3
        if np_image.ndim == 3 and np_image.shape[-1] != 3:
            # Handle incorrect number of channels, possibly convert to 3 channels
            # Example: np_image = np_image[:, :, :3] if there are more than 3 channels
            raise ValueError("Image has an incorrect number of channels.")

        np_image = (np_image * 255).astype(np.uint8)  # Scale from [0,1] to [0,255]

        best_width = None
        best_metric = float('-inf')  # Initialize with a very low value

        modelsize_to_preformance = {
            0.25: 0.0,
            0.5: 0.0,
            0.75: 0.0,
            1.0: 0.0
        }

        # Try all model widths
        for width in [0.25, 0.5, 0.75, 1.0]:
            self.model.set_width(width)

            # Get a prediction
            y_pred = self.model.forward(image)
            metrics = Metricise()
            metrics.calculate_metrics(mask, y_pred, "test")
            results = metrics.report(None)

            # Generate overlayed segmentation masks (ground truth and prediction)
            if self.save_image:
                self._generate_images(image, mask, y_pred)

            # Choose the metric to determine the best width. Here, using IoU for 'weeds' class as an example
            current_metric = results.get('test/iou/weeds', 0)

            # Set iou/weeds as model preformance
            modelsize_to_preformance[width] = current_metric

            # Update best width if current metric is better
            if current_metric > best_metric:
                best_metric = current_metric
                best_width = width

        if self.preformance_analysis == True:
            return modelsize_to_preformance

        return results, filename, best_width

# Helper functions
def write_feature(feature):
    """ Helper function to write a feature to CSV.
        If it's a single value, write it directly.
        If it's an iterable (like a list), join its elements.
    """
    if isinstance(feature, (list, tuple, np.ndarray)):
        return ','.join(map(str, feature))
    return str(feature)

if __name__ == "__main__":
    # Run this once to download the new dataset
    # setup_env()

    si = OracleComputation(
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            256,
            256,
        ),
        # slim or squeeze
        model_architecture="squeeze",
        dataset="geok",
        preformance_analysis=False,
        # Do you want to generate a mask/image overlay
        save_image=True,
        # Was segmentation model trained using transfer learning
        is_trans=True,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )

    best_widths_results = []

    all_images_path = "./all/images/"

    for path in os.listdir(all_images_path):
        results, filename, best_width = si.infer(path)
        print(f"Filename: {filename}, Best Width: {best_width}")
        print("Results: ")
        print(results)
        best_widths_results.append({'filename': filename, 'best_width': best_width})

    # Write results to CSV
    with open('./oracle_widths.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Best Width'])

        for result in best_widths_results:
            writer.writerow([result['filename'], result['best_width']])
