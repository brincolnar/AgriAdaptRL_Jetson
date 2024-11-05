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

import csv

class SingleImageInference:
    def __init__(
        self,
        dataset,
        image_resolution,
        model_architecture,
        fixed_image=-1,
        save_image=False,
        is_trans=False,
        is_best_fitting=False,
    ):
        self.project_path = Path(settings.PROJECT_DIR)
        if dataset == "infest":
            self.image_dir = "segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/test/images/"
        elif dataset == "geok":
            #self.image_dir = "segmentation/data/geok/test/images/"
            self.image_dir = "./test/images/"
        else:
            raise ValueError("Invalid dataset selected.")
        assert model_architecture in ["slim", "squeeze"]
        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
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
        self.fixed_image = fixed_image
        self.save_image = save_image
        self.adaptive_width = AdaptiveWidth(model_key)
        self.tensor_to_image = ImageImporter(dataset).tensor_to_image
        self.random_image_index = -1

    def _get_random_image_path(self):
        images = os.listdir(self.project_path / self.image_dir)
        if self.fixed_image < 0:
            self.random_image_index = randint(0, len(images) - 1)
            return images[self.random_image_index]
        else:
            return images[
                self.fixed_image if self.fixed_image < len(images) else len(images) - 1
            ]

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

    def _get_single_image(self):
        file_name = self._get_random_image_path()
        img_filename = file_name
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
        file_name = file_name[:-3] + "txt"
        with open(
            self.project_path / self.image_dir.replace("images", "labels") / file_name
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

        return img, mask, img_filename

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

    def infer(self, fixed=-1):
        # Get a random single image from test dataset.
        # Set the fixed attribute to always obtain the same image
        image, mask, filename = self._get_single_image()

        # Convert the image from Tensor to a numpy array for processing with OpenCV
        np_image = self.tensor_to_image(image.cpu())[0]

        # Ensure the image has the correct shape (H, W, C) where C should be 3
        if np_image.ndim == 3 and np_image.shape[-1] != 3:
            # Handle incorrect number of channels, possibly convert to 3 channels
            # Example: np_image = np_image[:, :, :3] if there are more than 3 channels
            raise ValueError("Image has an incorrect number of channels.")

        np_image = (np_image * 255).astype(np.uint8)  # Scale from [0,1] to [0,255]

        # Create an instance of SpectralFeatures with the numpy image
        spectral_features = SpectralFeatures(np_image)


        # Compute features
        brightness = spectral_features.compute_brightness()
        hue_hist = spectral_features.compute_hue_histogram()
        contrast = spectral_features.compute_contrast()
        saturation = spectral_features.compute_saturation()
        sift_feats = spectral_features.compute_sift_feats()

        # Print all features and IoU
        print(f"Filename: {filename}")
        print(f"Brightness: {brightness}")
        print(f"Hue Histogram: {hue_hist}")
        print(f"Contrast: {contrast}")
        print(f"Saturation: {saturation}")
        print(f"SIFT Features: {sift_feats}")

        feature_data = list(brightness) + list(hue_hist) + [contrast] + list(saturation) + [sift_feats]


        # Select and set the model width
        width = self.adaptive_width.get_image_width(
            self.tensor_to_image(image.cpu())[0]
        )
        self.model.set_width(width)

        # Get a prediction
        y_pred = self.model.forward(image)
        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # Generate overlayed segmentation masks (ground truth and prediction)
        if self.save_image:
            self._generate_images(image, mask, y_pred)

        return results, filename, feature_data


if __name__ == "__main__":
    # Run this once to download the new dataset
    # setup_env()

    si = SingleImageInference(
        # geok (new dataset)
        # dataset="geok",
        dataset="geok",
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            256,
            256,
        ),
        # slim or squeeze
        model_architecture="squeeze",
        # Set to a positive integer to select a specific image from the dataset, otherwise random
        fixed_image=-1,
        # Do you want to generate a mask/image overlay
        save_image=True,
        # Was segmentation model trained using transfer learning
        is_trans=True,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )

    feature_results = []

    for i in range(50):
        results, filename, feature_data = si.infer()
        print(f"Filename: {filename}")
        print("Feature data: ")
        print(feature_data)
        print("Results: ")
        print(results)
        # Append the results and features to the list
        feature_results.append({
            #'filename': filename,
            'mean_brightness': feature_data[0],
            'std_brightness': feature_data[1],
            'max_brightness': feature_data[2],
            'min_brightness': feature_data[3],
            'hue_hist_feature_1': feature_data[4],
            'hue_hist_feature_2': feature_data[5],
            'hue_std': feature_data[6],
            'contrast': feature_data[7],
            'mean_saturation': feature_data[8],
            'std_saturation': feature_data[9],
            'max_saturation': feature_data[10],
            'min_saturation': feature_data[11],
            'sift_features': feature_data[12],
            'iou_weeds': results['test/iou/weeds'],
            'iou_back': results['test/iou/back']
    	})

    header = ['Mean Brightness', 'Std Brightness', 'Max Brightness', 'Min Brightness', 
          'Hue Hist Feature 1', 'Hue Hist Feature 2', 'Hue Std', 'Contrast', 
          'Mean Saturation', 'Std Saturation', 'Max Saturation', 'Min Saturation', 
          'SIFT Features', 'IoU Weeds', 'IoU Back']

    with open('./features.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for result in feature_results:
            writer.writerow([
                #result['filename'], 
                result['mean_brightness'], 
            	result['std_brightness'], 
            	result['max_brightness'], 
            	result['min_brightness'], 
            	result['hue_hist_feature_1'], 
            	result['hue_hist_feature_2'], 
            	result['hue_std'], 
            	result['contrast'], 
            	result['mean_saturation'], 
            	result['std_saturation'], 
            	result['max_saturation'], 
            	result['min_saturation'], 
            	result['sift_features'], 
            	result['iou_weeds'], 
            	result['iou_back']
            ])
