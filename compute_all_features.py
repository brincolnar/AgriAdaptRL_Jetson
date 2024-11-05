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

class FeatureComputation:
    def __init__(
        self,
        image_resolution,
        model_architecture,
        dataset,
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
        # Get a random single image from test dataset.
        # Set the fixed attribute to always obtain the same image
        image, mask, filename = self._get_single_image(file_name)

        # Convert the image from Tensor to a numpy array for processing with OpenCV
        np_image = self.tensor_to_image(image.cpu())[0]

        # Ensure the image has the correct shape (H, W, C) where C should be 3
        if np_image.ndim == 3 and np_image.shape[-1] != 3:
            # Handle incorrect number of channels, possibly convert to 3 channels
            # Example: np_image = np_image[:, :, :3] if there are more than 3 channels
            raise ValueError("Image has an incorrect number of channels.")

        np_image = (np_image * 255).astype(np.uint8)  # Scale from [0,1] to [0,255]

        # Create an instance of all feature classes with the numpy image
        spectral_features = SpectralFeatures(np_image)
        texture_features = TextureFeatures(np_image)
        vegetation_indices = VegetationIndices(np_image)

        # Compute spectral features
        brightness = spectral_features.compute_brightness()
        hue_hist = spectral_features.compute_hue_histogram()
        contrast = spectral_features.compute_contrast()
        saturation = spectral_features.compute_saturation()
        sift_feats = spectral_features.compute_sift_feats()
        
        print(f"Filename: {filename}")
        print(f"Brightness: {brightness}")
        print(f"Hue Histogram: {hue_hist}")
        print(f"Contrast: {contrast}")
        print(f"Saturation: {saturation}")
        print(f"SIFT Features: {sift_feats}")

        # Compute texture features
        glcm = texture_features.compute_glcm()
        texture_contrast = texture_features.contrast_feature(glcm).flatten()
        texture_dissimilarity = texture_features.dissimilarity_feature(glcm).flatten()
        texture_homogeneity = texture_features.homogeneity_feature(glcm).flatten()
        texture_energy = texture_features.energy_feature(glcm).flatten()
        texture_correlation = texture_features.correlation_feature(glcm).flatten()
        texture_asm = texture_features.asm_feature(glcm).flatten()

        print(f"Texture Contrast: {texture_contrast}")
        print(f"Texture Dissimilarity: {texture_dissimilarity}")
        print(f"Texture Homogeneity: {texture_homogeneity}")
        print(f"Texture Energy: {texture_energy}")
        print(f"Texture Correlation: {texture_correlation}")
        print(f"Texture ASM: {texture_asm}")

        # Compute vegetation indices
        excess_green_index = vegetation_indices.excess_green_index().flatten()
        excess_red_index = vegetation_indices.excess_red_index().flatten()
        cive = vegetation_indices.colour_index_vegetation_extraction().flatten()
        exg_exr_ratio = vegetation_indices.excess_green_excess_red_index(excess_green_index, excess_red_index)
        cive_ratio = vegetation_indices.visualization_CIVE_Otsu_threshold(cive)

        print(f"Excess Green Index: {excess_green_index}")
        print(f"Excess Red Index: {excess_red_index}")
        print(f"CIVE: {cive}")
        print(f"ExG-ExR Ratio: {exg_exr_ratio}")
        print(f"CIVE Ratio: {cive_ratio}")

        feature_data = list(brightness) + list(hue_hist) + [contrast] + list(saturation) + [sift_feats]
        
        # Append texture and vegetation features to feature_data
        feature_data.extend([
            *texture_contrast, *texture_dissimilarity, *texture_homogeneity, 
            *texture_energy, *texture_correlation, *texture_asm, 
            *excess_green_index, *excess_red_index, *cive, 
            exg_exr_ratio, cive_ratio
        ])

        # Select and set the model width
        '''
        width = self.adaptive_width.get_image_width(
            self.tensor_to_image(image.cpu())[0]
        )
        '''
        width = 1

        print(f"Selected width: ", width)
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

    si = FeatureComputation(
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            256,
            256,
        ),
        # slim or squeeze
        model_architecture="squeeze",
        dataset="geok",
        # Do you want to generate a mask/image overlay
        save_image=True,
        # Was segmentation model trained using transfer learning
        is_trans=True,
        # Was segmentation model trained with find_best_fitting (utilising
        # model that has the highest difference in iou between widths
        is_best_fitting=False,
    )

    feature_results = []

    all_images_path = "./all/images/"

    for path in os.listdir(all_images_path):
        results, filename, feature_data = si.infer(path)
        print(f"Filename: {filename}")
        print("Feature data: ")
        print(feature_data)
        print("Results: ")
        print(results)
        # Append the results and features to the list
        feature_results.append({
            'filename': filename,
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
            'texture_contrast': feature_data[13],
            'texture_dissimilarity': feature_data[14],
            'texture_homogeneity': feature_data[15],
            'texture_energy': feature_data[16],
            'texture_correlation': feature_data[17],
            'texture_asm': feature_data[18],
            'excess_green_index': feature_data[19],
            'excess_red_index': feature_data[20],
            'cive': feature_data[21],
            'exg_exr_ratio': feature_data[22],
            'cive_ratio': feature_data[23],
            'iou_weeds': results['test/iou/weeds'],
            'iou_back': results['test/iou/back']
    	})

    header = ['Filename', 'Mean Brightness', 'Std Brightness', 'Max Brightness', 'Min Brightness', 
          'Hue Hist Feature 1', 'Hue Hist Feature 2', 'Hue Std', 'Contrast', 
          'Mean Saturation', 'Std Saturation', 'Max Saturation', 'Min Saturation', 
          'SIFT Features']

    header.extend([
        'Texture Contrast', 'Texture Dissimilarity', 'Texture Homogeneity', 
        'Texture Energy', 'Texture Correlation', 'Texture ASM', 
        'Excess Green Index', 'Excess Red Index', 'CIVE', 
        'ExG-ExR Ratio', 'CIVE Ratio'
    ])

    header.extend(['IoU Weeds', 'IoU Back'])

    with open('./features.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for result in feature_results:
            writer.writerow([
                result['filename'],
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
                write_feature(result['texture_contrast']), 
                write_feature(result['texture_dissimilarity']), 
                write_feature(result['texture_homogeneity']), 
                write_feature(result['texture_energy']), 
                write_feature(result['texture_correlation']), 
                write_feature(result['texture_asm']), 
                write_feature(result['excess_green_index']), 
                write_feature(result['excess_red_index']), 
                write_feature(result['cive']), 
                write_feature(result['exg_exr_ratio']), 
                write_feature(result['cive_ratio']),                
                result['iou_weeds'],
                result['iou_back'],
            ])
