import os
import json

from pathlib import Path
from random import randint

from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from adaptation.inference import AdaptiveWidth
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNetCofly, SlimSqueezeUNet
from segmentation.models.UNet import UNet
from segmentation.models.slim_unet import SlimUNet
from segmentation.helpers.masking import get_binary_masks_infest

from metaseg_io import probs_gt_save
from metaseg_eval import compute_metrics_i
from numpy import floor, ceil

import numpy as np # for softmax
from shapely import Polygon, Point

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
            # self.image_dir = "./test/images/"
            self.image_dir = "./all_ordered/images"
        else:
            raise ValueError("Invalid dataset selected.")

        assert model_architecture in ["pruned", "slim", "squeeze"]
        
        self.model_architecture = model_architecture
        self.image_resolution = image_resolution
        
        model_key = f"{dataset}_{model_architecture}_{image_resolution[0]}"
        if is_trans:
            model_key += "_trans"
        if is_best_fitting:
            model_key += "_opt"
        if model_architecture == "slim":
            self.model = SlimUNet(out_channels=2)
        if model_architecture == "pruned":
            self.model = UNet(out_channels=2).to("cuda")  
        elif dataset == "cofly" or is_trans:
            self.model = SlimSqueezeUNetCofly(out_channels=2)
        elif dataset == "geok":
            self.model = SlimSqueezeUNet(out_channels=2)
        
        print(f"model_key: {model_key}")

        # self.model.load_state_dict(
        #     torch.load(
        #         Path(settings.PROJECT_DIR)
        #         / f"segmentation/training/garage/{model_key}.pt"
        #     )
        # )

        self.model.load_state_dict(
            torch.load(
                Path(settings.PROJECT_DIR)
                / f"segmentation/training/garage/unet_512_pruned_00_iterative_1.pt"
            )
        )

        self.model = self.model.to("cuda")

        self.fixed_image = fixed_image
        self.save_image = save_image
        # self.adaptive_width = AdaptiveWidth(model_key)
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
        # Deconstruct a row

        label = label.split(" ")
        # We consider lettuce as the background, so we skip lettuce label extraction (for now at least).
        if label[0] == "0":
            return None, None
        # Some labels are in a rectangle format, while others are presented as polygons... great fun.
        # Rectangles
        if len(label) == 5:
            class_id, center_x, center_y, width, height = [float(x) for x in label]

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
        # Polygons
        else:
            class_id = label[0]
            # Create a polygon object
            points = [
                (float(label[i]) * image_width, float(label[i + 1]) * image_height)
                for i in range(1, len(label), 2)
            ]
            poly = Polygon(points)
            # We limit the area in which we search for points to make the process a tiny bit faster.
            pixels = []
            for x in range(
                int(floor(min([x[1] for x in points]))),
                int(ceil(max([x[1] for x in points]))),
            ):
                for y in range(
                    int(floor(min([x[0] for x in points]))),
                    int(ceil(max([x[0] for x in points]))),
                ):
                    if Point(y, x).within(poly):
                        pixels.append((x, y))

        return int(class_id), pixels

    def _get_single_image(self, i):
        file_name = f'frame_{i}.jpg' # self._get_random_image_path()
        img_filename = file_name
        print("img file path:")
        print(self.project_path / self.image_dir / file_name)
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


        print(f"file_name: {file_name}")

        # Then, label by label, add to other classes and remove from background.
        file_name = file_name[:-3] + "txt"

        print(f"file_name: {file_name}")

        
        print("label text file path: ")
        print(self.project_path / self.image_dir.replace("images", "labels") / file_name)

        with open(
            self.project_path / self.image_dir.replace("images", "labels") / file_name
        ) as rows:
            labels = [row.rstrip() for row in rows]

            print("len(labels)")
            print(len(labels))

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

    def infer(self, i, fixed=-1):
        # Get a random single image from test dataset.
        # Set the fixed attribute to always obtain the same image
        image, mask, filename = self._get_single_image(i)

        print(f'filename: {filename}')

        # Select and set the model width
        # width = self.adaptive_width.get_image_width(
        #     self.tensor_to_image(image.cpu())[0]
        # )
        # self.model.set_width(width)

        # Get a prediction
        y_pred = self.model.forward(image)
        probs = y_pred.cpu().detach().numpy()
        probs = probs.squeeze(0) # remove batch dimension
        probs = probs.transpose(1,2,0) # rearrange dimensions to (512, 512, 2)
        gt = torch.argmax(mask, dim=1) # convert to class IDs
        gt = gt.squeeze(0) # remove batch dimension 
        gt = gt.cpu().numpy()
        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # Generate overlayed segmentation masks (ground truth and prediction)
        if self.save_image:
            self._generate_images(image, mask, y_pred)

        return results, probs, gt, filename


if __name__ == "__main__":
    # Run this once to download the new dataset
    # setup_env()

    def count_files_in_directory(directory_path):
        return len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

    num_images = count_files_in_directory('./all_ordered/images/')

    # Example usage
    print("Number of files:", count_files_in_directory('./all_ordered/images/'))

    si = SingleImageInference(
        dataset="geok",
        # Tuple of two numbers: (128, 128), (256, 256), or (512, 512)
        image_resolution=(
            512, #256,
            512  #256,
        ),
        # slim,squeeze or pruned
        model_architecture="pruned",
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
    

    for i in range(414, num_images):
        results, probs, gt, filename = si.infer(i)

        # apply softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        probs_softmax = softmax(probs)

        # print("probs_softmax")
        # print(probs_softmax)
        # print("probs_softmax.shape: ")
        # print(probs_softmax.shape)
        # print("filename: ")
        # print(filename)
        # print("ground truth shape: ")
        # print(gt.shape)

        # save the probabilities and ground truth data to hdf5 file
        
        print('probs_gt_save')
        print(filename)

        print('gt.shape')
        print(gt.shape)

        # def visualize_array(array, filename="output_image.png"):
        #     """
        #     Visualize a 256x256 binary array with 1s as red and 0s as green,
        #     and save the result as an image.
            
        #     Parameters:
        #     array (np.array): 256x256 binary array with 1s and 0s
        #     filename (str): The name of the output image file
        #     """
        #     # Ensure the array is 256x256 and binary
        #     if array.shape != (256, 256) or not np.all((array == 0) | (array == 1)):
        #         raise ValueError("Input array must be 256x256 and contain only 0s and 1s")
            
        #     # Create a 3-channel RGB image
        #     img = np.zeros((256, 256, 3), dtype=np.uint8)
        #     img[array == 1] = [255, 0, 0]  # Red for 1s
        #     img[array == 0] = [0, 255, 0]  # Green for 0s
            
        #     # Convert to image and save
        #     output_image = Image.fromarray(img)
        #     output_image.save(filename)
        #     return output_image

        # visualize_array(gt, "output_image.png")

        probs_gt_save(probs_softmax, gt, filename, i)

        # retrieve Metaseg metric
        metaseg_metrics = compute_metrics_i(i)

        # Save MetaSeg metrics as JSON
        json_file_path = f"./metaseg/frame_{i}_metrics.json"
        with open(json_file_path, 'w') as json_file:
            json.dump(metaseg_metrics, json_file, indent=4)
        print(f"Metrics saved to {json_file_path}")


        # print(f"MetaSeg metrics: {metaseg_metrics}")