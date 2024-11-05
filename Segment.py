import os
import torch
# from segmentation.models.unet import unet
from garage.UNet import UNet 
from PIL import Image
from torchvision import transforms
from shapely import Polygon, Point
from numpy import floor, ceil
from segmentation.helpers.metricise import Metricise

class Segment:
    '''
    model_path - the path to the model used for inference
    image_dir - directory containing images that will be run
    label_dir - directory containing ground truth
    resolution - resolution of image
    device - GPU/CPU
    '''
    def __init__(self, model_path, image_dir, label_dir, resolution=(512, 512), device='cuda'):
        self.model_path = model_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.resolution = resolution
        self.device = torch.device(device)
        self.model = None
        self._load_model()

    def _load_model(self):
        # Load your segmentation model
        self.model = UNet(out_channels=2)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        return self.model

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

    def _get_single_image(self, file_name, image_dir, label_dir):
        img_filename = file_name
        img = Image.open(os.path.join(image_dir, file_name)).convert('RGB')
        create_tensor = transforms.ToTensor()
        smaller = transforms.Resize(self.resolution)

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
        label_file = file_name[:-3] + "txt"
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            with open(label_path) as rows:
                labels = [row.rstrip() for row in rows]
                for label in labels:
                    class_id, pixels = self._yolov7_label(label, image_width, image_height)
                    if class_id != 1:
                        continue
                    # Change values based on received pixels
                    for pixel in pixels:
                        mask[0][pixel[0]][pixel[1]] = 0
                        mask[class_id][pixel[0]][pixel[1]] = 1

        img = img.to(self.device)
        mask = mask.to(self.device)
        img = img[None, :]
        mask = mask[None, :]

        return img, mask, img_filename

    def infer(self, file_name):
        image, mask, filename = self._get_single_image(file_name, self.image_dir, self.label_dir)
        
        y_pred = self.model(image)
        probs = y_pred.cpu().detach().numpy()
        probs = probs.squeeze(0) # remove batch dimension
        probs = probs.transpose(1,2,0) # rearrange dimensions to (256, 256, 2)
        gt = torch.argmax(mask, dim=1) # convert to class IDs
        gt = gt.squeeze(0) # remove batch dimension 
        gt = gt.cpu().numpy()
        metrics = Metricise()
        metrics.calculate_metrics(mask, y_pred, "test")
        results = metrics.report(None)

        # TODO
        # Generate overlayed segmentation masks (ground truth and prediction)
        #if self.save_image:
        #    self._generate_images(image, mask, y_pred)

        return results, probs, gt, filename

    def process_images(self):
        iou_scores = []
        for file_name in os.listdir(self.image_dir):
            results, probs, gt, filename = self.infer(file_name)
            iou_scores.append(results['test/iou/weeds'])

        avg = sum(iou_scores) / len(iou_scores)
        min_ = min(iou_scores)
        max_ = max(iou_scores)

        return avg, min_, max_