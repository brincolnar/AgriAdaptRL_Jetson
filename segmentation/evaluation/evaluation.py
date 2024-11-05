from datetime import datetime
from time import sleep

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import unique
from torch import argmax, tensor, cat
from torch.cuda import memory_summary, mem_get_info
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.utils import save_image, draw_segmentation_masks
from plotly import graph_objects as go

import segmentation.settings as settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise


class EvaluationHelper:
    def __init__(
        self,
        device="cuda:0",
        dataset="cofly",
        visualise=False,
        class_num=5,
    ):
        self.device = device
        self.dataset = dataset
        self.visualise = visualise
        self.class_num = class_num

        self.test_loader = None

    def import_data(self):
        """
        Import dataset that is used for evaluation.
        """
        ii = ImageImporter(self.dataset, sample=False)
        train, test = ii.get_dataset()
        self.test_loader = DataLoader(test, batch_size=1, shuffle=False)

    def evaluate(self, model_path, device="cuda:0"):
        """
        Evaluate a given method with requested metrics.
        Optionally also visualise segmentation masks/overlays.
        """
        if not self.test_loader:
            print("Import data first.")
            return
        self.model = torch.load(model_path)
        self.model.eval()
        # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        starter, ender = starter, ender = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)

        i = 0
        jac_scores = {x: [] for x in settings.width_mult_list}
        run_times = {x: [] for x in settings.width_mult_list}
        with torch.no_grad():
            for X, y in self.test_loader:
                first = True
                X = X.to(device)
                torch.cuda.empty_cache()
                w = 1
                # Warmup to have a fair running time measurements
                self.model.forward(X)
                for width_mult in settings.width_mult_list:
                    self.model.set_width(width_mult)
                    starter.record()
                    images = self.model.forward(X)
                    ender.record()
                    torch.cuda.synchronize()
                    run_times[width_mult].append(starter.elapsed_time(ender))

                    # Save segmentation masks
                    if self.visualise:
                        self._save_images(X, y, images, i, w, first)

                    # Calculate Jaccard Index
                    y_jac = argmax(y[0], dim=0).to(device)
                    image = argmax(images[0], dim=0).to(device)
                    jac_scores[width_mult].append(
                        self.evaluate_jaccard(y_jac, image, self.class_num)
                    )
                    first = False

                    w += 1
                i += 1

        return jac_scores, run_times

    def evaluate_jaccard(self, y, pred, class_num):
        """
        Use the Jaccard Index to evaluate model's performance.
        """
        jaccard = MulticlassJaccardIndex(class_num).to(self.device)
        return jaccard(y, pred).cpu()

    def save_images(self, X, y, y_pred, batch, width):
        for i, (x, test, y) in enumerate(zip(X, y, y_pred)):
            # Generate an original rgb image with predicted mask overlay.
            x_mask = torch.tensor(
                torch.mul(x.clone().detach().cpu(), 255), dtype=torch.uint8
            )
            # To draw predictions
            mask = argmax(y.clone().detach(), dim=0)
            weed_mask = torch.where(mask == 1, True, False)[None, :, :]
            lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
            mask = cat((weed_mask, lettuce_mask), 0)

            image = draw_segmentation_masks(
                x_mask, mask, colors=["red", "green"], alpha=0.5
            )
            plt.imshow(image.permute(1, 2, 0))
            # For saving predictions
            plt.savefig(
                "plots/infest/slim_unet/{}_{}_prediction.png".format(
                    str(batch).zfill(3), width
                )
            )

            # To draw labels
            mask = test.clone().detach()[1:]
            weed_mask = torch.where(mask[0] == 1, True, False)[None, :, :]
            lettuce_mask = torch.where(mask[1] == 1, True, False)[None, :, :]
            mask = cat((weed_mask, lettuce_mask), 0)
            # print(unique(mask[0]))
            # print(unique(mask[1]))
            # 0 / 0

            # print(weed_mask.shape)
            # print(lettuce_mask.shape)
            # print(x_mask.shape)
            # print(mask.shape)
            # print(unique(mask.cpu()))
            image = draw_segmentation_masks(
                x_mask, mask, colors=["red", "green"], alpha=0.5
            )
            plt.imshow(image.permute(1, 2, 0))
            # plt.show()

            # For saving ground truth
            plt.savefig(
                "plots/infest/slim_unet/{}_000_groundtruth.png".format(
                    str(batch).zfill(3)
                )
            )

            # x = x.cpu().float().permute(1, 2, 0)
            # test = argmax(test, dim=0).float()
            # y = argmax(y, dim=0).cpu().float()
            # Generate separate images
            # plt.imshow(x)
            # plt.savefig("plots/{}_{}_org.png".format(batch, i))
            # if first:
            #     plt.imshow(test)
            #     plt.savefig("plots/slim_net/{}_{}_0_mask.png".format(batch, i))
            # plt.imshow(y)
            # plt.savefig("plots/{}_{}_pred.png".format(batch, i))

    @staticmethod
    def report_memory():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        print("Total: {}".format(t / 1024 / 1024))
        print("Reserved: {}".format(r / 1024 / 1024))
        print("Allocated: {}".format(a / 1024 / 1024))
        print()


if __name__ == "__main__":
    dataset = ImageImporter("infest", only_test=True, smaller=(128, 128))
    eh = EvaluationHelper()
    model = torch.load("../training/garage/squeeze.pt")

    _, test = dataset.get_dataset()
    batch = 0
    for X, y in test:
        # Add batch dimension
        X = X[None, :]
        y = y[None, :]
        X = X.to("cuda:0")
        y = y.to("cuda:0")

        for width in settings.WIDTHS:
            model.set_width(width)
            y_pred = model(X)
            eh.save_images(X, y, y_pred, batch, str(int(width * 100)).zfill(3))
        batch += 1

    # Generate images
    # device = "cuda:0"
    # eh = EvaluationHelper(device=device, dataset="infest", class_num=3, visualise=True)
    # eh.import_data()
    #
    # model = "garage/slim_model_1000.pt"
    # jac_scores, run_times = eh.evaluate(model)

    # Generate plots
    # device = "cuda:0"
    # eh = EvaluationHelper(device=device, dataset="infest", class_num=3, visualise=True)
    # eh.import_data()
