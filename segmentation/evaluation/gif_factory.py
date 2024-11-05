import os

import torch
from matplotlib import pyplot as plt, animation
from torch import argmax, cat
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import draw_segmentation_masks


from segmentation.data.data import ImageImporter


class GifFactory:
    def __init__(self, model_dir, image_dir):
        self.model_dir = model_dir
        self.image_dir = image_dir
        self.empty_images()

    def generate_images(self):
        device = "cuda:0"

        ii = ImageImporter("infest", sample=True, smaller=(128, 128), only_test=True)
        _, test = ii.get_dataset()
        test_loader = DataLoader(test)
        bigger = Resize((640, 640))
        it = iter(test_loader)
        # next(it)
        next(it)
        next(it)
        next(it)
        next(it)
        X, y = next(it)
        X, y = X.to(device), y.to(device)

        for model_path in sorted(os.listdir(self.model_dir))[::5]:
            if model_path == "slim_model.pt":
                continue

            print(model_path)
            model = torch.load(self.model_dir + model_path)
            model = model.to(device)
            model.eval()
            model.set_width(1)
            with torch.no_grad():
                y_pred = model(X)

                # # Upscale everything for a nicer visualisation
                # X = bigger(X)
                # y = bigger(y)
                # y_pred = bigger(y_pred)

                x_mask = torch.tensor(
                    torch.mul(X[0].clone().detach().cpu(), 255), dtype=torch.uint8
                )

                # Predicted mask
                mask = argmax(y_pred[0].clone().detach(), dim=0)
                weed_mask = torch.where(mask == 1, True, False)[None, :, :]
                lettuce_mask = torch.where(mask == 2, True, False)[None, :, :]
                mask = cat((weed_mask, lettuce_mask), 0)

                image = draw_segmentation_masks(
                    x_mask, mask, colors=["red", "green"], alpha=0.5
                )
                plt.title(model_path[:-3].split("_")[2].zfill(4))
                plt.imshow(image.permute(1, 2, 0))
                # plt.show()
                plt.savefig(
                    "{}{}.{}".format(
                        self.image_dir, model_path[:-3].split("_")[2].zfill(4), "png"
                    )
                )

            # Generate a ground truth image
            ground_truth = argmax(y[0].clone().detach(), dim=0)
            weed_mask = torch.where(ground_truth == 1, True, False)[None, :, :]
            lettuce_mask = torch.where(ground_truth == 2, True, False)[None, :, :]
            mask = cat((weed_mask, lettuce_mask), 0)
            image = draw_segmentation_masks(
                x_mask, mask, colors=["red", "green"], alpha=0.5
            )
            plt.title("Ground truth")
            plt.imshow(image.permute(1, 2, 0))
            plt.savefig("{}ground_truth.{}".format(self.image_dir, "png"))

    def generate_gif(self):
        # Create new figure for GIF
        fig, ax = plt.subplots()

        # Adjust figure so GIF does not have extra whitespace
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.axis("off")
        ims = []

        for image in sorted(os.listdir(self.image_dir)):
            if image == "ground_truth.png":
                continue
            im = ax.imshow(plt.imread(self.image_dir + image), animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("{}/masks.gif".format(self.image_dir))

    def empty_images(self):
        for image in os.listdir(self.image_dir):
            os.remove(self.image_dir + image)


if __name__ == "__main__":
    gf = GifFactory(
        model_dir="../training/garage/infest/0258 vivid paper/",
        image_dir="plots/infest/gif/",
    )
    print("Generating images...", end="")
    gf.generate_images()
    print("DONE\nGenerating gif...", end="")
    gf.generate_gif()
    print("DONE")
