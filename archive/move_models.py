import os
from pathlib import Path

import torch

from segmentation import settings
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNet, SlimSqueezeUNetCofly
from segmentation.models.slim_unet import SlimUNet

#
# print(torch.load("test.pt"))
# 0 / 0

directory = Path(settings.PROJECT_DIR) / "segmentation/training/garage/"
for model_name in sorted(
    os.listdir(Path(settings.PROJECT_DIR) / "segmentation/training/garage/")
):
    if model_name == "runs":
        continue
    print(model_name)
    model = torch.load(directory / model_name)
    print(model.__class__.__name__)
    if model.__class__.__name__ == "SlimUNet":
        model_class = SlimUNet(2)
        model_class.load_state_dict(model.state_dict())
        torch.save(model_class.state_dict(), directory / model_name)
    elif model.__class__.__name__ == "SlimSqueezeUNet":
        model_class = SlimSqueezeUNet(2)
        model_class.load_state_dict(model.state_dict())
        torch.save(model_class.state_dict(), directory / model_name)
    elif model.__class__.__name__ == "SlimSqueezeUNetCofly":
        model_class = SlimSqueezeUNetCofly(2)
        model_class.load_state_dict(model.state_dict())
        torch.save(model_class.state_dict(), directory / model_name)

# modelObj = torch.load(directory / "geok_squeeze_128.pt")
# torch.save(modelObj.state_dict(), "test.pt")
#
# model = SlimPrunedSqueezeUNet()
# model.load_state_dict(torch.load("test.pt"))


# except RuntimeError:
#     # print(f"{model} corrupted.")
#     continue
# print(model)
