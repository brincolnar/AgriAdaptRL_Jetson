from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from numpy import unique
from sklearn.metrics import precision_score
from torch import flatten
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryPrecision

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.drive_fetch import setup_env
from segmentation.helpers.masking import get_binary_masks_infest

CLASSES = ["back", "weeds", "lettuce"]


def precision_original(test_loader, model):
    scores = {x: [] for x in CLASSES}
    precs = []
    for X, y in test_loader:
        X = X.to("cuda:0")
        mask_pred = model.forward(X)

        # Calculate precision for all classes and print them
        # Iterate through classes
        for j in range(y.shape[1]):
            # If everything is predicted as negative
            if len(unique(mask_pred[0][j].cpu())) == 1:
                scores[CLASSES[j]].append(0)
            else:
                mask_pred = get_binary_masks_infest(mask_pred)
                s = datetime.now()
                result = precision_score(
                    [int(xx) for x in y[0][j].cpu().numpy().tolist() for xx in x],
                    [
                        int(xx)
                        for x in mask_pred[0][j].cpu().numpy().tolist()
                        for xx in x
                    ],
                )
                precs.append((datetime.now() - s).total_seconds())
                scores[CLASSES[j]].append(result)
    print("Prec time({}): {}".format(len(precs), sum(precs)))
    return scores


def precision_matmul(test_loader, model):
    scores = {x: [] for x in CLASSES}
    precs = []
    for X, y in test_loader:
        X = X.to("cuda:0")
        mask_pred = model.forward(X)
        mask_pred = get_binary_masks_infest(mask_pred)

        # Calculate precision for all classes and print them
        # Iterate through classes
        for j in range(y.shape[1]):
            true = flatten(y[:, j]).tolist()
            pred = flatten(mask_pred[:, j]).tolist()
            s = datetime.now()
            scores[CLASSES[j]].append(precision_score(true, pred, zero_division=0))
            precs.append((datetime.now() - s).total_seconds())

    print("Prec time({}): {}".format(len(precs), sum(precs)))
    return scores


def precision_torchmetrics(test_loader, model):
    scores = {x: [] for x in CLASSES}
    precs = []
    precision_calculation = BinaryPrecision(validate_args=False).to("cuda:0")
    for X, y in test_loader:
        X = X.to("cuda:0")
        y = y.to("cuda:0")
        mask_pred = model.forward(X)
        mask_pred = get_binary_masks_infest(mask_pred)

        # Calculate precision for all classes and print them
        # Iterate through classes

        for j in range(y.shape[1]):
            s = datetime.now()
            result = precision_calculation(y[:, j], mask_pred[:, j])
            precs.append((datetime.now() - s).total_seconds())
            result = float(result.cpu())
            scores[CLASSES[j]].append(result)
    for key in scores:
        scores[key] = round(np.mean(scores[key]), 3)
    return scores


if __name__ == "__main__":
    # Before running the script create a local_settings.py
    # file in the segmentation directory then set the PROJECT_DIR
    # variable to point to the project's root directory.

    # Prepare the env -- download data and models (only needed on data/models update)
    # Comment the line when not needed
    # setup_env(data=False, models=True)

    # Load the dataset ("infest" -> labelled part of the dataset from Geo-K)
    ii = ImageImporter("infest", only_test=True)
    _, test = ii.get_dataset()
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    for model_path in [
        "training/garage/no_decay.pt",
        "training/garage/linear.pt",
        "training/garage/exponential.pt",
        "training/garage/squeeze.pt",
        "training/garage/big_squeeze.pt",
    ]:
        print("Evaluating model {}".format(model_path))
        results = {}
        # Load the model
        model = torch.load(Path(settings.PROJECT_DIR) / model_path)
        model.eval()
        with torch.no_grad():
            # Iterate through different widths of the network
            for width_mult in settings.WIDTHS:
                model.set_width(width_mult)
                scores = precision_torchmetrics(test_loader, model)
                results[width_mult] = scores
        print("{:<8} {:<8} {:<8} {:<8}".format("width", "back", "weeds", "lettuce"))
        for key in results:
            print(
                "{:<8} {:<8} {:<8} {:<8}".format(
                    key,
                    results[key]["back"],
                    results[key]["weeds"],
                    results[key]["lettuce"],
                )
            )
        print()
