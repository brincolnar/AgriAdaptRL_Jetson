import os
from zipfile import ZipFile
from pathlib import Path

import gdown

from segmentation import settings


def setup_env(data=True, models=True):
    project_dir = Path(settings.PROJECT_DIR)

    # Download the data
    if data:
        # For infest (not interesting anymore)
        # setup_folder(
        #     "https://drive.google.com/file/d/1w2jbor9QR3iUQ0V1pphsmNHw4ZZdgf_f/view?usp=share_link",
        #     project_dir
        #     / "segmentation/data/agriadapt/NN_labeled_samples_salad_infesting_plants.v1i.yolov7pytorch/",
        # )
        # Cofly
        # setup_folder(
        #     "https://drive.google.com/file/d/1GJpML5gFrBO0LZ5rGTOkaJUv8474sJOm/view?usp=sharing",
        #     project_dir / "segmentation/data/cofly/",
        # )
        # Geok
        setup_folder(
            "https://drive.google.com/file/d/1bXuFyBnpoP2sUD5-e7lqAl4bnVN1WV5H/view?usp=sharing",
            project_dir / "segmentation/data/geok/",
        )

    # Download the models (are now just uploaded to git for convenience)
    if models:
        # setup_folder(
        #     "https://drive.google.com/file/d/1F8wtjxjESxWRomzAxuR43tJ2bOptTMfT/view?usp=share_link",
        #     project_dir / "segmentation/training/garage/",
        # )
        # setup_folder(
        #     "https://drive.google.com/file/d/1252exvJsH_ljJafm7qbQzXld3S1rxmuY/view?usp=sharing",
        #     project_dir / "segmentation/training/garage/",
        # )
        # setup_folder(
        #     "https://drive.google.com/file/d/1RFT-lvGjsLyJXNnJic8bw_OMmDqrycno/view?usp=sharing",
        #     project_dir / "segmentation/training/garage/",
        # )
        # setup_folder(
        #     "https://drive.google.com/file/d/1sfQ-rpP8EZnBLLbKF6hIBDyfAxcga-Dw/view?usp=sharing",
        #     project_dir / "segmentation/training/garage/",
        # )
        pass


def setup_folder(url, path):
    # Check if path directory exists and create it if not
    if not os.path.exists(path):
        os.makedirs(path)
    # Download the zip file
    gdown.download(url, str(path / "tmp.zip"), quiet=False, fuzzy=True)
    # Unzip the downloaded file
    with ZipFile(path / "tmp.zip", "r") as zip_ref:
        zip_ref.extractall(path)
    # Remove the zip file
    os.remove(path / "tmp.zip")
