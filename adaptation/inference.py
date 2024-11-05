from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from adaptation.image_processing.spectral_features import SpectralFeatures
from adaptation.image_processing.texture_features import TextureFeatures
from adaptation.image_processing.vegetation_features import VegetationIndices
from segmentation.data.data import ImageImporter
from segmentation import settings

import numpy as np
import pandas as pd
import joblib
import warnings

# Andraž is lazy... but these are not a problem.
warnings.filterwarnings(
    "ignore",
)


class AdaptiveWidth:
    def __init__(self, garage_dir):
        garage_dir = Path(settings.PROJECT_DIR) / "adaptation/garage" / garage_dir
        # Initialise the scaler for image scaling
        try:
            train_features = pd.read_pickle(garage_dir / "train_features.pickle").drop(
                ["index"], axis="columns"
            )
        except FileNotFoundError:
            raise ValueError(
                f"Knn models for this segmentation model ({str(garage_dir).split('/')[-1]}) have not yet been generated. "
                "Generate those first, then you can use the adaptive width algorithm."
            )
        self.scaler = MinMaxScaler()
        self.scaler.fit(train_features)

        # Load the models used to predict the width
        self.models = {}
        for width in settings.WIDTHS[0:-1]:
            self.models[width] = joblib.load(garage_dir / f"knn_{width}.pickle")

    def _calculate_image_features(self, image):
        """
        Generates a one row Pandas dataframe with features for a given image.
        """
        spectral_features = SpectralFeatures(image)
        texture_features = TextureFeatures(image)

        vegetation_index = VegetationIndices(image)

        (
            mean_brightness,
            std_brightness,
            max_brightness,
            min_brightness,
        ) = spectral_features.compute_brightness()
        (
            no_bins,
            contrast_hue_hist,
            std_hue_arc,
        ) = spectral_features.compute_hue_histogram()
        contrast = spectral_features.compute_contrast()
        (
            mean_saturation,
            std_saturation,
            max_saturation,
            min_saturation,
        ) = spectral_features.compute_saturation()
        keypoints = spectral_features.compute_sift_feats()

        ExG = vegetation_index.excess_green_index()
        ExR = vegetation_index.excess_red_index()
        ExG_ExR_img = vegetation_index.excess_green_excess_red_index(ExG, ExR)
        CIVE_index = vegetation_index.colour_index_vegetation_extraction()
        binary_CIVE_image = vegetation_index.visualization_CIVE_Otsu_threshold(
            CIVE_index
        )

        glcm_matrix = texture_features.compute_glcm()
        glcm_contrast = np.ravel(texture_features.contrast_feature(glcm_matrix))
        glcm_dissimilarity = np.ravel(
            texture_features.dissimilarity_feature(glcm_matrix)
        )
        glcm_homogeneity = np.ravel(texture_features.homogeneity_feature(glcm_matrix))
        glcm_energy = np.ravel(texture_features.energy_feature(glcm_matrix))
        glcm_correlation = np.ravel(texture_features.correlation_feature(glcm_matrix))
        glcm_asm = np.ravel(texture_features.asm_feature(glcm_matrix))

        column_headers = [
            "index",
            "mean_brightness",
            "std_brightness",
            "max_brightness",
            "min_brightness",
            "no_bins",
            "contrast_hue_hist",
            "std_hue_arc",
            "contrast",
            "mean_saturation",
            "std_saturation",
            "max_saturation",
            "min_saturation",
            "keypoints",
            "ExG_ExR",
            "CIVE_index",
            "glcm_contrast_1",
            "glcm_contrast_2",
            "glcm_contrast_3",
            "glcm_contrast_4",
            "glcm_correlation_1",
            "glcm_correlation_2",
            "glcm_correlation_3",
            "glcm_correlation_4",
            "glcm_dissimilarity_1",
            "glcm_dissimilarity_2",
            "glcm_dissimilarity_3",
            "glcm_dissimilarity_4",
            "glcm_asm_1",
            "glcm_asm_2",
            "glcm_asm_3",
            "glcm_asm_4",
            "glcm_energy_1",
            "glcm_energy_2",
            "glcm_energy_3",
            "glcm_energy_4",
            "glcm_homogeneity_1",
            "glcm_homogeneity_2",
            "glcm_homogeneity_3",
            "glcm_homogeneity_4",
        ]
        features_dictionary = {
            "index": 0,
            "mean_brightness": mean_brightness,
            "std_brightness": std_brightness,
            "max_brightness": max_brightness,
            "min_brightness": min_brightness,
            "no_bins": no_bins,
            "contrast_hue_hist": contrast_hue_hist,
            "std_hue_arc": std_hue_arc,
            "contrast": contrast,
            "mean_saturation": mean_saturation,
            "std_saturation": std_saturation,
            "max_saturation": max_saturation,
            "min_saturation": min_saturation,
            "keypoints": keypoints,
            "ExG_ExR": ExG_ExR_img,
            "CIVE_index": float(binary_CIVE_image),
            "glcm_contrast_1": glcm_contrast[0],
            "glcm_contrast_2": glcm_contrast[1],
            "glcm_contrast_3": glcm_contrast[2],
            "glcm_contrast_4": glcm_contrast[3],
            "glcm_correlation_1": glcm_correlation[0],
            "glcm_correlation_2": glcm_correlation[1],
            "glcm_correlation_3": glcm_correlation[2],
            "glcm_correlation_4": glcm_correlation[3],
            "glcm_dissimilarity_1": glcm_dissimilarity[0],
            "glcm_dissimilarity_2": glcm_dissimilarity[1],
            "glcm_dissimilarity_3": glcm_dissimilarity[2],
            "glcm_dissimilarity_4": glcm_dissimilarity[3],
            "glcm_asm_1": glcm_asm[0],
            "glcm_asm_2": glcm_asm[1],
            "glcm_asm_3": glcm_asm[2],
            "glcm_asm_4": glcm_asm[3],
            "glcm_energy_1": glcm_energy[0],
            "glcm_energy_2": glcm_energy[1],
            "glcm_energy_3": glcm_energy[2],
            "glcm_energy_4": glcm_energy[3],
            "glcm_homogeneity_1": glcm_homogeneity[0],
            "glcm_homogeneity_2": glcm_homogeneity[1],
            "glcm_homogeneity_3": glcm_homogeneity[2],
            "glcm_homogeneity_4": glcm_homogeneity[3],
        }

        df = pd.DataFrame([features_dictionary], columns=column_headers, index=None)
        df = df.drop(["index"], axis="columns")
        return df

    def get_image_width(self, image):
        image = self.scaler.transform(self._calculate_image_features(image))

        for width in settings.WIDTHS[0:-1]:
            try:
                prediction = self.models[width].predict(image)
                if int(prediction) == 1:
                    return width
            except IndexError:
                print("Something is broken")
                continue
        return 1


if __name__ == "__main__":
    ii = ImageImporter("cofly")
    _, test = ii.get_dataset()
    test_images = ii.tensor_to_image(test.X)

    aw = AdaptiveWidth("cofly_slim_128")
    for i in range(len(test_images)):
        print(test_images[i].shape)
        print(aw.get_image_width(test_images[i]))
