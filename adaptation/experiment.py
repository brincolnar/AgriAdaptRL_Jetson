#%%
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from segmentation.data.data import ImageImporter
from vegetation_indices.vegetation_indices import VegetationIndices

ii = ImageImporter("cofly")
_, test = ii.get_dataset()
test_images = ii.tensor_to_image(test.X)

print("mean: ", np.mean(test_images[1]))
image = Image.fromarray(test_images[1])
plt.imshow(image)

vegetation_index = VegetationIndices(test_images[1])

ExG = vegetation_index.excess_green_index()
ExR = vegetation_index.excess_red_index()
# ExG_ExR = ExG - ExR
ExG_ExR = vegetation_index.excess_green_excess_red_index(ExG, ExR)
CIVE_index = vegetation_index.colour_index_vegetation_extraction()
binary_CIVE_image = vegetation_index.visualization_CIVE_Otsu_threshold(CIVE_index)

# %%
import pandas as pd

df = pd.read_pickle("/home/agriadapt/agriadapt/test_features.pickle")
print(df)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    precision_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
)


class KnnPrediction:
    def __init__(self, width, train_pickle, test_pickle):
        self.width = width
        self.train_data = pd.read_pickle(train_pickle)
        self.test_data = pd.read_pickle(test_pickle)

    def load_data(self):
        y_train = self.train_data[f"label_{self.width}"]
        X_train = self.train_data.drop([f"label_{self.width}", "index"], axis="columns")

        y_test = self.test_data[f"label_{self.width}"]
        X_test = self.test_data.drop([f"label_{self.width}", "index"], axis="columns")

        return X_train, y_train, X_test, y_test

    def scale_data(self, X_train, X_test):
        scaler = MinMaxScaler()
        X_train_data = X_train.values
        scaled_X_train = scaler.fit_transform(X_train_data)
        scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
        # print(scaled_X_train)

        X_test_data = X_test.values
        scaled_X_test = scaler.transform(X_test_data)
        scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)

        return scaled_X_train, scaled_X_test

    def fit_model(
        self, X_train, y_train, X_test, y_test, save_model=False, draw_graph=True
    ):
        f1_scores = []
        matthews_coeff_scores = []
        precision_scores = []
        accuracy_scores = []
        for i in range(1, 20):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            if save_model:
                save_model = open(f"knn_{self.width}", "wb")
                pickle.dump(knn, save_model)
                save_model.close()

            y_pred = knn.predict(X_test)
            f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
            matthews_coeff_scores.append(matthews_corrcoef(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            print(
                f"Classification Report for width {self.width} for {i} number of neightbors \n {classification_report(y_test, y_pred)}"
            )

        if draw_graph:
            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                f1_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="blue",
                markersize=10,
            )
            plt.title(f"F1 score K value for width {width}")
            plt.xlabel("K value")
            plt.ylabel("F1 score")

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                matthews_coeff_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="red",
                markersize=10,
            )
            plt.title(f"Matthew's coefficiet score K value for width {width}")
            plt.xlabel("K value")
            plt.ylabel("Matthew's coefficiet score")

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                precision_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="green",
                markersize=10,
            )
            plt.title(f"Precision score K value for width {width}")
            plt.xlabel("K value")
            plt.ylabel("Precision score")

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                accuracy_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="orange",
                markersize=10,
            )
            plt.title(f"Accuracy score K value for width {self.width}")
            plt.xlabel("K value")
            plt.ylabel("Accuracy score")

        return y_pred


if __name__ == "__main__":
    knn = KnnPrediction(
        0.75,
        "/home/agriadapt/agriadapt/train_features_with_labels_width_0.75.pickle",
        "/home/agriadapt/agriadapt/test_features_with_labels_width_0.75.pickle",
    )
    X_train, y_train, X_test, y_test = knn.load_data()
    X_train, X_test = knn.scale_data(X_train, X_test)
    y_pred = knn.fit_model(X_train, y_train, X_test, y_test)
