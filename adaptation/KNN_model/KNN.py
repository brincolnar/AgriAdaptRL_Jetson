import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    matthews_corrcoef,
    classification_report,
)


class KnnPrediction:
    def __init__(
        self, width, train_dataframe, test_dataframe, k_neighbours, garage_dir=""
    ):
        self.model = None
        self.width = width
        self.train_data = train_dataframe
        self.test_data = test_dataframe
        self.k_neighbours = k_neighbours
        self.garage_dir = garage_dir

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
        # print(scaled_X_test)

        return scaled_X_train, scaled_X_test, scaler

    def fit_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        save_model=False,
        metrics=False,
    ):
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbours)
        knn.fit(X_train, y_train)
        self.model = knn
        if save_model:
            save_model = open(self.garage_dir / f"knn_{self.width}.pickle", "wb")
            pickle.dump(knn, save_model)
            save_model.close()
            print(f"KNN model for {self.width} saved to file")

        y_pred = knn.predict(X_test)

        if metrics:
            f1_scores = []
            matthews_coeff_scores = []
            precision_scores = []
            accuracy_scores = []
            for i in range(1, 20):
                knn = KNeighborsClassifier(n_neighbors=i)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                f1_scores.append(f1_score(y_test, y_pred, average="weighted"))
                matthews_coeff_scores.append(matthews_corrcoef(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred))
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                print(
                    f"Classification Report for width {self.width} for {i} number of neightbors \n {classification_report(y_test, y_pred)}"
                )

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                f1_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="blue",
                markersize=10,
            )
            plt.title(f"F1 score K value for width {self.width}")
            plt.xlabel("K value")
            plt.ylabel("F1 score")
            plt.show()

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                matthews_coeff_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="red",
                markersize=10,
            )
            plt.title(f"Matthew's coefficiet score K value for width {self.width}")
            plt.xlabel("K value")
            plt.ylabel("Matthew's coefficiet score")
            plt.show()

            plt.figure(figsize=(15, 6))
            plt.plot(
                range(1, 20),
                precision_scores,
                linestyle="dashed",
                marker="o",
                markerfacecolor="green",
                markersize=10,
            )
            plt.title(f"Precision score K value for width {self.width}")
            plt.xlabel("K value")
            plt.ylabel("Precision score")
            plt.show()

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
            plt.show()

        return y_pred
