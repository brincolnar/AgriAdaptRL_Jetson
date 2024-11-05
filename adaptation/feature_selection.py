import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from segmentation import settings


class FeatureSelection:
    def __init__(self, train_features, test_features):
        self.train_features = pd.read_pickle(train_features)
        self.test_features = pd.read_pickle(test_features)
        # self.data = self.data.drop(["max_saturation"], axis="columns")
        # self.data = self.data.drop(["min_saturation"], axis="columns")
        # self.data = self.data.drop(["index"], axis="columns")

    def select_features_by_threshold(self, threshold):
        corr_matrix_train_features = self.train_features.corr()
        corr_matrix_test_features = self.test_features.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix_train_features,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            xticklabels=True,
            yticklabels=True,
        )
        plt.title("Correlation Matrix Heatmap Train Dataset")
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix_test_features,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            xticklabels=True,
            yticklabels=True,
        )
        plt.title("Correlation Matrix Heatmap Test Dataset")
        plt.show()

        features_to_discard_train_features = set()

        for i in range(len(corr_matrix_train_features.columns)):
            for j in range(i):
                if (
                    corr_matrix_train_features.iloc[i, j] >= threshold
                    and corr_matrix_train_features.columns[i] != "index"
                    and corr_matrix_train_features.columns[j] != "index"
                ):
                    feature_i = corr_matrix_train_features.columns[i]
                    feature_j = corr_matrix_train_features.columns[j]

                    if feature_i not in features_to_discard_train_features:
                        features_to_discard_train_features.add(feature_j)

        selected_features_df_train = self.train_features.drop(
            columns=features_to_discard_train_features
        )

        features_to_discard_test_features = set()

        for i in range(len(corr_matrix_test_features.columns)):
            for j in range(i):
                if (
                    corr_matrix_train_features.iloc[i, j] >= threshold
                    and corr_matrix_train_features.columns[i] != "index"
                    and corr_matrix_train_features.columns[j] != "index"
                ):
                    feature_i = corr_matrix_test_features.columns[i]
                    feature_j = corr_matrix_test_features.columns[j]

                    if feature_i not in features_to_discard_test_features:
                        features_to_discard_test_features.add(feature_j)

        selected_features_df_test = self.test_features.drop(
            columns=features_to_discard_test_features
        )

        return selected_features_df_train, selected_features_df_test

    def filter_dataset(
        self,
        selected_features_train,
        selected_features_test,
        path,
        filename_train,
        filename_test,
    ):
        self.train_features = self.train_features[selected_features_train]
        self.test_features = self.test_features[selected_features_test]

        self.train_features.to_pickle(path / filename_train)
        print(f"Feature Selection executed - train file saved to {filename_train}")
        self.test_features.to_pickle(path / filename_test)
        print(f"Feature Selection executed - test file saved to {filename_test}")


# if __name__ == "__main__":
#     cfs = FeatureSelection(Path(settings.PROJECT_DIR),
#                            "adaptation/garage/geok_squeeze_128_trans_opt/train_features.pickle",
#                            "adaptation/garage/geok_squeeze_128_trans_opt/test_features.pickle")
#     threshold = 0.75
#     selected_features_df_train, selected_features_df_test = cfs.select_features_by_threshold(threshold)
#     print(f"Selected Features Train dataset {selected_features_df_train.columns.tolist()}")
#     print(f"Selected Features Test dataset {selected_features_df_test.columns.tolist()}")
#     cfs.filter_dataset(selected_features_df_train.columns.tolist(), selected_features_df_test.columns.tolist(), Path(settings.PROJECT_DIR),
#                            "adaptation/garage/geok_squeeze_128_trans_opt/train_features.pickle",
#                            "adaptation/garage/geok_squeeze_128_trans_opt/test_features.pickle")
#     print(cfs.train_features)
#     print(cfs.test_features)
