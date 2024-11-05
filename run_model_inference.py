# In short, what we need is a regular U-Net and squeeze U-Net trained on the CoFly-WeedDB dataset
# (using the methodology from the paper, with all weed classes grouped together in a single “Weed”
# label). Ideally, for both networks we should have a basic test bench script that would load the
# network, perform inference on the test set, and report both IoU and precision.
import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from torch.utils.data import DataLoader

from segmentation import settings
from segmentation.data.data import ImageImporter
from segmentation.helpers.metricise import Metricise
from segmentation.models.slim_squeeze_unet import SlimSqueezeUNetCofly, SlimSqueezeUNet
from segmentation.models.slim_unet import SlimUNet


class Inference:
    def __init__(
        self,
        model,
        metrics=None,
        image_resolution=None,
        create_images=False,
        dataset="cofly",
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model.split(".")[0]
        self.create_images = create_images
        self.model = self._load_test_model(model)
        self.model.eval()
        self.image_resolution = image_resolution
        self.metrics = settings.METRICS if metrics is None else metrics
        self.width_distribution = {x: 0 for x in settings.WIDTHS}
        self.width_selection_widths = []
        self.tensor_to_image = ImageImporter("cofly").tensor_to_image
        self.dataset = dataset

    @staticmethod
    def _load_test_model(model):
        parts = model.split("_")
        if parts[1] == "slim":
            model_class = SlimUNet(out_channels=2)
        elif parts[0] == "cofly" or parts[3] == "trans":
            model_class = SlimSqueezeUNetCofly(out_channels=2)
        elif parts[0] == "geok":
            model_class = SlimSqueezeUNet(out_channels=2)
        model_class.load_state_dict(
            torch.load(
                Path(settings.PROJECT_DIR) / f"segmentation/training/garage/" / model
            )
        )
        return model_class

    def _load_test_data(self):
        ii = ImageImporter(
            self.dataset,
            # validation=True,
            smaller=self.image_resolution,
            only_test=True,
        )
        _, test = ii.get_dataset()
        return DataLoader(test, batch_size=1, shuffle=False)

    def run(self):
        test_loader = self._load_test_data()
        # Infer for all available widths of the model
        with torch.no_grad():
            metrics = Metricise(device=self.device, use_adaptive=True, use_oracle=True)
            metrics.evaluate(
                self.model, test_loader, "test", 0, adaptive_knns=self.model_name
            )
            self.results = metrics.report(None)
        return self.results


class Comparator:
    def __init__(self, models, metrics=None, graphs=False):
        self.models = models
        self.metrics = settings.METRICS if metrics is None else metrics
        self.graphs = graphs

    def _draw_tab(self, results):
        for class_name in ["back", "weeds"]:
            for metric in self.metrics:
                values = []
                for model_name, _ in self.models:
                    model_values = []
                    for width in settings.WIDTHS + ["adapt", "oracle"]:
                        if type(width) == float:
                            width = str(int(width * 100))
                        model_values.append(
                            str(
                                round(
                                    results[model_name][
                                        f"test/{width}/{metric}/{class_name}"
                                    ],
                                    4,
                                )
                            )
                        )
                    model_values.append(results[model_name]["test/adapt/width"])
                    model_values.append(
                        results[model_name][f"test/oracle/{metric}/{class_name}/width"]
                    )
                    values.append(model_values)

                fig = go.Figure(
                    data=[
                        go.Table(
                            header={"values": [""] + [x[0] for x in self.models]},
                            cells={
                                "values": [
                                    settings.WIDTHS
                                    + ["adapt", "oracle", "adapt w", "oracle w"]
                                ]
                                + values
                            },
                        )
                    ]
                )
                fig.update_layout({"title": f"{class_name} {metric}"})
                if not os.path.exists("results"):
                    os.mkdir("results")
                fig.write_image(
                    f"results/{class_name}_{metric}.jpg", width=400 * len(self.models)
                )

    def _draw_graph(self, model, results, mean_width):
        for pred_class in ["weeds", "back"]:
            for metric in self.metrics:
                metric_scores = []
                for width in settings.WIDTHS + ["adapt"]:
                    score = round(results[width][metric][pred_class] * 100, 2)
                    metric_scores.append(score)
                x = settings.WIDTHS + ["adaptive"]
                y = metric_scores
                x_ticks = settings.WIDTHS + [round(mean_width, 2)]

                plt.plot(
                    x_ticks[:-1], y[:-1], marker="o", linestyle="-", label="widths"
                )
                plt.plot(
                    [round(mean_width, 2)],
                    [y[-1]],
                    marker="o",
                    color="red",
                    label="adaptive",
                )
                plt.xlabel("widths")
                plt.xticks(x_ticks, x_ticks)
                plt.ylabel(f"{metric} scores")
                plt.title(f"{metric} {pred_class} scores for model {model}")
                plt.legend()
                plt.show()

    def run(self):
        # Select a model from segmentation/training/garage directory and set the
        # image resolution tuple to match the image input size of the model
        results = {}
        graph = False
        for model, size in self.models:
            infer = Inference(
                model,
                image_resolution=(size, size),
                create_images=False,
                dataset="geok",
            )
            infer.run()
            results[model] = infer.results

            if graph:
                x = [k for k in infer.width_distribution]
                y = [v for _, v in infer.width_distribution.items()]
                fig = go.Figure(
                    data=go.Bar(
                        x=x,
                        y=y,
                    )
                )
                fig.update_layout(
                    {"title": "Distribution of selected widths for test images."}
                )
                fig.update_xaxes(type="category")
                fig.show()
                # Only show once as all the plots will be the same
                graph = False

            if self.graphs:
                self._draw_graph(model, results[model], 1)

        self._draw_tab(results)


if __name__ == "__main__":
    # Download the geok dataset and place it in a proper directory.
    # You only have to do this the first time, afterwards the data is ready to go.
    # setup_env()

    models = [
        ("geok_squeeze_128_trans_opt.pt", 128),
        ("geok_squeeze_256_trans_opt.pt", 256),
        ("geok_squeeze_512_trans_opt.pt", 512),
    ]
    comparator = Comparator(models)

    # Run inference for multiple models and display comparative tables
    comparator.run()
