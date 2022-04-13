from typing import Tuple

from numpy import uint16
from texttable import Texttable
from torch.utils.data import Dataset
import pytorch_lightning as pl
import os
import numpy as np
from configuration import Configuration
from visualization.utilities import display_tensor_as_image, save_tensor_as_image


def test_model(configuration: Configuration,
               dataset: Dataset,
               target_path: str,
               model: pl.LightningModule,
               device):
    total_median_height_error = 0.0
    total_mean_height_error = 0.0
    total_min_height_error = 1000000000
    total_max_height_error = -1000000000

    for i in range(len(dataset)):
        print("Processing test {} of {}".format(i, len(dataset)))
        x, y = dataset[i]
        x_input = x.reshape((1, 3, configuration.input_size[0], configuration.input_size[1]))
        y_output = model.forward(x_input.to(device=device)).detach().cpu()

        save_tensor_as_image(y, os.path.join(target_path, "{}_gt.png".format(i)), uint16, 65536)
        save_tensor_as_image(y_output, os.path.join(target_path, "{}_pd.png".format(i)), uint16, 65536)

        height_range = configuration.max_height - configuration.min_height
        height_error = (y_output - y).abs() * height_range

        mean_height_error = height_error.mean().item()
        median_height_error = height_error.median().item()
        min_height_error = height_error.min().item()
        max_height_error = height_error.max().item()

        total_median_height_error += median_height_error
        total_mean_height_error += mean_height_error

        if min_height_error < total_min_height_error:
            total_min_height_error = min_height_error

        if max_height_error > total_max_height_error:
            total_max_height_error = max_height_error

    table = Texttable()
    table.add_rows([
        ["Mean error", "{} m".format(total_mean_height_error / len(dataset))],
        ["Median error", "{} m".format(total_median_height_error / len(dataset))],
        ["Min error", "{} m".format(total_min_height_error)],
        ["Max error", "{} m".format(total_max_height_error)]
    ])

    return table.draw()
