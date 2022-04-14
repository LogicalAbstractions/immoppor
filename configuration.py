from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from texttable import Texttable
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.index_dataset import IndexDataset, random_split, take
from data.height_dataset import HeightDataset
from losses.height_loss import HeightLoss

from losses.rsme_loss import RMSELoss

import pathlib
import os

import pytorch_lightning as pl


class Configuration:
    def __init__(self,
                 dataset_path="e:/datasets/wien/output",
                 data_splits: tuple[float, float] = (0.7, 0.2),
                 epochs: int = 100,
                 batch_size: int = 4,
                 learning_rate: float = 1e-3,
                 num_dataloader_workers: int = 0,
                 min_height: float = -300.0,
                 max_height: float = 300.0,
                 training_dataset_limit=None,
                 validation_dataset_limit=None,
                 testing_dataset_limit=25,
                 use_mixed_precision: bool = False):
        self.dataset_path = dataset_path
        self.root_dataset = HeightDataset(dataset_path)

        test_dataset_size = 1.0 - data_splits[0] - data_splits[1]
        test_split = data_splits[1] / (test_dataset_size + data_splits[1])

        self.training_dataset, validation_and_test_dataset = random_split(self.root_dataset, data_splits[0])
        self.validation_dataset, self.testing_dataset = random_split(validation_and_test_dataset, test_split)

        if training_dataset_limit is not None:
            self.training_dataset = take(self.training_dataset, training_dataset_limit)

        if validation_dataset_limit is not None:
            self.validation_dataset = take(self.validation_dataset, validation_dataset_limit)

        if testing_dataset_limit is not None:
            self.testing_dataset = take(self.testing_dataset, testing_dataset_limit)

        self.training_data_loader = DataLoader(self.training_dataset, batch_size, shuffle=True,
                                               num_workers=num_dataloader_workers)
        self.validation_data_loader = DataLoader(self.validation_dataset, batch_size,
                                                 num_workers=num_dataloader_workers)

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Read one sample to infer input/output sizes
        x, y = self.training_dataset[0]
        self.input_size = (x.shape[1], x.shape[2])
        self.output_size = (y.shape[1], y.shape[2])

        self.use_mixed_precision = use_mixed_precision
        self.min_height = min_height
        self.max_height = max_height

    def create_trainer(self, module: LightningModule):
        logger = TensorBoardLogger("logs", log_graph=True)

        precision = 32

        if self.use_mixed_precision:
            precision = 16

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=self.epochs, logger=logger, precision=precision)

        return trainer

    def create_loss(self, module: LightningModule):
        return HeightLoss()

    def create_optimizer(self, module: LightningModule) -> Optimizer:
        return Adam(module.parameters())

    def create_lr_scheduler(self, optimizer: Optimizer, module: LightningModule):
        return ReduceLROnPlateau(optimizer, verbose=True)

    def __str__(self):
        table = Texttable()
        table.add_rows([
            ["dataset_path", self.dataset_path],
            ["input_size", self.input_size],
            ["output_size", self.output_size],
            ["min_height", self.min_height],
            ["max_height", self.max_height],
            ["batch_size", self.batch_size],
            ["epochs", self.epochs],
            ["learning_rate", self.learning_rate],
            ["training_samples", len(self.training_dataset)],
            ["validation_samples", len(self.validation_dataset)],
            ["testing_samples", len(self.testing_dataset)],
            ["mixed_precision", self.use_mixed_precision]
        ], False)

        return table.draw()
