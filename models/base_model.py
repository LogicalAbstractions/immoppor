import os.path

import torch
from pytorch_lightning import LightningModule
from torchinfo import summary

from configuration import Configuration
from data.index_dataset import take
from testing.test_runner import test_model


class BaseModel(LightningModule):

    def __init__(self, configuration: Configuration):
        super().__init__()
        self.configuration = configuration
        self.loss = configuration.create_loss(self)

        self.example_input_array = torch.rand(
            (1, 3, configuration.input_size[0], configuration.input_size[1]))

        self.learning_rate = configuration.learning_rate

    def training_step(self, batch, idx):
        x, y = batch

        z = self.forward(x)

        loss = self.loss(y, z)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch

        z = self.forward(x)

        loss = self.loss(y, z)
        self.log("validation_loss", loss)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.eval()

        epoch_path = os.path.join(self.logger.log_dir, "epoch-" + str(self.current_epoch))
        os.makedirs(epoch_path, exist_ok=True)

        test_path = os.path.join(epoch_path, "tests")
        os.makedirs(test_path, exist_ok=True)

        self.to_torchscript(os.path.join(epoch_path, "model.pt"), method='trace')
        # model.to_onnx(configuration.resolve_output_path("model.onnx"))

        report = test_model(self.configuration,
                            self.configuration.testing_dataset,
                            test_path,
                            self, self.device)

        with open(os.path.join(epoch_path, "report.txt"), 'w') as f:
            f.write(str(self.configuration))
            f.write('\n')
            f.write(report)
            print(report)

        self.train()

    def configure_optimizers(self):
        optimizer = self.configuration.create_optimizer(self)
        scheduler = self.configuration.create_lr_scheduler(optimizer, self)

        result = {
            "optimizer": optimizer,
            "monitor": "training_loss"
        }

        if scheduler is not None:
            result["lr_scheduler"] = scheduler

        return result

    def summary(self):
        summary(self, input_size=self.example_input_array.shape)
