from torch.nn import functional as F
import torch
import wandb
from torch import nn
import lightning.pytorch as pl
import torchmetrics.classification as classification


class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.activation = getattr(nn, config.activation)()
        self.dropout = nn.Dropout(config.dropout)

        self.model = self._process_model(config.model)

        # Metrics
        self.accuracy = classification.MulticlassAccuracy(
            num_classes=config.num_classes
        )
        self.conf_mat = classification.MulticlassConfusionMatrix(
            num_classes=config.num_classes
        )
        self.f1 = classification.MulticlassF1Score(
            num_classes=config.num_classes
        )
        self.precision = classification.MulticlassPrecision(
            num_classes=config.num_classes
        )
        self.recall = classification.MulticlassRecall(
            num_classes=config.num_classes
        )

    def _process_model(self, model_data):
        model = nn.Sequential()
        for layer in model_data:
            name = layer["name"]
            del layer["name"]
            getattr(self, f"_add_{name}")(model, **layer)

        return model

    def _add_conv(
            self,
            model: nn.Sequential,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            add_activation: bool = True,
            add_batch_norm: bool = True,
    ):
        conv_layer = nn.Sequential(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ))
        if self.config.batch_norm and add_batch_norm:
            conv_layer.append(nn.BatchNorm2d(out_channels))
        if add_activation:
            self._add_activation(conv_layer)

        model.append(conv_layer)

    def _add_maxpool(self, model: nn.Sequential, kernel_size: int):
        model.append(nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size))

    def _add_dropout(self, model: nn.Sequential):
        model.append(self.dropout)

    def _add_activation(self, model: nn.Sequential):
        model.append(self.activation)

    def _add_flatten(self, model: nn.Sequential):
        model.append(nn.Flatten())

    def _add_fc(
            self,
            model: nn.Sequential,
            in_features: int,
            out_features: int,
            add_activation: bool = True,
            add_batch_norm: bool = False,
    ):
        fc = nn.Sequential(nn.Linear(in_features, out_features))
        if self.config.batch_norm and add_batch_norm:
            fc.append(nn.BatchNorm1d(out_features))
        if add_activation:
            self._add_activation(fc)
        model.append(fc)

    def _save_model(self, name: str):
        dummy_input = torch.zeros([1] + list(self.config.input_shape),
                                  device=self.device)
        artifact = wandb.Artifact(name=f"{self.config.model_name}_{name}", type="model",
                                  metadata={"epoch": self.current_epoch})

        with artifact.new_file(f"{name}.onnx", mode="wb") as file:
            torch.onnx.export(self, dummy_input, file)

        self.logger.experiment.log_artifact(artifact)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        y_hat = self(x)  # calls forward
        loss = getattr(F, self.config.loss_function)(y_hat, y)
        return y_hat, loss

    def configure_optimizers(self):
        return getattr(torch.optim, self.config.optimizer)(
            self.parameters(),
            lr=self.config.learning_rate,
        )

    def _log_metrics(self, batch, stage):
        x, y = batch
        y_hat, loss = self.loss(x, y)
        preds = torch.argmax(y_hat, 1)

        self.accuracy(preds, y)
        self.log(f"{stage}/accuracy", self.accuracy, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}/epoch", self.current_epoch, on_step=True, on_epoch=False)

    def training_step(self, batch, batch_idx):
        self._log_metrics(batch, stage="train")

    def test_step(self, batch, batch_idx):
        self._log_metrics(batch, stage="test")

    def validation_step(self, batch, batch_idx):
        self._log_metrics(batch, stage="valid")

    def on_test_epoch_end(self):
        self._save_model("final_model")

    def on_validation_epoch_end(self):
        self._save_model("final_model")

