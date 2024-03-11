from torch.nn import functional as F
import torch
import wandb
from torch import nn
import lightning.pytorch as pl
import torchmetrics


class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        #self.activation = getattr(nn, config.activation)()
        #self.dropout = nn.Dropout(config.dropout)

        #self.model = self._process_model(config.model)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            getattr(nn, config.activation)(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            getattr(nn, config.activation)(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout)
        )
        self.fc = nn.Linear(1568, config.num_classes)

        # Metrics
        """
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
        """
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=config.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=config.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                              num_classes=config.num_classes)
    """
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

    def _add_max_pool(self, model: nn.Sequential, kernel_size: int):
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
        
    """
    """
    def _save_model(self, name: str):
        dummy_input = torch.zeros([1] + self.config.input_size,
                                  device=self.device)
        artifact = wandb.Artifact(name=f"{self.config.model_name}_{name}", type="model",
                                  metadata={"epoch": self.current_epoch})

        with artifact.new_file(f"{name}.onnx", mode="wb") as file:
            torch.onnx.export(self, dummy_input, file)

        self.logger.experiment.log_artifact(artifact)

    def forward(self, x):
        #x = self.model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        y_hat = self(x)  # calls forward
        loss = F.nll_loss(y_hat, y)
        return y_hat, loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=0.1,
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
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)
        self.log("train/epoch", self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)
        self.log("valid/epoch", self.current_epoch)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        self._save_model("final_model")

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs
        self._save_model("checkpoint_model")

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})
             
    """

    def _save_model(self, filename):
        dummy_input = torch.zeros([1] + [1, 28, 28],
                                  device=self.device)
        artifact = wandb.Artifact(name="model_checkpoint", type="model",
                                  metadata={"epoch": self.current_epoch})

        with artifact.new_file(filename, mode="wb") as file:
            torch.onnx.export(self, dummy_input, file)

        self.logger.experiment.log_artifact(artifact)

    def forward(self, x):
        """
        Defines a forward pass using the Stem-Learner-Task
        design pattern from Deep Learning Design Patterns:
        https://www.manning.com/books/deep-learning-design-patterns
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.fc(x)

        x = F.log_softmax(x, dim=1)

        return x

    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)
        self.log("train/epoch", self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):  # args are defined as part of pl API
        model_filename = "model_final.onnx"
        self._save_model(model_filename)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)
        self.log("valid/epoch", self.current_epoch)

        self.validation_step_outputs.append(logits)

        return logits

    def on_validation_epoch_end(self):
        validation_step_outputs = self.validation_step_outputs

        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        self._save_model(model_filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})

    def configure_optimizers(self):
        return getattr(torch.optim, self.config.optimizer)(self.parameters(),
                                                           lr=self.config.learning_rate)

