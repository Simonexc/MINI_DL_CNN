from torch.nn import functional as F
import torch
import wandb
from torch import nn
import lightning.pytorch as pl
import torchmetrics
from settings import CLASS_NAMES


class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = self._process_model(config.model)
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
        self.test_probabilities = []
        self.test_true_values = []
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=config.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=config.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                              num_classes=config.num_classes)

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
        model.append(nn.Dropout(self.config.dropout))

    def _add_activation(self, model: nn.Sequential):
        model.append(getattr(nn, self.config.activation)())

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

    def _save_model(self, filename):
        dummy_input = torch.zeros([1] + [3, 32, 32],
                                  device=self.device)
        artifact = wandb.Artifact(name="model_checkpoint", type="model",
                                  metadata={"epoch": self.current_epoch})

        with artifact.new_file(filename, mode="wb") as file:
            torch.onnx.export(self, dummy_input, file)

        self.logger.experiment.log_artifact(artifact)

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        y_hat = self(x)  # calls forward
        loss = F.nll_loss(y_hat, y)
        return y_hat, loss

    def on_test_epoch_start(self):
        self.test_probabilities = []
        self.test_true_values = []

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

        self.test_probabilities.append(torch.exp(logits))
        self.test_true_values.append(ys)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)
        self.log("valid/epoch", self.current_epoch)

        return logits

    def on_test_epoch_end(self):
        self._save_model("final_model")

        flattened_probabilities = torch.flatten(
            torch.cat(self.test_probabilities)).view(-1, self.config.num_classes).to(
            "cpu")
        flattened_true_values = torch.flatten(torch.cat(self.test_true_values)).to(
            "cpu")
        self.logger.experiment.log(
            {"test/roc": wandb.plot.roc_curve(flattened_true_values,
                                               flattened_probabilities, labels=CLASS_NAMES)}
        )
        #self.logger.experiment.log(
        #   {"test/confusion_matrix": wandb.sklearn.plot_confusion_matrix(flattened_true_values.numpy().tolist(), torch.argmax(flattened_probabilities, dim=1).numpy().tolist(), CLASS_NAMES)}
        #)
        self.logger.experiment.log(
            {"test/confusion_matrix": wandb.plot.confusion_matrix(probs=flattened_probabilities, y_true=flattened_true_values.numpy().tolist(), class_names=CLASS_NAMES)}
        )

    def on_validation_epoch_end(self):
        if self.global_step == 0:
            return
        self._save_model("checkpoint_model")

    def configure_optimizers(self):
        return getattr(torch.optim, self.config.optimizer)(self.parameters(),
                                                           lr=self.config.learning_rate)

