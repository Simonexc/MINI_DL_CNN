from torch.nn import functional as F
import torch
import wandb
from torch import nn
import lightning.pytorch as pl
import torchmetrics
from settings import CLASS_NAMES
import wandb


class ConstructModelMixin:
    config: wandb.sdk.wandb_config.Config

    def _construct_model(self, model_data: list[dict]):
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

    def _add_avg_pool(self, model: nn.Sequential, kernel_size: int):
        model.append(nn.AvgPool2d(kernel_size=kernel_size))

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


class ResidualBlock(nn.Module, ConstructModelMixin):
    def __init__(
            self,
            config: wandb.sdk.wandb_config.Config,
            main_path_model: list[dict],
            shortcut_path_model: list[dict],
    ):
        super().__init__()
        self.config = config

        self.main_path = self._construct_model(main_path_model)
        self.shortcut_path = self._construct_model(shortcut_path_model)

    def forward(self, x):
        main_path = self.main_path(x)
        shortcut_path = self.shortcut_path(x)
        return main_path + shortcut_path


class Net(pl.LightningModule, ConstructModelMixin):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = self._construct_model(config.model)
        # Metrics
        self.test_probabilities = []
        self.test_true_values = []
        metric_train_params = {"average": "weighted"}
        metric_train_params["task"] = "multiclass"
        metric_train_params["num_classes"] = self.config.num_classes

        self.train_acc = torchmetrics.Accuracy(**metric_train_params)
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.num_classes,
            average="weighted",
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.num_classes,
            average="weighted",
        )
        self.train_prec = torchmetrics.Precision(**metric_train_params)
        self.valid_prec = torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.test_prec = torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.train_recall = torchmetrics.Recall(**metric_train_params)
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.valid_recall = torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.train_f1score = torchmetrics.F1Score(**metric_train_params)
        self.test_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.valid_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )

    def _add_res_block(
            self,
            model: nn.Sequential,
            main_path_model: list[dict],
            shortcut_path_model: list[dict],
            add_activation: bool = True,
            add_batch_norm: bool = True,
    ):
        res_block = nn.Sequential(
            ResidualBlock(self.config, main_path_model, shortcut_path_model)
        )
        if add_activation and shortcut_path_model:
            self._add_activation(res_block)
        if self.config.batch_norm and add_batch_norm and shortcut_path_model:
            res_block.append(nn.BatchNorm2d(shortcut_path_model[-1]["out_channels"]))
        model.append(res_block)

    def _save_model(self, filename):
        dummy_input = torch.zeros([1] + self.config.input_size,
                                  device=self.device)
        full_filename = f"{filename}_{self.config.model_name}"
        artifact = wandb.Artifact(name=full_filename, type="model",
                                  metadata={"epoch": self.current_epoch})

        with artifact.new_file(full_filename + ".onnx", mode="wb") as file:
            torch.onnx.export(self, dummy_input, file)

        self.logger.experiment.log_artifact(artifact)

    def forward(self, x):
        x = self.model(x)

        return x

    def loss(self, x, y):
        logits = self(x)  # calls forward
        loss = F.cross_entropy(logits, y)

        return logits, loss

    def on_test_epoch_start(self):
        self.test_probabilities = []
        self.test_true_values = []

    def training_step(self, batch, batch_idx):
        xs, ys = batch
        preds, loss = self.loss(xs, ys)
        preds = torch.argmax(preds, 1)
        if self.config.cutmix_add != "none":
            ys = torch.argmax(ys, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True, on_step=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/accuracy', self.train_acc, on_epoch=True, on_step=True)
        self.train_prec(preds, ys)
        self.log("train/precision", self.train_prec, on_epoch=True, on_step=True)
        self.train_recall(preds, ys)
        self.log("train/recall", self.train_recall, on_epoch=True, on_step=True)
        self.train_f1score(preds, ys)
        self.log("train/f1_score", self.train_f1score, on_epoch=True, on_step=True)
        self.log("train/epoch", self.current_epoch)

        return loss

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_acc(preds, ys)
        self.log("test/accuracy", self.test_acc, on_step=False, on_epoch=True)
        self.test_prec(preds, ys)
        self.log("test/precision", self.test_prec, on_epoch=True, on_step=False)
        self.test_recall(preds, ys)
        self.log("test/recall", self.test_recall, on_epoch=True, on_step=False)
        self.test_f1score(preds, ys)
        self.log("test/f1_score", self.test_f1score, on_epoch=True, on_step=False)

        self.test_probabilities.append(torch.exp(logits))
        self.test_true_values.append(ys)

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.log("valid/loss", loss, on_epoch=True, on_step=False)
        self.valid_acc(preds, ys)
        self.log('valid/accuracy', self.valid_acc, on_epoch=True, on_step=False)
        self.valid_prec(preds, ys)
        self.log("valid/precision", self.valid_prec, on_epoch=True, on_step=False)
        self.valid_recall(preds, ys)
        self.log("valid/recall", self.valid_recall, on_epoch=True, on_step=False)
        self.valid_f1score(preds, ys)
        self.log("valid/f1_score", self.valid_f1score, on_epoch=True, on_step=False)
        self.log("valid/epoch", self.current_epoch)

        return logits

    def on_test_epoch_end(self):
        self._save_model("final")

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
        self._save_model("checkpoint")

    def configure_optimizers(self):
        kwargs = dict()
        if self.config.optimizer == "Adam":
            kwargs["betas"] = (self.config.beta1, self.config.beta2)
        else:
            kwargs["momentum"] = self.config.beta1
        return getattr(torch.optim, self.config.optimizer)(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_penalty,
            **kwargs
        )

