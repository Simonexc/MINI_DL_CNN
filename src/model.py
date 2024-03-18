from torch.nn import functional as F
import torch
import wandb
from torch import nn
from torchvision.models import (resnet50, vgg19, densenet121)
import lightning.pytorch as pl
import torchmetrics
from settings import CLASS_NAMES


class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model = self._process_model(config.model)
        # Metrics
        self.test_probabilities = []
        self.test_true_values = []
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.num_classes,
            average="weighted",
        )
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
        self.train_prec = torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.valid_prec = torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.test_prec = torchmetrics.Precision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.train_recall = torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.test_recall = torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.valid_recall = torchmetrics.Recall(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.train_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.test_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        self.valid_f1score = torchmetrics.F1Score(
            task="multiclass", num_classes=config.num_classes, average="weighted"
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

    def _add_resnet50(self, model: nn.Sequential):
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():  # freeze all conv layers
            param.requires_grad = False
        for param in resnet.fc.parameters():  # unfreeze classifier
            param.requires_grad = True
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features,
                              out_features=self.config.num_classes, bias=True)
        model.append(resnet)

    def _add_vgg19(self, model: nn.Sequential):
        vgg = vgg19(pretrained=True)
        for param in vgg.parameters():  # freeze all conv layers
            param.requires_grad = False
        for param in vgg.fc.parameters():  # unfreeze classifier
            param.requires_grad = True
        vgg.classifier._modules['6'] = nn.Linear(in_features=4096,
                                                 out_features=self.config.num_classes,
                                                 bias=True)
        model.append(vgg)

    def _add_densenet121(self, model: nn.Sequential):
        densenet = densenet121(pretrained=True)
        for param in densenet.parameters():  # freeze all conv layers
            param.requires_grad = False
        for param in densenet.classifier.parameters():  # unfreeze classifier
            param.requires_grad = True
        densenet.classifier = nn.Linear(in_features=1024,
                                        out_features=self.config.num_classes,
                                        bias=True)
        model.append(densenet)

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

        self.test_acc(preds, ys)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
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
        self.valid_acc(preds, ys)

        self.log("valid/loss", loss, on_epoch=True, on_step=False)
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

