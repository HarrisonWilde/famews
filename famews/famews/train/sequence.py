# ========================================
#
# Pytorch Lightning modules for
# training simple sequence models
#
# ========================================
import logging
import os
from pathlib import Path
from typing import Callable

import gin
import lightgbm
import numpy as np
import pytorch_lightning as pl
import sklearn.base
import torch
import torch.nn as nn
import wandb
from coolname import generate_slug
from joblib import dump, load
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from wandb.lightgbm import wandb_callback

from famews.models.encoders import SequenceModel
from famews.train.utils import (
    binary_task_masked_select,
    identity_logit_transform,
    init_wandb,
    sigmoid_binary_output_transform,
    softmax_multi_output_transform,
)


@gin.configurable("SequenceWrapper")
class SequenceWrapper(pl.LightningModule):
    """
    A Pytorch Lightning Wrapper to train
    sequence models (classifiers/regressors)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = gin.REQUIRED,
        weight_decay: float = 1e-6,
        loss: Callable = gin.REQUIRED,
        label_scaler: Callable = None,
        task: str = "classification/binary",
        l1_reg_emb: float = 0.0,
        print_model: bool = False,
        model_req_mask: bool = False,
        load_batch_data_tuple: bool = False,
    ) -> None:
        """
        Constructor for SequenceWrapper

        Parameters
        ----------
        model : nn.Module
            The model to train
        learning_rate : float
            The learning rate for the optimizer
        weight_decay : float
            The weight decay for the optimizer
        loss : Callable
            The loss function to use
        label_scaler : Callable
            A function to scale the labels
        task : str
            The task type, one of:
                - classification/binary
                - classification/multi
                - regression/single
        l1_reg_emb : float
            The L1 regularization strength for the time-point embedding layer
            usually a single linear layer
        print_model : bool
            Whether to print the model summary
        model_req_mask : bool
            Whether the model requires a mask as input
        load_batch_two_data: bool
            When loading a batch, whether the data element
            is a tuple of (indeces, scaling) (tuple: True) or just features (not a tuple: False)
        """
        super().__init__()

        self.model = model
        if print_model:
            logging.info(f"[{self.__class__.__name__}] Model Architecture:")
            logging.info(self.model)

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.loss = loss
        self.label_scaler = label_scaler
        self.l1_reg_emb = l1_reg_emb
        self.model_req_mask = model_req_mask
        self.load_batch_data_tuple = load_batch_data_tuple
        if self.load_batch_data_tuple:
            logging.info(
                f"[{self.__class__.__name__}] Loading batch as tuple of (indeces, scaling)"
            )

        # set metrics and transforms
        self.output_transform = lambda x: x
        self.logit_transform = identity_logit_transform
        self.set_metrics(task)
        self.smooth_labels = False

        if self.l1_reg_emb > 0.0:
            assert isinstance(self.model, SequenceModel), "Regularizer assumes SequenceModel"
            logging.info(
                f"[{self.__class__.__name__}] Adding L1 regularization to embedding layer with strength {l1_reg_emb}"
            )

    def set_smooth(self, smooth: bool):
        self.smooth_labels = smooth

    def set_metrics(self, task: str):

        if task == "classification/binary":
            self.output_transform = sigmoid_binary_output_transform
            self.logit_transform = binary_task_masked_select
            self.metrics = {
                "train": {
                    "AuPR": BinaryAveragePrecision(validate_args=False),
                    "AuROC": BinaryAUROC(validate_args=False),
                },
                "val": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
                "test": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
            }

        elif task == "classification/multi":
            self.output_transform = softmax_multi_output_transform
            self.metrics = {}
            # TODO: implement
            raise NotImplementedError(
                f"Multi-Class Classification not yet supported, need to add label scaling"
            )

        elif task == "regression/single":
            self.output_transform = lambda x: x
            self.metrics
            # TODO: implement
            raise NotImplementedError(f"Regression not yet supported, need to add label scaling")

        else:
            raise ValueError(f"Unsupported task type: {task}")

    def l1_regularization(self):
        embedding_layer = self.model.encoder.time_step_embedding
        n_params = sum(
            len(
                p.reshape(
                    -1,
                )
            )
            for p in embedding_layer.parameters()
        )
        return sum(torch.abs(p).sum() for p in embedding_layer.parameters()) / n_params

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        if self.model_req_mask:
            return self.model(x, mask)

        return self.model(x)

    def _get_batch(self, batch):
        if self.load_batch_data_tuple:
            if len(batch) == 5:  # there is also patient ids
                data_indeces, data_scaling, labels, mask, patient_ids = batch
                return (data_indeces, data_scaling), labels, mask, patient_ids
            else:
                data_indeces, data_scaling, labels, mask = batch
                return (data_indeces, data_scaling), labels, mask
        else:
            return batch

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        batch = self._get_batch(batch)
        if len(batch) == 3:
            data, labels, mask = batch
            patient_ids = None
        else:
            data, labels, mask, patient_ids = batch

        logits = self(data, mask)  # calls forward
        preds, _ = self.output_transform((logits, labels))

        return preds, labels, patient_ids

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        # Task specific loss
        loss = self.loss(logits, labels)

        # L1 regularization loss
        if self.l1_reg_emb > 0.0:
            l1_loss = self.l1_reg_emb * self.l1_regularization()
            loss += l1_loss
        else:
            l1_loss = 0.0

        return loss, l1_loss

    def training_step(self, batch, batch_idx: int):

        data, labels, mask = self._get_batch(batch)
        logits = self(data, mask)  # calls forward

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits_flat, labels_flat = self.logit_transform(logits, loss_labels, mask)
        _, metric_labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, _ = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss}

        preds_flat, labels_trans = self.output_transform((logits_flat, metric_labels_flat))
        for metric in self.metrics["train"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def training_epoch_end(self, outputs) -> None:

        # nan to num as in distributed training some batches may be empty towards end of epoch
        train_loss = np.mean([np.nan_to_num(x["loss"].detach().cpu().numpy()) for x in outputs])
        self.log("train/loss", train_loss, prog_bar=True)  # logger=False)

        for name, metric in self.metrics["train"].items():
            metric_val = metric.compute()
            self.log(f"train/{name}", metric_val)
            metric.reset()

    def validation_step(self, batch, batch_idx: int):

        data, labels, mask = self._get_batch(batch)
        logits = self(data, mask)  # calls forward

        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels

        logits_flat, labels_flat = self.logit_transform(logits, loss_labels, mask)
        _, metric_labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, _ = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss}

        preds_flat, labels_trans = self.output_transform((logits_flat, metric_labels_flat))
        for metric in self.metrics["val"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def validation_epoch_end(self, outputs) -> None:

        # nan to num as in distributed training some batches may be empty towards end of epoch
        val_loss = np.mean([np.nan_to_num(x["loss"].detach().cpu().numpy()) for x in outputs])
        self.log("val/loss", val_loss, prog_bar=True)  # logger=False)

        for name, metric in self.metrics["val"].items():
            metric_val = metric.compute()
            self.log(f"val/{name}", metric_val)
            metric.reset()

    def test_step(self, batch, batch_idx: int, dataset_idx: int = 0):

        data, labels, mask = self._get_batch(batch)  # Never smoothed labels
        if self.smooth_labels:
            loss_labels = labels[..., 1]  # We dont use smooth label for loss
            metric_labels = labels[..., 0]
        else:
            loss_labels = labels
            metric_labels = labels
        logits = self(data, mask)  # calls forward

        logits_flat, labels_flat = self.logit_transform(logits, metric_labels, mask)
        loss, _ = self.compute_loss(logits_flat, labels_flat)
        step_dict = {"loss": loss}

        preds_flat, labels_trans = self.output_transform((logits_flat, labels_flat))
        for metric in self.metrics["test"].values():
            metric.update(preds_flat.detach().cpu(), labels_trans.detach().cpu())

        return step_dict

    def test_epoch_end(self, outputs):
        test_loss = np.mean([x["loss"].item() for x in outputs])
        self.log("test/loss", test_loss, prog_bar=True)  # logger=False)

        for name, metric in self.metrics["test"].items():
            metric_val = metric.compute()
            self.log(f"test/{name}", metric_val)
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer


@gin.configurable("TabularWrapper")
class TabularWrapper(object):
    """
    A  Wrapper to train sequence as tabular models (classifiers/regressors).
    """

    def __init__(
        self,
        model: sklearn.base.BaseEstimator,
        label_scaler: Callable = None,
        task: str = "classification/binary",
    ) -> None:
        super().__init__()

        self.model = model
        self.label_scaler = label_scaler

        # set metrics and transforms
        self.output_transform = lambda x: x
        self.logit_transform = identity_logit_transform
        self.set_metrics(task)
        self.trained = False
        self.task = task

    def set_metrics(self, task: str):

        if task == "classification/binary":
            self.output_transform = sigmoid_binary_output_transform
            self.logit_transform = binary_task_masked_select
            self.metrics = {
                "train": {
                    "AuPR": BinaryAveragePrecision(validate_args=False),
                    "AuROC": BinaryAUROC(validate_args=False),
                },
                "val": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
                "test": {
                    "AuPR": BinaryAveragePrecision(validate_args=True),
                    "AuROC": BinaryAUROC(validate_args=True),
                },
            }

        elif task == "classification/multi":
            self.output_transform = softmax_multi_output_transform
            self.metrics = {}
            # TODO: implement
            raise NotImplementedError(
                f"Multi-Class Classification not yet supported, need to add label scaling"
            )

        elif task == "regression/single":
            self.output_transform = lambda x: x
            self.metrics = {}
            # TODO: implement
            raise NotImplementedError(f"Regression not yet supported, need to add label scaling")

        else:
            raise ValueError(f"Unsupported task type: {task}")

    def train(
        self,
        X,
        y,
        eval_set=None,
        eval_metric: list[str] = ["binary"],
        early_stopping_rounds: int = None,
        wandb_project: str = None,
    ):

        # TODO implement early stopping for non lightgbm methods
        callbacks = []
        if early_stopping_rounds is not None:
            es_callback = lightgbm.early_stopping(early_stopping_rounds, first_metric_only=True)
            callbacks += [es_callback]
        if wandb_project is not None:
            params = {}
            params["metric"] = sorted(eval_metric)
            init_wandb(wandb_project, config=params)
            callbacks += [wandb_callback()]
        if len(callbacks) == 0:
            callbacks = None

        self.model.fit(X, y, eval_set=eval_set, eval_metric=eval_metric, callbacks=callbacks)
        self.trained = True
        # if wandb_project is not None:
        #     log_summary(self.model.booster_, save_model_checkpoint=True)

    def evaluate(self, X, y, prefix, wandb_project: str = None):
        init_wandb(wandb_project)

        # TODO account for model that need predict_proba
        pred = self.model.predict_proba(X)[:, 1]
        metric_results = {}
        for name, metric in self.metrics[prefix].items():
            metric_results["/".join([prefix, name])] = metric(
                torch.tensor(pred), torch.tensor(y).int()
            )

        if wandb_project is not None:
            wandb.log(metric_results)

        return metric_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities on a batch of samples
        """
        return self.model.predict_proba(X)

    def load_model(self, weights_path):
        self.model = load(weights_path)
        self.trained = True

    def save_model(self, weights_path):
        if os.path.isdir(weights_path):
            weights_path = os.path.join(weights_path, "model.joblib")
        dump(self.model, weights_path)
