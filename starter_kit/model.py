#!/bin/env python
#
# Built for the CI 2026 hackathon starter kit


# System modules
import abc
import logging
import os.path
from typing import Any, Dict, List, Tuple

# External modules
import torch
import torch.nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# Internal modules
from starter_kit import lat_weights

main_logger = logging.getLogger(__name__)


class CSVLogger:
    r"""
    Simple CSV logger for training metrics.

    This logger buffers metric rows and writes them to a CSV file when
    flushed.
    """

    def __init__(self, csv_path: str) -> None:
        r"""
        Initialize the CSV logger.

        Parameters
        ----------
        csv_path : str
            Destination path for the CSV log file.

        Notes
        -----
        Rows are buffered until flush() is called.
        """
        self.csv_path = csv_path
        self._rows_to_log: List[Dict] = []

    def log_row(self, row_dict: Dict) -> None:
        r"""
        Buffer a row for later CSV writing.
        Parameters
        ----------
        row_dict : Dict
            Dictionary containing scalar metrics for one step.

        Notes
        -----
        The row is appended to the internal buffer.
        """
        self._rows_to_log.append(row_dict)

    def flush(self) -> None:
        r"""
        Flush buffered rows to disk.

        Writes all buffered rows to the CSV file, appending if the file
        already exists.

        Notes
        -----
        Uses pandas.DataFrame.to_csv() for serialization.
        """
        if not self._rows_to_log:
            return
        import pandas as pd

        df = pd.DataFrame(self._rows_to_log)
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.csv_path, index=False)
        self._rows_to_log = []


class BaseModel(abc.ABC):
    r"""
    Base model for PyTorch models.

    Handles optimizer setup, training and validation loops, checkpointing,
    and CSV logging.
    """

    _best_loss = float("inf")
    _optimizer: torch.optim.Optimizer

    def __init__(
        self,
        network: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        store_path: str,
        device: torch.device = torch.device("cpu"),
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        best_threshold: float = 0.99,
        log_csv: bool = True,
        lr_patience: int = 5,
        lr_factor: float = 0.2,
    ) -> None:
        r"""
        Initialize the base trainer.

        Parameters
        ----------
        network : torch.nn.Module
            Model to train.
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        store_path : str
            Directory to save checkpoints and logs.
        device : torch.device, optional
            Device for model and tensors, by default CPU.
        n_epochs : int, optional
            Number of training epochs, by default 10.
        learning_rate : float, optional
            Optimizer learning rate, by default 1e-3.
        weight_decay : float, optional
            Weight decay for the optimizer, by default 1e-4.
        best_threshold : float, optional
            Relative improvement threshold to save a new checkpoint.
        log_csv : bool, optional
            Whether to log metrics to CSV, by default True.

        Notes
        -----
        Configures the optimizer, checkpoint path, and CSV logger.
        """
        self._rows_to_log: List[Dict] = []

        self.network = network
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.store_path = store_path
        self.best_model_path = os.path.join(store_path, "best_model.ckpt")
        self.best_threshold = best_threshold

        self.csv_logger = CSVLogger(os.path.join(store_path, "train_log.csv"))
        self.log_csv = log_csv

        self.device = device
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.lat_weights = torch.as_tensor(
            lat_weights, device=self.device, dtype=torch.float32
        ).reshape(-1, 1)

        self._setup_optimizer()

    @torch.inference_mode()
    def __call__(self, **batch) -> torch.Tensor:
        r"""
        Perform a forward pass in inference mode and clamp outputs into valid
        range [0, 1].

        Parameters
        ----------
        batch : dict
            Keyword arguments matching the model signature.

        Returns
        -------
        torch.Tensor
            Model output tensor.

        Notes
        -----
        Runs the model with torch.inference_mode() enabled.
        """
        prediction = self.network(**batch)
        return prediction.clamp(0.0, 1.0)

    def _setup_optimizer(self) -> None:
        r"""
        Instantiates AdamW optimizer and ReduceLROnPlateau scheduler for the
        model parameters with the configured learning rate and weight decay.
        """
        self._optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        print("[DEBUG] ReduceLROnPlateau config:")
        print(f"        patience = {self.lr_patience}")
        print(f"        factor   = {self.lr_factor}")

        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=self.lr_patience,
            factor=self.lr_factor,
        )

    def _move_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        r"""
        Move a batch of tensors to the configured device.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch containing model inputs and targets.

        Returns
        -------
        Dict[str, torch.Tensor]
            Batch with tensors moved to the device.
        """
        return {k: v.to(self.device) for k, v in batch.items()}

    def _check_save_checkpoint(self, val_loss: float) -> None:
        r"""
        Save a checkpoint when validation improves.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Notes
        -----
        Saves the model when val_loss is lower than the previous best
        scaled by best_threshold.
        """
        if val_loss < self._best_loss * self.best_threshold:
            main_logger.debug(
                f"New best validation loss: {val_loss:.4f} "
                f"(previous: {self._best_loss:.4f}). Saving checkpoint."
            )
            self._best_loss = val_loss
            torch.save(self.network.state_dict(), self.best_model_path)

    def _load_best_checkpoint(self) -> None:
        r"""
        Load the best checkpoint from disk, placing onto the configured device.
        """
        main_logger.debug(f"Loading checkpoint from {self.best_model_path}.")
        self.network.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

    def _train_epoch(self) -> float:
        r"""
        Execute one training epoch: iterate over training batches, perform
        updates, and log batch losses.

        Returns
        -------
        float
            Training loss averaged over the full epoch.
        """
        _n_samples: int = 0
        _acc_loss: float = 0.0
        self.network.train()
        train_pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in train_pbar:
            batch = self._move_to_device(batch)

            self._optimizer.zero_grad()
            output_dict = self.estimate_loss(batch)
            output_dict["loss"].backward()
            self._optimizer.step()

            curr_loss = output_dict["loss"].item()
            train_pbar.set_postfix(loss=curr_loss)
            self.log({"train/loss": curr_loss}, flush=False)
            curr_samples = batch["input_level"].shape[0]
            _n_samples += curr_samples
            _acc_loss += curr_loss * curr_samples
        return _acc_loss / _n_samples

    def _val_epoch(self) -> Tuple[float, Dict[str, float]]:
        r"""
        Execute one validation epoch, aggregating the validation loss and
        auxiliary metrics over the validation set.

        Returns
        -------
        float
            Validation loss averaged over the full validation set.
        Dict[str, float]
            Auxiliary metrics averaged over the full validation set.
        """
        _n_samples: int = 0
        _losses_list: List[Dict[str, float]] = []
        self.network.eval()
        val_pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for batch in val_pbar:
            batch = self._move_to_device(batch)

            with torch.no_grad():
                output_dict = self.estimate_loss(batch)
            loss_aux = self.estimate_auxiliary_loss(batch, output_dict)

            curr_samples = batch["input_level"].shape[0]
            _n_samples += batch["input_level"].shape[0]
            curr_loss_dict = {k: v.item() * curr_samples for k, v in loss_aux.items()}
            curr_loss = output_dict["loss"].item()
            curr_loss_dict["loss"] = curr_loss * curr_samples
            _losses_list.append(curr_loss_dict)
            val_pbar.set_postfix(loss=curr_loss)

        val_loss = sum(_loss["loss"] for _loss in _losses_list) / _n_samples
        aux_losses = {
            k: sum(_loss[k] for _loss in _losses_list) / _n_samples
            for k in _losses_list[0]
            if k != "loss"
        }
        return val_loss, aux_losses

    def log(self, log_dict: Dict, flush: bool = False) -> None:
        r"""
        Log metrics and optionally flush them to disk.

        Parameters
        ----------
        log_dict : Dict
            Metric values to record.
        flush : bool, optional
            Whether to flush the CSV logger immediately, by default False.
        """
        if self.log_csv:
            self.csv_logger.log_row(log_dict)
            if flush:
                self.csv_logger.flush()
        if wandb.run is not None:
            wandb.log(log_dict)

    def train(self) -> torch.nn.Module:
        r"""
        Train the model for the configured number of epochs.

        Performs training and validation, log metrics, and checkpoint the best
        model. The best model is loaded at the end of training and returned.

        Returns
        -------
        torch.nn.Module
            The model loaded with the best checkpoint weights.
        """
        epoch_pbar = tqdm(range(1, self.n_epochs + 1), desc="Epochs")
        for idx_epoch in epoch_pbar:
            train_loss = self._train_epoch()
            val_loss, aux_losses = self._val_epoch()
            epoch_pbar.set_postfix(
                train_loss=train_loss,
                val_loss=val_loss,
            )
            self._check_save_checkpoint(val_loss)
            self._scheduler.step(val_loss)
            self.log(
                {
                    "epoch": idx_epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "lr": self._optimizer.param_groups[0]["lr"],
                    **{f"val/{k}": v for k, v in aux_losses.items()},
                },
                flush=True,
            )
        if os.path.exists(self.best_model_path):
            self._load_best_checkpoint()
        else:
            main_logger.warning(
                "No checkpoint was saved during training. "
                "Returning model with final weights."
            )
        return self.network

    def validate(self) -> Tuple[float, Dict[str, float]]:
        r"""
        Validate the model without updating parameters.

        Permorms a validation epoch and returns the averaged validation loss
        and auxiliary metrics. The output is not logged or checkpointed.

        Returns
        -------
        Tuple[float, Dict[str, float]]
            Validation loss and auxiliary losses averaged per sample.
        """
        return self._val_epoch()

    @abc.abstractmethod
    def estimate_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        r"""
        Estimate loss values for a batch.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of inputs and targets.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing loss and auxiliary outputs.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Notes
        -----
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "estimate_loss method must be implemented in subclass."
        )

    def estimate_auxiliary_loss(
        self, batch: Dict[str, torch.Tensor], outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
        Compute auxiliary loss terms from model outputs.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch of input tensors.
        outputs : Dict[str, Any]
            Output dictionary from estimate_loss().

        Returns
        -------
        Dict[str, Any]
            Auxiliary loss components.

        Notes
        -----
        Default implementation returns an empty dictionary.
        """
        return {}
