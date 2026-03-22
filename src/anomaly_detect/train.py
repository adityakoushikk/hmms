"""Anomaly detection training entry point.

Usage
-----
python -m anomaly_detect.train \
    data.provider_month_csv=data/outputs/provider_month.csv \
    model=isolation_forest

# Autoencoder (unsupervised, train on negatives):
python -m anomaly_detect.train \
    data.provider_month_csv=data/outputs/provider_month.csv \
    data.train_on_negatives=true \
    model=autoencoder

# Logistic Regression (supervised, stratified kfold):
python -m anomaly_detect.train \
    data.provider_month_csv=data/outputs/provider_month.csv \
    data/splitter=stratified_kfold \
    model=logistic_regression
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger

from anomaly_detect.data.anomaly_datamodule import AnomalyDataModule
from anomaly_detect.models.anomaly_module import AnomalyLitModule
from anomaly_detect.utils.instantiators import instantiate_callbacks
from anomaly_detect.utils.logging_utils import log_hyperparameters
from anomaly_detect.utils.metrics import compute_lift_at_percentiles, print_lift_table

log = logging.getLogger(__name__)


# ── Sklearn path ──────────────────────────────────────────────────────────────

def _train_sklearn(model, datamodule: AnomalyDataModule, cfg: DictConfig, wandb_run) -> dict:
    eval_set = getattr(model, "evaluation_set", "all")

    log.info(f"Fitting {type(model).__name__}...")
    model.fit(datamodule.X_train_np, datamodule.y_train_np)

    if eval_set == "test":
        X_s, y_s, npis_s = datamodule.X_test_np, datamodule.y_test_np, datamodule.npis_test
    else:
        X_s, y_s, npis_s = datamodule.X_all_np, datamodule.y_all_np, datamodule.npis_all

    scores = model.anomaly_scores(X_s)
    percentiles = list(cfg.lift_percentiles)
    metrics, results_df = compute_lift_at_percentiles(scores, y_s, npis_s, percentiles)

    if hasattr(model, "extra_metrics"):
        metrics.update(model.extra_metrics(X_s, y_s))
    metrics["n_features"] = datamodule.n_features

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.log({"top_500_providers": wandb.Table(dataframe=results_df.head(500))})

    _save_outputs(results_df, datamodule, cfg)
    print_lift_table(metrics, percentiles)
    return metrics


# ── Neural path ───────────────────────────────────────────────────────────────

def _train_neural(net, datamodule: AnomalyDataModule, cfg: DictConfig) -> dict:
    lit = AnomalyLitModule(net=net, optimizer_cfg=cfg.model.optimizer)

    # WANDB logger
    wandb_logger = None
    if "logger" in cfg:
        try:
            wandb_logger = instantiate(
                cfg.logger,
                name=cfg.task_name,
                tags=list(cfg.get("tags", [])),
            )
        except Exception as e:
            log.warning(f"Could not init WANDB logger: {e}")

    callbacks = instantiate_callbacks(cfg.get("callbacks", {}))

    # Skip EarlyStopping if no val data (NoneSplitter)
    if datamodule.val_dataloader() is None:
        callbacks = [cb for cb in callbacks if not hasattr(cb, "monitor")]
        log.info("No val split — EarlyStopping disabled.")

    trainer: L.Trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger if wandb_logger else False,
        callbacks=callbacks if callbacks else None,
        default_root_dir=cfg.paths.output_dir,
    )

    if wandb_logger:
        log_hyperparameters({"cfg": cfg, "model": lit, "trainer": trainer})

    trainer.fit(lit, datamodule)

    # Score all providers
    log.info("Scoring all providers...")
    scores = lit.compute_anomaly_scores(datamodule.X_all_np)
    y_all = datamodule.y_all_np
    npis_all = datamodule.npis_all

    percentiles = list(cfg.lift_percentiles)
    metrics, results_df = compute_lift_at_percentiles(scores, y_all, npis_all, percentiles)
    metrics["n_features"] = datamodule.n_features

    if wandb_logger:
        wandb_logger.experiment.log(metrics)
        wandb_logger.experiment.log({
            "top_500_providers": wandb.Table(dataframe=results_df.head(500))
        })

    _save_outputs(results_df, datamodule, cfg)
    print_lift_table(metrics, percentiles)
    return metrics


# ── Shared output ─────────────────────────────────────────────────────────────

def _save_outputs(results_df, datamodule: AnomalyDataModule, cfg: DictConfig) -> None:
    out_dir = Path(cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "scored_providers.csv"
    results_df.to_csv(scored_path, index=False)
    log.info(f"Scored providers → {scored_path}")

    if datamodule.auroc_df is not None:
        auroc_path = out_dir / "feature_auroc.csv"
        datamodule.auroc_df.to_csv(auroc_path, index=False)
        log.info(f"Feature AUROC → {auroc_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def train(cfg: DictConfig) -> dict:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # ── DataModule ────────────────────────────────────────────────────────
    log.info("Instantiating datamodule...")
    datamodule: AnomalyDataModule = instantiate(cfg.data)
    datamodule.setup()
    log.info(f"Features: {datamodule.n_features}")

    # ── Model ─────────────────────────────────────────────────────────────
    is_neural = "net" in cfg.model

    if is_neural:
        # Patch input_dim (??? in config) with actual feature count
        with open_dict(cfg):
            cfg.model.net.input_dim = datamodule.n_features
        net = instantiate(cfg.model.net)
        return _train_neural(net, datamodule, cfg)

    else:
        model = instantiate(cfg.model)

        # Init WANDB directly for sklearn models
        wandb_run = None
        if "logger" in cfg:
            try:
                wandb_run = wandb.init(
                    project=cfg.logger.project,
                    name=cfg.task_name,
                    tags=list(cfg.get("tags", [])),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
            except Exception as e:
                log.warning(f"Could not init WANDB: {e}")

        metrics = _train_sklearn(model, datamodule, cfg, wandb_run)

        if wandb_run:
            wandb_run.finish()

        return metrics


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> Optional[float]:
    metrics = train(cfg)
    opt_metric = cfg.get("optimized_metric")
    if opt_metric and opt_metric in metrics:
        return float(metrics[opt_metric])
    return None


if __name__ == "__main__":
    main()
