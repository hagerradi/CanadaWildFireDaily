"""
Comet ML experiment logger.
"""
from __future__ import annotations

import os
from pathlib import Path

from comet_ml import Experiment


class CometLogger:
    """Thin wrapper around :class:`comet_ml.Experiment` for metric and image logging.

    Args:
        project_name: Comet project to log into.
        workspace: Comet workspace (username or organisation).
        experiment_name: Human-readable name for this run.
        experiment_tags: Optional list of string tags applied to the experiment.
        api_key: Comet API key.  Falls back to the ``COMET_API_KEY`` environment
            variable when *None*.

    Raises:
        ValueError: If no API key can be resolved.
    """

    def __init__(
        self,
        project_name: str,
        workspace: str,
        experiment_name: str,
        experiment_tags: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("COMET_API_KEY")
        if not self.api_key:
            raise ValueError("COMET_API_KEY env. variable not set or no API key provided.")

        self.experiment = Experiment(
            api_key=self.api_key,
            project_name=project_name,
            workspace=workspace,
            log_code=False,
            log_graph=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_histogram_tensorboard_logging=False,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
            auto_output_logging="simple",
            auto_log_co2=False,
            log_env_details=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_network=False,
            log_env_host=False,
            log_git_metadata=False,
            log_git_patch=False,
        )

        self.experiment.set_name(experiment_name)

        if experiment_tags:
            self.experiment.add_tags(experiment_tags)

    def log_metrics(self, metrics: dict, step: int | None = None, epoch: int | None = None) -> None:
        """Log a dictionary of scalar metrics.

        Args:
            metrics: Mapping of metric name → scalar value.
            step: Global step counter.
            epoch: Current epoch index.
        """
        if epoch is not None:
            self.experiment.set_epoch(epoch)
        self.experiment.log_metrics(metrics, step=step, epoch=epoch)

    def log_params(self, params: dict) -> None:
        """Log a dictionary of hyper-parameters.

        Args:
            params: Mapping of parameter name → value.
        """
        self.experiment.log_parameters(params)

    def log_image(
        self,
        image_path: str | os.PathLike[str],
        name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log an image file to Comet.

        Args:
            image_path: Path to the image file on disk.
            name: Optional display name for the image in Comet.
            step: Global step counter.
            epoch: Current epoch index.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        if epoch is not None:
            self.experiment.set_epoch(epoch)

        self.experiment.log_image(
            image_path,
            name=name,
            step=step,
        )
