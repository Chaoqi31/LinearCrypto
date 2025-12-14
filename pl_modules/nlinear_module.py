from models.nlinear import NLinear
from .base_module import BaseModule


class NLinearModule(BaseModule):
    """
    PyTorch Lightning Module wrapper for NLinear.

    NLinear is a simple yet effective model for time series forecasting
    that uses last-value normalization to handle distribution shift.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int = 1,
        lr: float = 0.0002,
        lr_step_size: int = 50,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        logger_type: str = None,
        window_size: int = 14,
        y_key: str = 'Close',
        optimizer: str = 'adam',
        mode: str = 'default',
        loss: str = 'rmse',
        **kwargs
    ):
        """
        Args:
            num_features: Number of input features
            seq_len: Input sequence length (should match window_size)
            pred_len: Prediction length (default 1)
            lr: Learning rate
            lr_step_size: Step size for learning rate scheduler
            lr_gamma: Gamma for learning rate scheduler
            weight_decay: Weight decay for optimizer
            logger_type: Type of logger (tensorboard, wandb, etc.)
            window_size: Window size for data loading
            y_key: Target variable key
            optimizer: Optimizer type (adam, sgd)
            mode: Forward mode (default, diff)
            loss: Loss function (rmse, mse, mae, mape)
        """
        super().__init__(
            lr=lr,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            logger_type=logger_type,
            y_key=y_key,
            optimizer=optimizer,
            mode=mode,
            window_size=window_size,
            loss=loss,
        )

        # Ensure seq_len matches window_size
        assert seq_len == window_size, \
            f"seq_len ({seq_len}) must match window_size ({window_size})"

        self.model = NLinear(
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
        )
