from models.linear_crypto import (
    LinearCrypto,
    LinearCryptoLite,
    LinearCryptoFeatureOnly,
)
from .base_module import BaseModule


class LinearCryptoModule(BaseModule):
    """
    PyTorch Lightning Module wrapper for LinearCrypto.

    LinearCrypto combines:
    1. RevIN (Reversible Instance Normalization) - handles distribution shift
    2. Dual Linear Attention - O(n) attention on both temporal and feature dimensions
    3. Individual mode - separate linear layer per feature

    Architecture:
        RevIN(norm) -> Dual Linear Attention -> Linear -> RevIN(denorm) -> Projector
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int = 1,
        individual: bool = True,
        revin: bool = True,
        revin_affine: bool = True,
        num_heads_temporal: int = 1,
        num_heads_feature: int = 1,
        feature_map: str = 'elu',
        attention_dropout: float = 0.0,
        lr: float = 0.001,
        lr_step_size: int = 50,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        logger_type: str = None,
        window_size: int = 14,
        y_key: str = 'Close',
        optimizer: str = 'adam',
        mode: str = 'default',
        loss: str = 'mse',
        **kwargs
    ):
        """
        Args:
            num_features: Number of input features
            seq_len: Input sequence length (should match window_size)
            pred_len: Prediction length (default 1)
            individual: If True, use separate linear layer for each feature
            revin: If True, apply RevIN normalization
            revin_affine: If True, learn affine parameters in RevIN
            num_heads_temporal: Number of attention heads for temporal dimension
            num_heads_feature: Number of attention heads for feature dimension
            feature_map: Feature map type ('elu', 'relu', 'identity')
            attention_dropout: Dropout rate in attention layers
            lr: Learning rate
            lr_step_size: Step size for learning rate scheduler
            lr_gamma: Gamma for learning rate scheduler
            weight_decay: Weight decay for optimizer
            logger_type: Type of logger
            window_size: Window size for data loading
            y_key: Target variable key
            optimizer: Optimizer type
            mode: Forward mode
            loss: Loss function
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

        assert seq_len == window_size, \
            f"seq_len ({seq_len}) must match window_size ({window_size})"

        self.model = LinearCrypto(
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            revin=revin,
            revin_affine=revin_affine,
            num_heads_temporal=num_heads_temporal,
            num_heads_feature=num_heads_feature,
            feature_map=feature_map,
            attention_dropout=attention_dropout,
        )

        # Log parameter count
        param_count = self.model.count_parameters()
        print(f"LinearCrypto initialized with {param_count} parameters")


class LinearCryptoLiteModule(BaseModule):
    """
    PyTorch Lightning Module wrapper for LinearCryptoLite.

    Lite version: Only temporal Linear Attention (no feature attention).
    Lighter weight while still capturing temporal dependencies.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int = 1,
        revin_affine: bool = True,
        num_heads: int = 1,
        feature_map: str = 'elu',
        lr: float = 0.001,
        lr_step_size: int = 50,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        logger_type: str = None,
        window_size: int = 14,
        y_key: str = 'Close',
        optimizer: str = 'adam',
        mode: str = 'default',
        loss: str = 'mse',
        **kwargs
    ):
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

        assert seq_len == window_size, \
            f"seq_len ({seq_len}) must match window_size ({window_size})"

        self.model = LinearCryptoLite(
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
            revin_affine=revin_affine,
            num_heads=num_heads,
            feature_map=feature_map,
        )

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LinearCryptoLite initialized with {param_count} parameters")


class LinearCryptoFeatureOnlyModule(BaseModule):
    """
    PyTorch Lightning Module wrapper for LinearCryptoFeatureOnly.

    Feature-Only version: Only applies attention across features (OHLCV dimensions).
    Learns dynamic relationships between features at each time step.

    Architecture:
        RevIN(norm) -> Feature Linear Attention -> Linear -> RevIN(denorm) -> Projector
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int = 1,
        individual: bool = True,
        revin: bool = True,
        revin_affine: bool = True,
        num_heads: int = 1,
        feature_map: str = 'elu',
        attention_dropout: float = 0.0,
        lr: float = 0.001,
        lr_step_size: int = 50,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        logger_type: str = None,
        window_size: int = 14,
        y_key: str = 'Close',
        optimizer: str = 'adam',
        mode: str = 'default',
        loss: str = 'mse',
        **kwargs
    ):
        """
        Args:
            num_features: Number of input features
            seq_len: Input sequence length (should match window_size)
            pred_len: Prediction length (default 1)
            individual: If True, use separate linear layer for each feature
            revin: If True, apply RevIN normalization
            revin_affine: If True, learn affine parameters in RevIN
            num_heads: Number of attention heads for feature attention
            feature_map: Feature map type ('elu', 'relu', 'identity')
            attention_dropout: Dropout rate in attention layers
            lr: Learning rate
            lr_step_size: Step size for learning rate scheduler
            lr_gamma: Gamma for learning rate scheduler
            weight_decay: Weight decay for optimizer
            logger_type: Type of logger
            window_size: Window size for data loading
            y_key: Target variable key
            optimizer: Optimizer type
            mode: Forward mode
            loss: Loss function
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

        assert seq_len == window_size, \
            f"seq_len ({seq_len}) must match window_size ({window_size})"

        self.model = LinearCryptoFeatureOnly(
            num_features=num_features,
            seq_len=seq_len,
            pred_len=pred_len,
            individual=individual,
            revin=revin,
            revin_affine=revin_affine,
            num_heads=num_heads,
            feature_map=feature_map,
            attention_dropout=attention_dropout,
        )

        # Log parameter count
        param_count = self.model.count_parameters()
        print(f"LinearCryptoFeatureOnly initialized with {param_count} parameters")
