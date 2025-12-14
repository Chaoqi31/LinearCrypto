import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.

    Paper: "Reversible Instance Normalization for Accurate Time-Series Forecasting
            against Distribution Shift" (ICLR 2022)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

        self._mean = None
        self._std = None

    def forward(self, x, mode: str = 'norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x

        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self._std + self._mean
            return x

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'norm' or 'denorm'.")


class LinearAttention(nn.Module):
    """
    Linear Attention with O(n) complexity.

    Paper: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
           (Katharopoulos et al., ICML 2020)

    Key idea: Replace softmax(QK^T)V with φ(Q)(φ(K)^T V)
    where φ is a feature map (ELU + 1 to ensure non-negativity).

    Complexity: O(n * d^2) instead of O(n^2 * d)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = True,
        feature_map: str = 'elu',
    ):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q/K/V projections
            feature_map: Type of feature map ('elu', 'relu', 'identity')
        """
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.feature_map = feature_map
        self.eps = 1e-6

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def _feature_map(self, x):
        """Apply feature map φ to ensure non-negativity."""
        if self.feature_map == 'elu':
            return F.elu(x) + 1
        elif self.feature_map == 'relu':
            return F.relu(x) + self.eps
        elif self.feature_map == 'identity':
            return x
        else:
            raise ValueError(f"Unknown feature map: {self.feature_map}")

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, seq_len, dim)

        Returns:
            Output tensor of shape (B, seq_len, dim)
        """
        B, N, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (B, N, D)
        K = self.k_proj(x)  # (B, N, D)
        V = self.v_proj(x)  # (B, N, D)

        # Reshape for multi-head attention
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)

        # Apply feature map
        Q = self._feature_map(Q)  # (B, H, N, d)
        K = self._feature_map(K)  # (B, H, N, d)

        # Linear attention: φ(Q) @ (φ(K)^T @ V)
        # Instead of: softmax(Q @ K^T) @ V  [O(n^2)]
        # We compute: φ(Q) @ (φ(K)^T @ V)   [O(n * d^2)]

        # K^T @ V: (B, H, d, N) @ (B, H, N, d) -> (B, H, d, d)
        KV = torch.einsum('bhnd,bhne->bhde', K, V)

        # Q @ KV: (B, H, N, d) @ (B, H, d, d) -> (B, H, N, d)
        output = torch.einsum('bhnd,bhde->bhne', Q, KV)

        # Normalization: divide by sum of K for numerical stability
        # Z = sum(φ(K), dim=seq)
        Z = K.sum(dim=2)  # (B, H, d)
        # Q @ Z: (B, H, N, d) @ (B, H, d) -> (B, H, N)
        normalizer = torch.einsum('bhnd,bhd->bhn', Q, Z).unsqueeze(-1) + self.eps  # (B, H, N, 1)

        output = output / normalizer  # (B, H, N, d)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)

        # Output projection
        output = self.out_proj(output)

        return output


class DualLinearAttention(nn.Module):
    """
    Dual Linear Attention: applies attention on both temporal and feature dimensions.

    Architecture:
        Input: (B, seq_len, num_features)
            ↓
        Temporal Linear Attention (across time steps)
            ↓
        Feature Linear Attention (across features)
            ↓
        Output: (B, seq_len, num_features)
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_heads_temporal: int = 1,
        num_heads_feature: int = 1,
        feature_map: str = 'elu',
        dropout: float = 0.0,
    ):
        """
        Args:
            seq_len: Sequence length (temporal dimension)
            num_features: Number of features
            num_heads_temporal: Number of heads for temporal attention
            num_heads_feature: Number of heads for feature attention
            feature_map: Feature map type for linear attention
            dropout: Dropout rate
        """
        super(DualLinearAttention, self).__init__()

        self.seq_len = seq_len
        self.num_features = num_features

        # Temporal attention: attend across time steps
        # Input shape: (B, seq_len, num_features)
        self.temporal_attention = LinearAttention(
            dim=num_features,
            num_heads=num_heads_temporal,
            feature_map=feature_map,
        )

        # Feature attention: attend across features
        # We need to transpose to (B, num_features, seq_len) first
        self.feature_attention = LinearAttention(
            dim=seq_len,
            num_heads=num_heads_feature,
            feature_map=feature_map,
        )

        # Layer norms for residual connections
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(seq_len)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, seq_len, num_features)

        Returns:
            Output tensor of shape (B, seq_len, num_features)
        """
        # Temporal attention with residual connection
        # x: (B, seq_len, num_features)
        residual = x
        x = self.norm1(x)
        x = self.temporal_attention(x)
        x = self.dropout(x)
        x = x + residual

        # Feature attention with residual connection
        # Transpose to (B, num_features, seq_len)
        x = x.transpose(1, 2)  # (B, num_features, seq_len)
        residual = x
        x = self.norm2(x)
        x = self.feature_attention(x)
        x = self.dropout(x)
        x = x + residual

        # Transpose back to (B, seq_len, num_features)
        x = x.transpose(1, 2)

        return x


class LinearCrypto(nn.Module):
    """
    LinearCrypto: Linear Attention for Cryptocurrency Price Prediction.

    Architecture:
        Input: (B, num_features, seq_len)
            ↓ permute
        (B, seq_len, num_features)
            ↓
        RevIN (normalize)
            ↓
        Dual Linear Attention (temporal + feature)
            ↓
        Linear Projection (seq_len -> pred_len)
            ↓
        RevIN (denormalize)
            ↓
        Projector (num_features -> 1)
            ↓
        Output: (B, pred_len)

    Features:
    - O(n) attention complexity
    - Captures both temporal and cross-feature dependencies
    - RevIN handles distribution shift
    - Individual mode for feature-specific temporal patterns
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
    ):
        """
        Args:
            num_features: Number of input features (e.g., 6 for OHLCV + Timestamp)
            seq_len: Input sequence length (e.g., 14)
            pred_len: Prediction length (default 1)
            individual: If True, use separate linear layer for each feature
            revin: If True, apply RevIN normalization
            revin_affine: If True, learn affine parameters in RevIN
            num_heads_temporal: Number of attention heads for temporal dimension
            num_heads_feature: Number of attention heads for feature dimension
            feature_map: Feature map type ('elu', 'relu', 'identity')
            attention_dropout: Dropout rate in attention
        """
        super(LinearCrypto, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.use_revin = revin

        # RevIN layer
        if self.use_revin:
            self.revin = RevIN(num_features, affine=revin_affine)

        # Dual Linear Attention
        self.attention = DualLinearAttention(
            seq_len=seq_len,
            num_features=num_features,
            num_heads_temporal=num_heads_temporal,
            num_heads_feature=num_heads_feature,
            feature_map=feature_map,
            dropout=attention_dropout,
        )

        # Linear layers for temporal projection
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.num_features)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Projector: maps num_features -> 1 for final scalar output
        self.projector = nn.Linear(self.num_features, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        if self.individual:
            for linear in self.Linear:
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
                nn.init.zeros_(linear.bias)
        else:
            nn.init.xavier_uniform_(self.Linear.weight, gain=0.1)
            nn.init.zeros_(self.Linear.bias)

        nn.init.xavier_uniform_(self.projector.weight, gain=0.1)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        """
        Forward pass with RevIN, Dual Linear Attention, and Linear projection.

        Args:
            x: Input tensor of shape (B, num_features, seq_len)

        Returns:
            Output tensor of shape (B, pred_len)
        """
        # x: (B, num_features, seq_len) -> (B, seq_len, num_features)
        x = x.permute(0, 2, 1)

        # Apply RevIN normalization (方案A: before attention)
        if self.use_revin:
            x = self.revin(x, mode='norm')

        # Apply Dual Linear Attention
        x = self.attention(x)

        # Linear projection on temporal dimension
        if self.individual:
            output = torch.zeros(
                x.size(0), self.pred_len, self.num_features,
                dtype=x.dtype, device=x.device
            )
            for i in range(self.num_features):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply RevIN denormalization
        if self.use_revin:
            x = self.revin(x, mode='denorm')

        # x: (B, pred_len, num_features) -> (B, pred_len, 1) -> (B, pred_len)
        output = self.projector(x)
        output = output.squeeze(-1)

        return output

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearCryptoLite(nn.Module):
    """
    LinearCryptoLite: Lightweight version with only temporal Linear Attention.
    Even lighter weight while still capturing temporal dependencies.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int = 1,
        revin_affine: bool = True,
        num_heads: int = 1,
        feature_map: str = 'elu',
    ):
        super(LinearCryptoLite, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len

        # RevIN layer
        self.revin = RevIN(num_features, affine=revin_affine)

        # Temporal Linear Attention only
        self.attention = LinearAttention(
            dim=num_features,
            num_heads=num_heads,
            feature_map=feature_map,
        )
        self.norm = nn.LayerNorm(num_features)

        # Shared linear layer
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Projector
        self.projector = nn.Linear(self.num_features, 1, bias=True)

    def forward(self, x):
        # x: (B, num_features, seq_len) -> (B, seq_len, num_features)
        x = x.permute(0, 2, 1)

        # RevIN normalize
        x = self.revin(x, mode='norm')

        # Temporal attention with residual
        residual = x
        x = self.norm(x)
        x = self.attention(x)
        x = x + residual

        # Linear projection (shared)
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # RevIN denormalize
        x = self.revin(x, mode='denorm')

        # Project to scalar
        output = self.projector(x).squeeze(-1)

        return output

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearCryptoFeatureOnly(nn.Module):
    """
    LinearCryptoFeatureOnly: Feature-Only Linear Attention version.

    Only applies attention across features (OHLCV dimensions), not across time steps.
    This learns dynamic relationships between features at each time step.

    Architecture:
        Input: (B, num_features, seq_len)
            ↓ permute
        (B, seq_len, num_features)
            ↓
        RevIN (normalize)
            ↓
        Feature Linear Attention (across features only)
            ↓
        Linear Projection (seq_len -> pred_len)
            ↓
        RevIN (denormalize)
            ↓
        Projector (num_features -> 1)
            ↓
        Output: (B, pred_len)

    Use case:
    - When feature interactions (Open↔High↔Low↔Close↔Volume) are more important
      than temporal patterns
    - Lighter than dual attention version
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
    ):
        """
        Args:
            num_features: Number of input features (e.g., 6 for OHLCV + Timestamp)
            seq_len: Input sequence length (e.g., 14)
            pred_len: Prediction length (default 1)
            individual: If True, use separate linear layer for each feature
            revin: If True, apply RevIN normalization
            revin_affine: If True, learn affine parameters in RevIN
            num_heads: Number of attention heads for feature attention
            feature_map: Feature map type ('elu', 'relu', 'identity')
            attention_dropout: Dropout rate in attention
        """
        super(LinearCryptoFeatureOnly, self).__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.use_revin = revin

        # RevIN layer
        if self.use_revin:
            self.revin = RevIN(num_features, affine=revin_affine)

        # Feature Linear Attention only
        # Input: (B, num_features, seq_len) - attend across features
        self.attention = LinearAttention(
            dim=seq_len,
            num_heads=num_heads,
            feature_map=feature_map,
        )
        self.norm = nn.LayerNorm(seq_len)
        self.dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()

        # Linear layers for temporal projection
        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len)
                for _ in range(self.num_features)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Projector: maps num_features -> 1 for final scalar output
        self.projector = nn.Linear(self.num_features, 1, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        if self.individual:
            for linear in self.Linear:
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
                nn.init.zeros_(linear.bias)
        else:
            nn.init.xavier_uniform_(self.Linear.weight, gain=0.1)
            nn.init.zeros_(self.Linear.bias)

        nn.init.xavier_uniform_(self.projector.weight, gain=0.1)
        nn.init.zeros_(self.projector.bias)

    def forward(self, x):
        """
        Forward pass with RevIN, Feature Linear Attention, and Linear projection.

        Args:
            x: Input tensor of shape (B, num_features, seq_len)

        Returns:
            Output tensor of shape (B, pred_len)
        """
        # x: (B, num_features, seq_len) -> (B, seq_len, num_features)
        x = x.permute(0, 2, 1)

        # Apply RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')

        # Feature attention with residual connection
        # Transpose to (B, num_features, seq_len) for feature attention
        x = x.transpose(1, 2)  # (B, num_features, seq_len)
        residual = x
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + residual
        # Transpose back to (B, seq_len, num_features)
        x = x.transpose(1, 2)

        # Linear projection on temporal dimension
        if self.individual:
            output = torch.zeros(
                x.size(0), self.pred_len, self.num_features,
                dtype=x.dtype, device=x.device
            )
            for i in range(self.num_features):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply RevIN denormalization
        if self.use_revin:
            x = self.revin(x, mode='denorm')

        # x: (B, pred_len, num_features) -> (B, pred_len, 1) -> (B, pred_len)
        output = self.projector(x)
        output = output.squeeze(-1)

        return output

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
