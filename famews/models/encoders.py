# ========================================
#
# Time-Series Encoder Modules
# Ref: https://github.com/ratschlab/HIRID-ICU-Benchmark/blob/master/icu_benchmarks/models/encoders.py
#
# ========================================
import logging

import gin
import lightgbm
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import torch
import torch.nn as nn

gin.config.external_configurable(lightgbm.sklearn.LGBMClassifier, module="lightgbm.sklearn")
gin.config.external_configurable(
    sklearn.linear_model.LogisticRegression, module="sklearn.linear_model"
)
gin.config.external_configurable(sklearn.ensemble.RandomForestClassifier, module="sklearn.ensemble")
gin.config.external_configurable(sklearn.ensemble.ExtraTreesClassifier, module="sklearn.ensemble")
gin.config.external_configurable(
    sklearn.ensemble.GradientBoostingClassifier, module="sklearn.ensemble"
)

from famews.models.utility_layers import PositionalEncoding


@gin.configurable("LSTMEncoder")
class LSTMEncoder(nn.Module):
    """Simple LSTM based sequence encoder"""

    def __init__(
        self, input_dim: int, hidden_dim: int = None, num_layers: int = 1, dropout: float = 0.0
    ):
        """
        Constructor for `LSTMEncoder`

        Parameter
        ---------
        input_dim: int
            input vector dimension
        hidden_dim: int
            model hidden dimension
        num_layers: int
            number of stacked LSTM layers
        dropout: float
            dropout fraction applied in the hidden dimension
        """
        super().__init__()

        self.hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_dim, self.hidden_dim, num_layers, batch_first=True, dropout=dropout
        )

    def init_hidden(
        self, x: torch.Tensor, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the sequence inputs to the RNN module with 0-vectors

        Parameter
        ---------
        x: torch.Tensor
            input batch to get dimensions
        device: torch.device
            device to initialize data on
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        return (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0, c0 = self.init_hidden(x, device=x.device)
        x_seq, _ = self.rnn(x, (h0, c0))
        return x_seq


@gin.configurable("GRUEncoder")
class GRUEncoder(nn.Module):
    """Simple GRU based sequence encoder"""

    def __init__(
        self, input_dim: int, hidden_dim: int = None, num_layers: int = 1, dropout: float = 0.0
    ):
        """
        Constructor for `GRUEncoder`
        Parameter
        ---------
        input_dim: int
            input vector dimension
        hidden_dim: int
            model hidden dimension
        num_layers: int
            number of stacked GRU layers
        dropout: float
            dropout fraction applied in the hidden dimension
        """
        super().__init__()
        self.hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, self.hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def init_hidden(
        self, x: torch.Tensor, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        return h0

    def forward(self, x: torch.Tensor):
        h0 = self.init_hidden(x, device=x.device)
        x_seq, _ = self.rnn(x, h0)
        return x_seq


@gin.configurable("TransformerSequenceEncoder")
class TransformerSequenceEncoder(nn.Module):
    """Transformer based causal sequence encoder"""

    def __init__(
        self,
        input_dim: int,
        is_causal: bool = True,
        num_layers: int = 1,
        num_heads: int = 1,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        pos_enc: bool = True,
        pos_enc_length: int = 3000,
    ) -> None:
        super().__init__()

        self.is_causal = is_causal
        if is_causal:
            logging.info(f"[{self.__class__.__name__}] Using causal attention")

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )

        transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.transformer_encoder = transformer_encoder

        if pos_enc:
            self.pos_enc = PositionalEncoding(input_dim, pos_enc_length)
        else:
            self.pos_enc = nn.Identity()

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameter
        ---------
        x: torch.Tensor
            input sequence of shape (batch_size, seq_len, input_dim)
        mask: torch.Tensor
            mask for the input sequence of shape (batch_size, seq_len) (default: None)
        """
        attn_mask = None
        if self.is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)

        x = self.pos_enc(x)
        x = self.transformer_encoder(
            x, is_causal=self.is_causal, mask=attn_mask, src_key_padding_mask=pad_mask
        )

        return x


@gin.configurable("TransformerCLSEncoder")
class TransformerCLSEncoder(TransformerSequenceEncoder):
    """Transformer based encoder with CLS token, collapsing the sequence into a single vector"""

    def __init__(self, input_dim, **kwargs) -> None:

        # remove is_causal from kwargs
        is_causal_kwargs = kwargs.pop("is_causal", None)
        if is_causal_kwargs is not None:
            warning_msg = f"[{self.__class__.__name__}] is_causal is set to {is_causal_kwargs} and will be ignored"
            logging.warning(warning_msg)

        super().__init__(input_dim=input_dim, is_causal=False, **kwargs)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.cls_token)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameter
        ---------
        x: torch.Tensor
            input sequence of shape (batch_size, seq_len, input_dim)
        mask: torch.Tensor
            mask for the input sequence of shape (batch_size, seq_len) (default: None)
        """
        # add CLS token to the input sequence
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # pass through transformer encoder
        x = super().forward(x, pad_mask=mask)

        # return the CLS token output
        return x[:, 0, :]


@gin.configurable("EmbeddedSequenceEncoder")
class EmbeddedSequenceEncoder(nn.Module):
    """
    Sequence encoder taking in two modules:
    - time-step embedding module
    - sequence encoder module
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        step_embedding_class: nn.Module = gin.REQUIRED,
        seq_encoder_class: nn.Module = gin.REQUIRED,
    ):
        """
        Constructor for `GRUNet`
        Parameter
        ---------
        input_dim: int
            input vector dimension of the raw features at each timepoint
        hidden_dim: int
            model hidden dimension
        step_embedding_class: nn.Module
            time-step embedding module
        seq_encoder_class: nn.Module
            sequence encoder module
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.time_step_embedding = step_embedding_class(input_dim, hidden_dim)
        self.sequence_encoder = seq_encoder_class(hidden_dim)

    def forward(self, x: torch.Tensor):

        # Time-Step Embedding
        batch_dim, seq_dim, feature_dim = x.shape
        x = x.view(batch_dim * seq_dim, feature_dim)
        x_emb = self.time_step_embedding(x).view(batch_dim, seq_dim, self.hidden_dim)

        # Sequence Modeling
        x_seq = self.sequence_encoder(x_emb)

        return x_seq


@gin.configurable("SequenceModel")
class SequenceModel(nn.Module):
    """
    Simple sequence model encoding a time-series with a sequence encoder and
    predicting a label with an MLP or Linear layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        encoder: nn.Module = gin.REQUIRED,
        logit_layer: nn.Module = nn.Linear,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.encoder = encoder(input_dim, hidden_dim)
        self.logit_layer = logit_layer(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_seq = self.encoder(x)
        x_logits = self.logit_layer(x_seq)

        return x_logits
