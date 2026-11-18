# CLF: logistica, arboles, MLP, transformers y LSTM o GRU

import torch
import torch.nn as nn
from src.models.regression import Layer

class MLPClassifier(nn.Module):
    def __init__(self, layers):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=8,  # Variables por visita médica (ej. 8 en T0 y T1)
        hidden_size=32,  # Memoria de la LSTM
        num_lstm_layers=1,  # Capas LSTM apiladas
        lstm_dropout=0.0,  # Dropout entre capas LSTM (si num_layers > 1)
        layer_norm=True,  # Normalización de la salida de la LSTM
        head_layers=None,  # Lista de objetos Layer() para la cabecera
        out_size=3,  # Número de clases (ej. 3 diagnósticos: CN, MCI, AD)
    ):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )

        self.layer_norm = (nn.LayerNorm(hidden_size) if layer_norm else nn.Identity())

        if head_layers is None or len(head_layers) == 0:
            self.head = nn.LazyLinear(out_size)
        else:
            self.head = nn.Sequential(*head_layers, nn.LazyLinear(out_size))

    def forward(self, x):
        out, _ = self.lstm(x) # x = [batch_size, seq_len=2, input_size=8]
        last_out = self.layer_norm(out[:, -1, :])
        return self.head(last_out)

class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size=8, # Variables por paso temporal (ej. 8 en T0 y T1)
        hidden_size=32, # Tamaño de la memoria interna de la GRU
        num_gru_layers=1, # Número de capas GRU apiladas
        gru_dropout=0.0, # Dropout entre capas GRU (si num_layers > 1)
        layer_norm=True, # Normalización de la salida de la GRU
        head_layers=None, # Lista de objetos Layer() para la cabecera
        out_size=3, # Número de clases (ej. 3 diagnósticos: CN, MCI, AD)
    ):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=gru_dropout if num_gru_layers > 1 else 0.0,
        )

        self.layer_norm = (nn.LayerNorm(hidden_size) if layer_norm else nn.Identity())

        if head_layers is None or len(head_layers) == 0:
            self.head = nn.LazyLinear(out_size)
        else:
            self.head = nn.Sequential(*head_layers, nn.LazyLinear(out_size))

    def forward(self, x):
        out, _ = self.gru(x) # x = [batch_size, seq_len=2, input_size=8] -> Devuelve logits
        last_out = self.layer_norm(out[:, -1, :])
        return self.head(last_out)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size=8, # Variables por paso temporal (ej. 8 en T0 y T1)
        d_model=32, # Tamaño de proyección interna de las características
        nhead=4, # Número de cabezas de atención (d_model divisible entre nhead)
        num_transformer_layers=1, # Número de bloques Transformer apilados
        dim_feedforward=64, # Capa densa interna dentro de cada bloque Transformer
        transformer_dropout=0.0, # Dropout interno del Transformer
        layer_norm=True, # Normalización al salir del Transformer
        head_layers=None, # Capas densas intermedias para la cabecera (usando Layer)
        out_size=3, # Número de clases (ej. 3 diagnósticos: CN, MCI, AD)
    ):
        super(TransformerClassifier, self).__init__()
        assert d_model % nhead == 0

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else nn.Identity()

        if head_layers is None or len(head_layers) == 0:
            self.head = nn.LazyLinear(out_size)
        else:
            self.head = nn.Sequential(*head_layers, nn.LazyLinear(out_size))

    def forward(self, x):
        out = self.input_projection(x) + self.pos_embedding[:, : x.size(1), :]
        out = self.transformer_encoder(out)
        last_out = self.layer_norm(out[:, -1, :])
        return self.head(last_out)