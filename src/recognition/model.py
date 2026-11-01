"""
Kurzschrift-HTR Modellarchitekturen:
1. DekCRNN: Kompaktes Standard-CRNN (CNN + 2-Layer BiLSTM + CTC)
2. DekResNetAttentionHTR: Großes ResNet-BiLSTM-Transformer-Attention Hybridmodell
   (SE-Residual-Backbone + 3-Layer BiLSTM + Multi-Head Self-Attention Encoder + CTC)

Speicheroptimiert für NVIDIA RTX A500 (FP16 Mixed Precision kompatibel).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation2d(nn.Module):
    """Kanal-Aufmerksamkeit (Squeeze-and-Excitation) zur feinen Unterscheidung von Haar- und Druckstrichen."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.silu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock2d(nn.Module):
    """ResNet-Block mit Squeeze-and-Excitation und optionalem Stride-Downsampling."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation2d(out_channels)

        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act(out + residual)
        return out


class PositionalEncoding1D(nn.Module):
    """Sinusförmiges 1D-Positional-Encoding für die Transformer-Attention-Schichten."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Form: (Max_Len, 1, D_Model)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (Time, Batch, D_Model)"""
        seq_len = x.size(0)
        if seq_len <= self.pe.size(0):
            x = x + self.pe[:seq_len]
        return self.dropout(x)


class DekResNetAttentionHTR(nn.Module):
    """
    Großes ResNet-BiLSTM-Attention Hybridmodell (TrOCR-Stil Vision-Sequence-Architecture).

    Vorteile gegenüber einfachem CRNN:
    - Tiefe Residual-Blöcke mit Squeeze-and-Excitation (erkennt subtile Schattierungen / Stufen)
    - Unterstützt höhere Zeilenauflösungen (Height=64 oder Height=96 via adaptivem Vertikal-Pooling)
    - 3-schichtiges BiLSTM (Hidden=384 -> 768 Feature-Kanäle) für lokalen Strichfluss
    - 2-schichtiger Multi-Head Self-Attention Transformer-Encoder für globalen Satzkontext
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        d_model: int = 512,
        rnn_hidden_size: int = 384,
        rnn_layers: int = 3,
        attn_heads: int = 8,
        attn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.arch_name = "resnet_attention"

        # 1. Stem + ResNet-SE Backbone (Breite wird exakt um Faktor 4 reduziert: W -> W/2 -> W/4)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResidualBlock2d(64, 128, stride=(2, 2)),     # H/2, W/2
            ResidualBlock2d(128, 128, stride=(1, 1)),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock2d(128, 256, stride=(2, 2)),    # H/4, W/4
            ResidualBlock2d(256, 256, stride=(1, 1)),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock2d(256, 384, stride=(2, 1)),    # H/8, W/4
            ResidualBlock2d(384, 384, stride=(1, 1)),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock2d(384, d_model, stride=(2, 1)),  # H/16, W/4
            ResidualBlock2d(d_model, d_model, stride=(1, 1)),
        )

        # Vertikale Aggregation auf Höhe 1 (funktioniert für H=64, H=96 und H=128)
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        self.feature_norm = nn.LayerNorm(d_model)

        # 2. Lokale Sequenzmodellierung: 3-Layer BiLSTM
        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=False,
        )
        self.rnn_proj = nn.Linear(rnn_hidden_size * 2, d_model)
        self.rnn_norm = nn.LayerNorm(d_model)

        # 3. Globale Kontextmodellierung: Multi-Head Self-Attention (Transformer-Encoder)
        self.pos_encoder = PositionalEncoding1D(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=attn_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=attn_layers, enable_nested_tensor=False
        )

        # 4. CTC Projektionskopf
        self.fc = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (Batch, 1, Height, Width)
        Returns:
            log_probs: Tensor (Time=Width//4, Batch, Num_Classes)
        """
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        # Vertikal auf 1 zusammenfassen: (B, D, 1, W/4) -> (B, D, W/4)
        feat = self.vertical_pool(feat).squeeze(2)

        # Umordnen auf (Time, Batch, D_Model)
        seq = feat.permute(2, 0, 1)
        seq = self.feature_norm(seq)

        # BiLSTM + Residual-Projektion
        rnn_out, _ = self.rnn(seq)
        seq = self.rnn_norm(seq + self.rnn_proj(rnn_out))

        # Transformer Self-Attention über die gesamte Zeile
        seq = self.pos_encoder(seq)
        attn_out = self.transformer(seq)

        logits = self.fc(attn_out)
        return self.log_softmax(logits)


class DekCRNN(nn.Module):
    """Kompaktes Standard-CRNN für Zeilen-Texterkennung mit CTC Loss."""

    def __init__(self, num_classes: int, in_channels: int = 1, rnn_hidden_size: int = 256):
        super().__init__()
        self.arch_name = "crnn"

        self.cnn = nn.Sequential(
            # Block 1: 64 -> 32
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 32, W/2)

            # Block 2: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, 16, W/4)

            # Block 3: Merkmalsvertiefung
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: Nur Höhe reduzieren: 16 -> 8
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 256, 8, W/4)

            # Block 5: Merkmalsvertiefung
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 6: Nur Höhe reduzieren: 8 -> 4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # (B, 512, 4, W/4)

            # Block 7: Letzte Höhenreduktion: 4 -> 1
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),  # (B, 512, 1, W/4)
        )

        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False,
        )

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        if features.size(2) != 1:
            features = self.vertical_pool(features)
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return self.log_softmax(logits)


def build_htr_model(arch: str, num_classes: int, in_channels: int = 1) -> nn.Module:
    """Erstellt ein HTR-Modell nach Architekturname ('crnn' oder 'resnet_attention')."""
    arch_key = arch.strip().lower()
    if arch_key in ("resnet_attention", "resnet", "trocr_hybrid", "large"):
        return DekResNetAttentionHTR(num_classes=num_classes, in_channels=in_channels)
    elif arch_key == "crnn":
        return DekCRNN(num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(f"Unbekannte Modellarchitektur: {arch}")
