"""
PyTorch Dataset und Collate-Funktion für Zeilen-HTR mit CTC-Loss.

Lädt Zeilenbilder aus dem Manifest, skaliert sie auf feste Zeilenhöhe
(unter Beibehaltung des Seitenverhältnisses) und bereitet Batches für
das Training mit variablen Breiten vor.
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.recognition.vocabulary import DekVocabulary


class DekLineDataset(Dataset):
    """Dataset für einzelne Kurzschrift-Zeilenstreifen mit optionaler On-the-fly-Augmentierung."""

    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        vocab: Optional[DekVocabulary] = None,
        target_height: int = 64,
        max_width: int = 1200,
        augment: Optional[bool] = None,
    ):
        self.manifest_path = Path(manifest_csv)
        self.split = split
        self.vocab = vocab or DekVocabulary()
        self.target_height = target_height
        self.max_width = max_width
        self.augment = (split == "train") if augment is None else augment
        self.samples: List[Dict] = []

        self._load_manifest()

    def _load_manifest(self):
        """Lädt Zeileneinträge des gewählten Splits aus der CSV-Datei."""
        with open(self.manifest_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == self.split:
                    img_p = Path(row["line_image_path"])
                    if img_p.exists():
                        self.samples.append({
                            "line_id": row["line_id"],
                            "image_path": img_p,
                            "text": row["text"]
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], str]:
        sample = self.samples[idx]
        img_path = sample["image_path"]
        text = sample["text"]

        # Bild als Graustufe laden (sehr VRAM-effizient, 1 Kanal)
        with Image.open(img_path) as img:
            gray_img = img.convert("L")

            # Skalierung auf Zielhöhe unter Beibehaltung der Proportionen
            w, h = gray_img.size
            if h <= 0:
                h = 1
            scale = self.target_height / float(h)
            # On-the-fly horizontale Dehnung/Stauchung (Tempo-Variation der Handschrift)
            if self.augment and random.random() < 0.5:
                scale *= random.uniform(0.92, 1.08)

            new_w = max(16, min(int(w * scale), self.max_width))
            resized_img = gray_img.resize((new_w, self.target_height), Image.Resampling.BILINEAR)

            # Leichte affine Neigungs-/Verschiebungs-Augmentierung
            if self.augment and random.random() < 0.4:
                angle = random.uniform(-1.2, 1.2)
                shear = random.uniform(-4.0, 4.0)
                translate_y = random.randint(-2, 2)
                resized_img = TF.affine(
                    resized_img,
                    angle=angle,
                    translate=[0, translate_y],
                    scale=1.0,
                    shear=[shear, 0.0],
                    fill=245,
                )

            # Normalisierung: Invertieren (Striche = 1.0, Hintergrund = 0.0) für sparse Features
            tensor_img = TF.to_tensor(resized_img)
            # Invertieren: Dunkle Tinte (nahe 0) wird zu Signal (nahe 1)
            tensor_img = 1.0 - tensor_img

            if self.augment:
                tensor_img = self._apply_tensor_augmentation(tensor_img)

        target_indices = self.vocab.encode(text)
        return tensor_img, target_indices, text

    def _apply_tensor_augmentation(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """
        Wendet schnelle On-the-fly-Augmentierungen auf dem invertierten Zeilen-Tensor (1, H, W) an:
        1. Kontrast-/Gamma-Variation (simuliert unterschiedlichen Bleistift-/Federdruck)
        2. Feines Sensor-/Papierrauschen
        3. SpecAugment für Handschrift (horizontale und vertikale Cutout-Masken),
           damit das BiLSTM/Attention-Modell den Satzkontext nutzt.
        """
        _, h, w = tensor_img.shape

        # 1. On-the-fly Simulation von unliniertem, weißem Papier (unterdrückt schwache Hilfslinien & Papiergrau)
        if random.random() < 0.5:
            bg_cutoff = random.uniform(0.14, 0.24)
            tensor_img = torch.where(tensor_img > bg_cutoff, tensor_img, torch.zeros_like(tensor_img))

        # 2. Kontrast / Strichdichte variieren
        if random.random() < 0.5:
            gain = random.uniform(0.85, 1.2)
            tensor_img = torch.clamp(tensor_img * gain, 0.0, 1.0)

        # 2. Leichtes Gaußsches Rauschen
        if random.random() < 0.35:
            noise = torch.randn_like(tensor_img) * random.uniform(0.01, 0.04)
            tensor_img = torch.clamp(tensor_img + noise, 0.0, 1.0)

        # 3. SpecAugment: 1-2 schmale Zeitmasken (vertikale Streifen)
        if w > 64 and random.random() < 0.5:
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                mask_w = random.randint(4, min(18, w // 10))
                x0 = random.randint(0, w - mask_w)
                tensor_img[:, :, x0 : x0 + mask_w] *= random.uniform(0.0, 0.25)

        # 4. SpecAugment: 1 schmale Frequenz-/Höhenmaske (horizontaler Streifen)
        if h >= 48 and random.random() < 0.3:
            mask_h = random.randint(2, 6)
            y0 = random.randint(0, h - mask_h)
            tensor_img[:, y0 : y0 + mask_h, :] *= random.uniform(0.0, 0.3)

        return tensor_img


def ctc_collate_fn(batch: List[Tuple[torch.Tensor, List[int], str]]) -> Dict[str, torch.Tensor]:
    """
    Kombiniert unterschiedlich breite Zeilenbilder in einen gepolsterten Batch Tensor.
    Berechnet die zeitlichen Sequenzlängen nach Durchlauf durch die CNN-Downsampling-Schichten.
    """
    # Maximale Breite in diesem Batch ermitteln
    max_w = max(item[0].shape[2] for item in batch)
    # Auf Vielfaches von 4 runden (wegen 2x MaxPool im CNN)
    if max_w % 4 != 0:
        max_w += (4 - (max_w % 4))

    batch_size = len(batch)
    height = batch[0][0].shape[1]

    # Gepolstertes Tensor-Gitter anlegen (Hintergrund = 0.0)
    padded_images = torch.zeros((batch_size, 1, height, max_w), dtype=torch.float32)

    targets_list: List[int] = []
    target_lengths: List[int] = []
    input_lengths: List[int] = []
    raw_texts: List[str] = []

    for i, (img, target, text) in enumerate(batch):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img

        targets_list.extend(target)
        target_lengths.append(len(target))
        raw_texts.append(text)

        # CNN reduziert die Breite um Faktor 4 (2x Stride 2)
        cnn_output_time_steps = max(1, w // 4)
        input_lengths.append(cnn_output_time_steps)

    return {
        "images": padded_images,
        "targets": torch.tensor(targets_list, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "raw_texts": raw_texts,
    }
