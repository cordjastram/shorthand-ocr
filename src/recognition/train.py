"""
Training-Skript für DEK-CRNN mit CTC-Loss und GPU-Beschleunigung.

Optimiert für NVIDIA RTX A500 (4 GB VRAM):
- Automatische CUDA- und Tensor-Core-Erkennung
- Mixed-Precision Training (FP16 / GradScaler)
- Speicherschonende DataLoader mit variabler Zeilenbreite
- Validierungsschleife mit Stichproben-Dekodierung
- Automatisches Speichern der besten Modellgewichte
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Projektverzeichnis zum Pfad hinzufügen
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition.vocabulary import DekVocabulary
from src.recognition.dataset import DekLineDataset, ctc_collate_fn
from src.recognition.model import DekCRNN, build_htr_model


def train(
    manifest_csv: str = "data/synthetic/historical_pretraining/lines_manifest.csv",
    models_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    use_fp16: bool = True,
    resume: bool = True,
    model_filename: str = "dek_crnn_best.pt",
    arch: str = "resnet_attention",
    target_height: int = 96,
):
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Hardware prüfen
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"Kurzschrift HTR - Modelltraining ({arch.upper()}, H={target_height}px)")
    print("=" * 65)
    print(f"Rechengerät: {device.type.upper()}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        print(f"  • GPU-Modell: {gpu_name}")
        print(f"  • Verfügbarer VRAM: {vram_gb} GB")
        print(f"  • FP16 Mixed Precision: {'Aktiviert (Tensor Cores)' if use_fp16 else 'Deaktiviert'}")
    else:
        print("  • HINWEIS: Training läuft auf CPU.")
    print("=" * 65)

    # 2. Vokabular & Datasets laden
    vocab = DekVocabulary()
    print(f"\nVokabular geladen: {len(vocab)} Zeichen (inkl. Blank-Token)")

    train_dataset = DekLineDataset(
        manifest_csv, split="train", vocab=vocab, target_height=target_height, augment=True
    )
    val_dataset = DekLineDataset(
        manifest_csv, split="val", vocab=vocab, target_height=target_height, augment=False
    )

    print(f"Datensatz: {len(train_dataset)} Trainingszeilen (On-the-fly Augment) | {len(val_dataset)} Validierungszeilen")
    if len(train_dataset) == 0:
        raise ValueError(f"Keine Trainingsdaten im Manifest gefunden: {manifest_csv}")

    # DataLoader (num_workers=0 für maximale Windows-Stabilität)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Modell initialisieren
    model = build_htr_model(arch=arch, num_classes=len(vocab), in_channels=1)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modell ({arch}) initialisiert: {total_params:,} trainierbare Parameter (~{total_params * 4 / (1024**2):.1f} MB)")

    # 4. Verlustfunktion, Optimierer & CosineAnnealingLR Scheduler
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=lr * 0.03
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and use_fp16))

    best_val_loss = float("inf")
    best_model_file = models_path / model_filename
    start_epoch = 1

    if resume and best_model_file.exists():
        ckpt = torch.load(best_model_file, map_location=device, weights_only=False)
        ckpt_arch = ckpt.get("arch", "crnn")
        if ckpt_arch == arch:
            print(f"\nLade bestehenden Checkpoint: {best_model_file}...")
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            # Scheduler auf die Ziel-Epochenzahl synchronisieren
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(start_epoch, epochs), eta_min=lr * 0.03
            )
            for _ in range(1, start_epoch):
                scheduler.step()
            print(f"  -> Setze Training ab Epoche {start_epoch:02d}/{epochs:02d} fort (bisheriger Best Val Loss: {best_val_loss:.4f})")
        else:
            print(f"\nHinweis: Checkpoint {best_model_file} hat Architektur '{ckpt_arch}' statt '{arch}'. Starte neues Training ab Epoche 1.")

    # 5. Training-Schleife
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoche {epoch:02d}/{epochs:02d} [Train]")

        for batch in train_pbar:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_fp16)):
                # log_probs shape: (Time, Batch, Num_Classes)
                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step()
        avg_train_loss = train_loss / max(1, len(train_loader))

        # 5b. Windows-Standby-Schutz pro Epoche erneuern
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            except Exception:
                pass

        # 6. Validierung
        model.eval()
        val_loss = 0.0
        sample_preds = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                targets = batch["targets"].to(device)
                input_lengths = batch["input_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and use_fp16)):
                    log_probs = model(images)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                val_loss += loss.item()

                # Greedy CTC Dekodierung für die ersten 2 Zeilen des ersten Batches
                if len(sample_preds) < 2:
                    # log_probs: (Time, Batch, Num_Classes) -> argmax über Klassen
                    preds = log_probs.argmax(dim=-1).transpose(0, 1)  # (Batch, Time)
                    for b_idx in range(min(2, preds.shape[0])):
                        pred_indices = preds[b_idx].tolist()
                        pred_text = vocab.ctc_decode(pred_indices)
                        true_text = batch["raw_texts"][b_idx]
                        sample_preds.append((true_text, pred_text))

        avg_val_loss = val_loss / max(1, len(val_loader))

        print(f"\nEpoche {epoch:02d} abgeschlossen: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Stichprobe ausgeben
        if sample_preds:
            print("  --- Stichproben-Erkennung ---")
            for t_text, p_text in sample_preds[:2]:
                print(f"  SOLL : \"{t_text}\"")
                print(f"  IST  : \"{p_text}\"")

        # Bestes Modell sichern
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "arch": arch,
                "target_height": target_height,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "vocab": vocab.char_to_idx,
            }, best_model_file)
            print(f"  -> Neues bestes Modell gespeichert ({best_model_file.name})!")

    print("\n" + "=" * 65)
    print(f"Training erfolgreich beendet! Bestes Modell: {best_model_file}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainiere DEK-CRNN Modell.")
    parser.add_argument("--manifest", type=str, default="data/synthetic/historical_pretraining/lines_manifest.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train(
        manifest_csv=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
