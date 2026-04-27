import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import SpeechCommandsDataset, DatasetManager
from src.data.transforms import get_base_transforms
from src.models.clip_mlp import CLIPMLPClassifier
from src.models.cnn import SimpleCNN
from src.training.trainer import Trainer
from src.utils.labels import LabelManager

def run_experiment(model_type: str, epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data Setup
    root_dir = "google-speech-commands-spec"
    train_sp, val_sp, test_sp = DatasetManager.get_speaker_splits(root_dir)
    
    transform = get_base_transforms()
    
    train_ds = SpeechCommandsDataset(root_dir, train_sp, transform=transform)
    val_ds = SpeechCommandsDataset(root_dir, val_sp, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = LabelManager.get_num_categories()

    # Model Selection
    if model_type == "clip":
        model = CLIPMLPClassifier(num_classes=num_classes, device=device)
    else:
        model = SimpleCNN(num_classes=num_classes)

    trainer = Trainer(model, device, lr=lr)

    # Training Loop
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            trainer.save_checkpoint(f"best_{model_type}_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hito 2: Voice Command Classification")
    parser.add_argument("--model", type=str, choices=["clip", "cnn"], default="clip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    run_experiment(args.model, args.epochs, args.batch_size, args.lr)
