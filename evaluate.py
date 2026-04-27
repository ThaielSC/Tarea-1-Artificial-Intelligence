import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from src.data.dataset import SpeechCommandsDataset, DatasetManager
from src.data.transforms import get_base_transforms
from src.models.clip_mlp import CLIPMLPClassifier
from src.models.cnn import SimpleCNN
from src.utils.labels import LabelManager

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return all_labels, all_preds

def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = "google-speech-commands-spec"
    num_classes = LabelManager.get_num_categories()
    categories = LabelManager.get_all_categories()
    
    # Load Test Data (10% speakers reserved)
    _, _, test_sp = DatasetManager.get_speaker_splits(root_dir)
    transform = get_base_transforms()
    test_ds = SpeechCommandsDataset(root_dir, test_sp, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    
    print(f"Evaluating on {len(test_ds)} test samples (Speakers: {len(test_sp)})")

    # 1. Evaluate CNN
    print("\n" + "="*30)
    print("Métrica: CNN CUSTOM")
    print("="*30)
    cnn = SimpleCNN(num_classes=num_classes).to(device)
    cnn.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
    y_true, y_pred = evaluate_model(cnn, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=categories))

    # 2. Evaluate CLIP+MLP
    print("\n" + "="*30)
    print("Métrica: CLIP + MLP")
    print("="*30)
    clip_mlp = CLIPMLPClassifier(num_classes=num_classes, device=device).to(device)
    clip_mlp.load_state_dict(torch.load("best_clip_model.pth", map_location=device))
    y_true, y_pred = evaluate_model(clip_mlp, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=categories))

if __name__ == "__main__":
    run_comparison()
