import torch
from src.data.dataset import SpeechCommandsDataset, DatasetManager
from src.data.transforms import get_base_transforms
from src.utils.labels import LabelManager
from src.models.clip_mlp import CLIPMLPClassifier
from src.models.cnn import SimpleCNN

def test_dataset_loading():
    root_dir = "google-speech-commands-spec"
    train_sp, val_sp, test_sp = DatasetManager.get_speaker_splits(root_dir)
    print(f"Speakers - Train: {len(train_sp)}, Val: {len(val_sp)}, Test: {len(test_sp)}")
    assert set(train_sp).isdisjoint(set(val_sp)), "Data leakage detected between Train and Val!"
    
    transform = get_base_transforms()
    ds = SpeechCommandsDataset(root_dir, train_sp, transform=transform)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        img, label = ds[0]
        print(f"Sample - Image shape: {img.shape}, Label index: {label}")
        assert img.shape == (3, 224, 224), "Image shape mismatch"
        assert 0 <= label < LabelManager.get_num_categories(), "Label index out of range"

def test_models_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = LabelManager.get_num_categories()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print("Testing CNN forward pass...")
    cnn = SimpleCNN(num_classes=num_classes).to(device)
    cnn_out = cnn(dummy_input)
    assert cnn_out.shape == (1, num_classes), f"CNN output shape mismatch: {cnn_out.shape}"
    
    print("Testing CLIP+MLP forward pass...")
    clip_mlp = CLIPMLPClassifier(num_classes=num_classes, device=device).to(device)
    clip_out = clip_mlp(dummy_input)
    assert clip_out.shape == (1, num_classes), f"CLIP+MLP output shape mismatch: {clip_out.shape}"
    print("Forward passes successful!")

if __name__ == '__main__':
    print("--- Starting Sanity Tests ---")
    test_dataset_loading()
    test_models_forward()
    print("--- All Sanity Tests Passed ---")
