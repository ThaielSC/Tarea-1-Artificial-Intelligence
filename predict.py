import argparse
import torch
from PIL import Image
from src.data.transforms import get_base_transforms
from src.models.clip_mlp import CLIPMLPClassifier
from src.models.cnn import SimpleCNN
from src.utils.labels import LabelManager

def predict(image_path: str, model_type: str, checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = LabelManager.get_num_categories()
    
    # Load Model
    if model_type == "clip":
        model = CLIPMLPClassifier(num_classes=num_classes, device=device)
    else:
        model = SimpleCNN(num_classes=num_classes)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Preprocess Image
    transform = get_base_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        category = LabelManager.get_category_from_idx(predicted.item())
        
    print(f"Model: {model_type.upper()}")
    print(f"Image: {image_path}")
    print(f"Predicted Category: {category.value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Hito 2")
    parser.add_argument("--image", type=str, required=True, help="Path to the spectrogram image")
    parser.add_argument("--model", type=str, choices=["clip", "cnn"], default="cnn")
    parser.add_argument("--checkpoint", type=str, help="Path to model weights")
    
    args = parser.parse_args()
    
    # Default checkpoint path if not provided
    checkpoint = args.checkpoint or f"best_{args.model}_model.pth"
    
    predict(args.image, args.model, checkpoint)
