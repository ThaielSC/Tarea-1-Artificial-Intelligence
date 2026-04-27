import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor

class CLIPFeatureExtractor:
    def __init__(self, device: str, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(pixel_values=images.to(self.device))
            return outputs.pooler_output

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class CLIPMLPClassifier(nn.Module):
    def __init__(self, num_classes: int, device: str, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.extractor = CLIPFeatureExtractor(device, model_name)
        # CLIP-ViT-B/32 has 768 output dimensions for vision pooler
        self.mlp = MultiLayerPerceptron(input_dim=768, hidden_dim=256, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extractor.get_features(images)
        return self.mlp(features)
