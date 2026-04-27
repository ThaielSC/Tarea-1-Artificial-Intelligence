import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model: nn.Module, device: str, lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for images, labels in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        return avg_loss, accuracy

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)
