import os
import random
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.utils.labels import LabelManager, Category

class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir: str, speaker_ids: List[str], transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self._load_dataset(speaker_ids)

    def _load_dataset(self, allowed_speakers: List[str]):
        allowed_set = set(allowed_speakers)
        for category_name in os.listdir(self.root_dir):
            cat_path = os.path.join(self.root_dir, category_name)
            if not os.path.isdir(cat_path):
                continue
            
            try:
                cat_enum = Category(category_name)
                cat_idx = LabelManager.get_category_idx(cat_enum)
            except ValueError:
                continue

            for word_name in os.listdir(cat_path):
                word_path = os.path.join(cat_path, word_name)
                if not os.path.isdir(word_path):
                    continue
                
                for file_name in os.listdir(word_path):
                    if not file_name.endswith(".png"):
                        continue
                    
                    speaker_id = file_name.split("_")[0]
                    if speaker_id in allowed_set:
                        full_path = os.path.join(word_path, file_name)
                        self.samples.append((full_path, cat_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

class DatasetManager:
    @staticmethod
    def get_speaker_splits(root_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
        all_speakers = set()
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".png"):
                    all_speakers.add(f.split("_")[0])
        
        speakers_list = sorted(list(all_speakers))
        random.seed(42)
        random.shuffle(speakers_list)
        
        n = len(speakers_list)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        return (
            speakers_list[:train_idx],
            speakers_list[train_idx:val_idx],
            speakers_list[val_idx:]
        )
