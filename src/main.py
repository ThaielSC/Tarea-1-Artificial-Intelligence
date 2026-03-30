import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from PIL import Image
import random


SEED = 42
np.random.seed(SEED)
random.seed(SEED)


DATASET_PATH = "google-speech-commands-spec"
TARGET_SHAPE = (1025, 28)


def process_image_resize(img):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)

    img_rescaled = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_rescaled)

    resized_pil = pil_img.resize(
        (TARGET_SHAPE[1], TARGET_SHAPE[0]), Image.Resampling.LANCZOS
    )

    return np.array(resized_pil).flatten() / 255.0


def load_and_preprocess(num_samples=2377):
    X = []
    y = []

    categories = [
        d
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith("_")
    ]

    target_categories = ["yes", "bed", "cat"]
    other_categories = [c for c in categories if c not in target_categories]

    # Load targets
    for idx, target in enumerate(target_categories):
        target_path = os.path.join(DATASET_PATH, target)
        target_files = os.listdir(target_path)
        random.shuffle(target_files)
        target_files = target_files[:num_samples]

        print(f"Loading {len(target_files)} samples for category '{target}'...")
        for f in target_files:
            img = plt.imread(os.path.join(target_path, f))
            X.append(process_image_resize(img))
            y.append(idx + 1)

    # Load others
    samples_per_other = num_samples // len(other_categories) + 1
    count_0 = 0

    print("Loading samples for other categories...")
    for cat in other_categories:
        cat_path = os.path.join(DATASET_PATH, cat)
        cat_files = os.listdir(cat_path)
        random.shuffle(cat_files)
        cat_files = cat_files[:samples_per_other]
        for f in cat_files:
            if count_0 >= num_samples:
                break
            img = plt.imread(os.path.join(cat_path, f))
            X.append(process_image_resize(img))
            y.append(0)
            count_0 += 1

    return np.array(X), np.array(y)


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    target_names = ["others", "yes", "bed", "cat"]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    return acc


if __name__ == "__main__":
    print("--- Training Model (Recognizing: yes, bed, cat) ---")
    num_samples = 1700

    X, y = load_and_preprocess(num_samples=num_samples)
    accuracy = train_and_evaluate(X, y)

    print(f"\nFinal Accuracy: {accuracy:.4f}")
