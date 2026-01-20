import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def count_images_by_class(root_dir: str):
    """
    Retorna:
      classes: list[str]
      counts: dict[str, int]
    """
    classes = []
    counts = {}

    if not os.path.isdir(root_dir):
        return [], {}

    for name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, name)
        if not os.path.isdir(class_dir):
            continue

        classes.append(name)
        c = 0
        for f in os.listdir(class_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXTS:
                c += 1
        counts[name] = c

    return classes, counts

def has_class_folders(root_dir: str):
    if not os.path.isdir(root_dir):
        return False
    for name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, name)):
            return True
    return False

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dir = "data/train"
    val_dir = "data/val"

    # --- Diagnóstico do dataset ---
    train_classes, train_counts = count_images_by_class(train_dir)
    if not train_classes:
        raise FileNotFoundError(
            f"Não achei classes em {train_dir}.\n"
            f"Crie assim: data/train/classeA/*.jpg e data/train/classeB/*.jpg"
        )

    print("\n Dataset (train):")
    for cls in train_classes:
        print(f"  - {cls}: {train_counts.get(cls, 0)} imagens")


    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    use_val_folder = has_class_folders(val_dir)

    if use_val_folder:
        val_classes, val_counts = count_images_by_class(val_dir)
        print("\n Dataset (val):")
        for cls in val_classes:
            print(f"  - {cls}: {val_counts.get(cls, 0)} imagens")

        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        class_names = train_ds.classes

    else:
        print("\n Não encontrei pastas de classe em data/val.")
        print(" Vou dividir automaticamente o data/train em treino + validação (80/20).")

        full_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_size = max(1, int(0.2 * len(full_ds)))
        train_size = len(full_ds) - val_size


        g = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=g)


        val_ds.dataset.transform = val_tf

        class_names = full_ds.classes

    print("\n Classes detectadas:", class_names)

    # loaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # modelo
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, 11):
        # treino
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss /= max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": class_names
            }, "checkpoints/best.pt")
            print(f" Salvou best.pt (val_acc={best_val_acc:.3f})")

if __name__ == "__main__":
    main()
