import sys
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

def main(img_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    print("Predição:", classes[pred])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/predict.py caminho/da/imagem.jpg")
        sys.exit(1)
    main(sys.argv[1])
