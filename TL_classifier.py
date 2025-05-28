import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np, cv2
from PIL import Image


class TLClassifier:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(self, ckpt_path: str, device: str | torch.device = "cpu"):
        self.device = torch.device(device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.class_names = ckpt["class_names"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    @torch.inference_mode()
    def __call__(self, img_bgr: np.ndarray) -> bool:
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected BGR ndarray with shape (H,W,3).")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        tensor = self.tfms(pil_img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        idx = logits.argmax(1).item()
        return self.class_names[idx] != "red"
