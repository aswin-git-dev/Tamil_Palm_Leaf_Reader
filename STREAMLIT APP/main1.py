# main1.py
import cv2
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import json
import os

# TrOCR model config
TROCR_DIR = "TR OCR"  # directory containing best_model.pth and id2label.json
MODEL_PATH = os.path.join(TROCR_DIR, "best_model.pth")
BASE_MODEL = "microsoft/trocr-base-handwritten"
IMG_SIZE = (384, 384)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load label map
with open(os.path.join(TROCR_DIR, "id2label.json"), "r", encoding="utf-8") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# Define TrOCR classifier
class TrOCRClassifier(torch.nn.Module):
    def __init__(self, base_model_name: str, num_classes: int):
        super().__init__()
        base_model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
        self.encoder = base_model.encoder
        hidden = self.encoder.config.hidden_size
        self.classifier = torch.nn.Linear(hidden, num_classes)
    def forward(self, pixel_values):
        enc_out = self.encoder(pixel_values=pixel_values)
        pooled = enc_out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

model = TrOCRClassifier(BASE_MODEL, len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# TrOCR processor and transform
processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
image_mean = processor.image_processor.image_mean
image_std = processor.image_processor.image_std
if len(image_mean) == 1:
    image_mean = image_mean * 3
    image_std = image_std * 3
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std)
])

# Set input shape for resize
h, w = IMG_SIZE
c = 3

# Label map
label_map = {
    0: 'மூ', 1: 'ா', 2: 'ட', 3: 'ம', 4: 'ப', 5: 'ெ', 6: 'ய', 7: 'ல ', 8: 'ன',
    9: 'ற', 10: 'க', 11: 'வ', 12: 'ண', 13: 'த', 14: 'ச', 15: 'ங', 16: 'ள',
    17: 'தி', 18: 'வி', 19: 'றி', 20: 'லி', 21: 'னி', 22: 'யி', 23: 'ழி',
    24: 'பி', 25: 'ரி', 26: 'சி', 27: 'ணி', 28: 'மி', 29: 'கி', 30: 'டு',
    31: 'கு', 32: 'ளு', 33: 'லு', 34: 'மு', 35: 'ணு', 36: 'னு', 37: 'ஞ',
    38: 'பு', 39: 'று', 40: 'ரு', 41: 'சு', 42: 'து', 43: 'வு', 44: 'யு',
    45: 'ழு', 46: 'டி', 47: 'ளி', 48: 'எ', 49: 'ழ', 50: 'கீ', 51: 'றூ',
    52: 'மீ', 53: 'வீ', 54: 'நூ', 55: 'றீ', 56: 'தீ', 57: 'கூ', 58: 'தூ',
    59: 'சூ', 60: 'யீ', 61: 'லூ', 62: 'உ', 63: 'அ', 64: 'ழீ', 65: 'யூ',
    66: 'சீ', 67: 'ணீ', 68: 'ஆ', 69: 'ளூ', 70: 'இ', 71: 'ை', 72: 'ர', 73: 'ந', 74: 'ஒ'
}

def process_letter(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"complete_line": "Invalid image", "words": [], "valid_segments": [], "rejected_segments": []}

    # Invert if background is white
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)


    # Resize and normalize for TrOCR
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        pred_label = torch.argmax(logits, dim=1).item()
        pred_character = label_map.get(pred_label, "Unknown")

    return {
        "complete_line": pred_character,
        "words": [pred_character],
        "valid_segments": [],  # no segments
        "rejected_segments": [],
        "preprocessing": (img, img, img, img, img, img),  # for visuals if needed
        "projection": []  # empty
    }
