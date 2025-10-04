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

# Thresholds
MIN_CONFIDENCE = 0.1
MIN_AREA = 150
MIN_HEIGHT = 12
MAX_ASPECT_RATIO = 6
MIN_CHARACTER_WIDTH = 7
MAX_CHARACTER_WIDTH = 100

def preprocess(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    morphed = cv2.dilate(morphed, kernel, iterations=1)
    return img, gray, enhanced, binary, morphed

def resize_and_pad(image, target=(w, h), padding=5):
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)
    padded_img = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    scale = min((target[1]) / padded_img.shape[0], (target[0]) / padded_img.shape[1])
    resized = cv2.resize(padded_img, (int(padded_img.shape[1]*scale), int(padded_img.shape[0]*scale)))
    final = np.zeros(target, dtype=np.uint8)
    x_offset = (target[0] - resized.shape[1]) // 2
    y_offset = (target[1] - resized.shape[0]) // 2
    final[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    return final

def extract_segments(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > MIN_AREA and h_box > MIN_HEIGHT and MIN_CHARACTER_WIDTH <= w_box <= MAX_CHARACTER_WIDTH:
            segments.append((x, y, w_box, h_box))
    return sorted(segments, key=lambda b: (b[1]//25, b[0]))

def get_projection(binary_img):
    return np.sum(binary_img, axis=0) / 255.0

def process_sentence(image_path):
    original, gray, enhanced, binary, morphed = preprocess(image_path)
    segments = extract_segments(morphed)

    segment_data = []
    padded_images = []

    for idx, (x, y, w_box, h_box) in enumerate(segments):
        roi = binary[y:y+h_box, x:x+w_box]
        padded = resize_and_pad(roi)
        if padded is not None:
            padded_images.append(padded)
            segment_data.append({
                'idx': idx,
                'bbox': (x, y, w_box, h_box),
                'roi': roi,
                'padded': padded
            })

    valid_segments = []
    rejected_segments = []

    if not padded_images:
        return {}


    padded_tensors = []
    for padded in padded_images:
        img_rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = transform(pil_img)
        padded_tensors.append(input_tensor)

    if padded_tensors:
        batch_tensor = torch.stack(padded_tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.max(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
        for i, seg in enumerate(segment_data):
            label = preds[i]
            confidence = confidences[i]
            char = label_map.get(label, "?")
            x, y, w_box, h_box = seg['bbox']
            aspect_ratio = w_box / h_box if h_box else 0
            pixel_density = np.count_nonzero(seg['roi']) / seg['roi'].size if seg['roi'].size else 0
            is_valid = (confidence >= MIN_CONFIDENCE and 
                        aspect_ratio <= MAX_ASPECT_RATIO and
                        pixel_density > 0.08)
            seg_info = {
                **seg,
                'char': char,
                'confidence': confidence,
                'label': label,
                'is_valid': is_valid,
                'reason': "Valid" if is_valid else "Low confidence / bad shape"
            }
            if is_valid:
                valid_segments.append(seg_info)
            else:
                rejected_segments.append(seg_info)

    lines = {}
    for seg in valid_segments:
        x, y, w_box, h_box = seg['bbox']
        line_key = y // 25
        lines.setdefault(line_key, []).append(seg)

    words_data = []
    predicted_line = []

    for line_idx in sorted(lines.keys()):
        chars = sorted(lines[line_idx], key=lambda s: s['bbox'][0])
        word = ''.join([s['char'] for s in chars])
        predicted_line.append(word)
        words_data.append((word, chars))

    return {
        'complete_line': ' '.join(predicted_line),
        'words': predicted_line,
        'word_segments': words_data,
        'valid_segments': valid_segments,
        'rejected_segments': rejected_segments,
        'total_confidence': np.mean([s['confidence'] for s in valid_segments]) if valid_segments else 0.0,
        'projection': get_projection(binary),
        'preprocessing': (original, gray, enhanced, binary, morphed)
    }