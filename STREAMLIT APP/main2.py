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

# === Label map ===
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

# === Preprocessing utilities ===
def resize_with_padding(image, size=(w, h), border=10, extra_padding=4):
    h_img, w_img = image.shape
    if h_img == 0 or w_img == 0:
        return None

    # Step 1: Resize with internal scaling border
    scale = min((size[0] - 2 * border) / w_img, (size[1] - 2 * border) / h_img)
    if scale <= 0:
        return None

    new_w, new_h = int(w_img * scale), int(h_img * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros(size, dtype=np.uint8)
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Step 2: Add external extra padding (black border)
    if extra_padding > 0:
        padded = cv2.copyMakeBorder(
            padded, 
            extra_padding, extra_padding, extra_padding, extra_padding, 
            borderType=cv2.BORDER_CONSTANT, 
            value=0  # 0 for black; use 255 for white
        )

        # Crop back to original size in case it exceeds model input shape
        padded = cv2.resize(padded, size, interpolation=cv2.INTER_AREA)

    return padded


def is_valid_segment(roi, bbox, confidence, predicted_char):
    MIN_CONFIDENCE = 0.2
    MIN_AREA = 200
    MIN_HEIGHT = 20
    MAX_ASPECT_RATIO = 5
    MIN_PIXEL_DENSITY = 0.1

    x, y, w_box, h_box = bbox
    area = w_box * h_box
    aspect_ratio = w_box / h_box if h_box > 0 else float('inf')
    non_zero_pixels = np.count_nonzero(roi)
    pixel_density = non_zero_pixels / roi.size if roi.size > 0 else 0

    if confidence < MIN_CONFIDENCE:
        return False, f"Low confidence: {confidence:.2f}"
    if area < MIN_AREA:
        return False, f"Small area: {area}"
    if h_box < MIN_HEIGHT:
        return False, f"Too short: {h_box}"
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"Bad aspect ratio: {aspect_ratio:.2f}"
    if pixel_density < MIN_PIXEL_DENSITY:
        return False, f"Low pixel density: {pixel_density:.2f}"
    if predicted_char == "?":
        return False, "Unrecognized character"
    return True, "Valid"

def process_word(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 15, -2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if w_box * h_box > 100 and h_box > 15:
            boxes.append((x, y, w_box, h_box))

    boxes = sorted(boxes, key=lambda b: b[0])

    segments = []
    valid_segments = []
    rejected_segments = []
    final_word = ""


    padded_tensors = []
    segment_infos = []
    for idx, bbox in enumerate(boxes):
        x, y, w_box, h_box = bbox
        roi = gray[y:y+h_box, x:x+w_box]
        padded = resize_with_padding(roi, size=(w, h))
        if padded is not None:
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
            pil_img = Image.fromarray(img_rgb)
            input_tensor = transform(pil_img)
            padded_tensors.append(input_tensor)
            segment_infos.append({
                'idx': idx,
                'bbox': bbox,
                'roi': roi,
                'padded': padded
            })

    if padded_tensors:
        batch_tensor = torch.stack(padded_tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            confidences = torch.max(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
        for i, seg in enumerate(segment_infos):
            label = preds[i]
            confidence = confidences[i]
            char = label_map.get(label, "?")
            is_valid, reason = is_valid_segment(seg['roi'], seg['bbox'], confidence, char)
            segment_info = {
                **seg,
                'char': char,
                'confidence': confidence,
                'is_valid': is_valid,
                'reason': reason
            }
            segments.append(segment_info)
            if is_valid:
                valid_segments.append(segment_info)
                final_word += char
            else:
                rejected_segments.append(segment_info)

    word_result = final_word.strip()
    return {
        "complete_line": word_result,
        "words": [word_result],
        "valid_segments": valid_segments,
        "rejected_segments": rejected_segments,
        "word_segments": [(word_result, valid_segments)],
        "preprocessing": (img, gray, thresh, dilated, None, None),
        "projection": np.sum(thresh == 0, axis=0).tolist() if thresh is not None else []
    }