import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import json

# For Google Colab: Mount Google Drive if running in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted at /content/drive")
except ImportError:
    pass

print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

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

# Thresholds
MIN_CONFIDENCE = 0.1
MIN_AREA = 150
MIN_HEIGHT = 18
MAX_ASPECT_RATIO = 5
MIN_CHARACTER_WIDTH = 10
MAX_CHARACTER_WIDTH = 80

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

def resize_and_pad(image, target=IMG_SIZE, padding=5):
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

def predict_char(image_np):
    img_pil = Image.fromarray(image_np).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0][pred_idx].item()
        return label_map.get(pred_idx, "?"), confidence

def process_paragraph(image_path):
    original, gray, enhanced, binary, morphed = preprocess(image_path)
    segments = extract_segments(morphed)
    valid_segments = []
    rejected_segments = []
    if not segments:
        return {}
    predicted_line = []
    words_data = []
    for idx, (x, y, w_box, h_box) in enumerate(segments):
        roi = binary[y:y+h_box, x:x+w_box]
        padded = resize_and_pad(roi)
        char, conf = predict_char(padded)
        aspect_ratio = w_box / h_box if h_box else 0
        pixel_density = np.count_nonzero(roi) / roi.size if roi.size else 0
        is_valid = (conf >= MIN_CONFIDENCE and aspect_ratio <= MAX_ASPECT_RATIO and pixel_density > 0.08)
        seg_info = {
            'idx': idx,
            'bbox': (x, y, w_box, h_box),
            'roi': roi,
            'padded': padded,
            'char': char,
            'confidence': conf,
            'label': char,
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
        'preprocessing': (original, gray, enhanced, binary, morphed)
    }

if __name__ == "__main__":
    # For Colab: set image_path here, e.g. image_path = '/content/drive/MyDrive/path/to/image.png'
    image_path = ''  # <-- Set your image path here
    if not image_path:
        print("Please set the image_path variable to your image file.")
    else:
        result = process_paragraph(image_path)
        print("Predicted Output:")
        print(result['complete_line'])
        print("Details:")
        print(result)
