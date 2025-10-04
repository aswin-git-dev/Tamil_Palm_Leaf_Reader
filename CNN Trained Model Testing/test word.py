import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# === Load model ===
model = load_model("vattaeluthu_tamil_advanced.keras")
_, h, w, c = model.input_shape
print("model input shape:", model.input_shape)

# === Label mapping ===
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

input_image_path = "w1.png"

# === Configuration for automatic filtering ===
MIN_CONFIDENCE = 0.2  # Minimum confidence threshold
MIN_AREA = 200        # Minimum bounding box area
MIN_HEIGHT = 20       # Minimum height for valid characters
MAX_ASPECT_RATIO = 5  # Maximum width/height ratio
MIN_PIXEL_DENSITY = 0.1  # Minimum ratio of non-zero pixels

# === Resize and pad ===
def resize_with_padding(image, size=(w, h), border=10):
    h_img, w_img = image.shape
    if h_img == 0 or w_img == 0:
        return None
    scale = min((size[0] - 2 * border) / w_img, (size[1] - 2 * border) / h_img)
    if scale <= 0:
        return None
    new_w, new_h = int(w_img * scale), int(h_img * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded = np.zeros(size, dtype=np.uint8)
    x_offset = (size[0] - new_w) // 2
    y_offset = (size[1] - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded

# === Function to check if segment is valid ===
def is_valid_segment(roi, bbox, confidence, predicted_char):
    x, y, w_box, h_box = bbox
    
    # Check basic geometric constraints
    area = w_box * h_box
    aspect_ratio = w_box / h_box if h_box > 0 else float('inf')
    
    # Check pixel density (ratio of non-zero pixels)
    non_zero_pixels = np.count_nonzero(roi)
    total_pixels = roi.size
    pixel_density = non_zero_pixels / total_pixels if total_pixels > 0 else 0
    
    # Apply filters
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

# === Read image and basic preprocessing ===
img = cv2.imread(input_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert if necessary
if np.mean(gray) > 127:
    gray = cv2.bitwise_not(gray)

# Simple adaptive thresholding
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 15, -2)

# Basic dilation to connect components
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Contour detection
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and sort bounding boxes
boxes = []
for cnt in contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    if w_box * h_box > 100 and h_box > 15:  # Basic size filter
        boxes.append((x, y, w_box, h_box))

boxes = sorted(boxes, key=lambda b: b[0])

print(f"Found {len(boxes)} potential character regions")

# Extract segments and predict
segments = []
predictions = []
valid_segments = []
rejected_segments = []

for idx, bbox in enumerate(boxes):
    x, y, w_box, h_box = bbox
    roi = gray[y:y+h_box, x:x+w_box]
    padded = resize_with_padding(roi, size=(w, h))
    
    if padded is not None:
        # Predict character
        img_norm = padded / 255.0
        if c == 1:
            input_img = img_norm.reshape(1, h, w, 1)
        else:
            input_img = np.repeat(img_norm[..., np.newaxis], c, axis=-1)
            input_img = input_img.reshape(1, h, w, c)

        pred = model.predict(input_img, verbose=0)
        label = np.argmax(pred)
        confidence = np.max(pred)
        char = label_map.get(label, "?")
        
        # Check if segment is valid
        is_valid, reason = is_valid_segment(roi, bbox, confidence, char)
        
        segment_info = {
            'idx': idx,
            'bbox': bbox,
            'roi': roi,
            'padded': padded,
            'char': char,
            'confidence': confidence,
            'is_valid': is_valid,
            'reason': reason
        }
        
        segments.append(segment_info)
        
        if is_valid:
            valid_segments.append(segment_info)
        else:
            rejected_segments.append(segment_info)

print(f"Valid segments: {len(valid_segments)}")
print(f"Rejected segments: {len(rejected_segments)}")

# Display all segments with validation status
if len(segments) > 0:
    cols = min(8, len(segments))
    rows = int(np.ceil(len(segments) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axs = [axs] if cols == 1 else axs
    else:
        axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    for i, segment in enumerate(segments):
        if i < len(axs):
            color = 'green' if segment['is_valid'] else 'red'
            axs[i].imshow(segment['padded'], cmap='gray')
            axs[i].set_title(f"{segment['idx']}: {segment['char']}\n"
                           f"Conf: {segment['confidence']:.2f}\n"
                           f"{'✓' if segment['is_valid'] else '✗'} {segment['reason']}", 
                           color=color, fontsize=9)
            axs[i].axis('off')
            
            # Add colored border
            for spine in axs[i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    plt.tight_layout()
    plt.suptitle("All Segments (Green=Valid, Red=Rejected)", y=1.02)
    plt.show()

# Display only valid segments
if len(valid_segments) > 0:
    predicted_word = ""
    plt.figure(figsize=(min(15, 2 * len(valid_segments)), 3))
    
    for i, segment in enumerate(valid_segments):
        predicted_word += segment['char']
        
        plt.subplot(1, len(valid_segments), i+1)
        plt.imshow(segment['padded'], cmap='gray')
        plt.title(f"{segment['char']}\n{segment['confidence']:.2f}", 
                 color='green', fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Valid Character Segments Only", y=1.02, fontweight='bold')
    plt.show()

    print(f"\n✅ Final Predicted Word: {predicted_word}")
    print(f"✅ Total valid characters: {len(valid_segments)}")
else:
    print("\n❌ No valid character segments found!")

# Print rejection summary
if len(rejected_segments) > 0:
    print(f"\n📋 Rejection Summary:")
    rejection_reasons = {}
    for segment in rejected_segments:
        reason = segment['reason']
        if reason not in rejection_reasons:
            rejection_reasons[reason] = 0
        rejection_reasons[reason] += 1
    
    for reason, count in rejection_reasons.items():
        print(f"  - {reason}: {count} segments")

# Optional: Save results
print(f"\n📊 Processing Summary:")
print(f"  - Total regions detected: {len(boxes)}")
print(f"  - Segments processed: {len(segments)}")
print(f"  - Valid segments: {len(valid_segments)}")
print(f"  - Rejected segments: {len(rejected_segments)}")
print(f"  - Success rate: {len(valid_segments)/len(segments)*100:.1f}%" if len(segments) > 0 else "  - Success rate: 0%")
