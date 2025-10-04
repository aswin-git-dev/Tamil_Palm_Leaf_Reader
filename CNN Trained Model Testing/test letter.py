import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("vattaeluthu_tamil_advanced.keras")
_, h, w, c = model.input_shape
print("model input shape:", model.input_shape)

# Load input image (white bg with black letters or vice versa)
test_image_path = 'image_0001.png'  # replace with uploaded filename
img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# Invert image if the background is white
mean_pixel = np.mean(img)
if mean_pixel > 127:
    img = cv2.bitwise_not(img)  # invert to make bg black, letters white

# Resize image to model expected size
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# Normalize image to range [0, 1]
img_norm = img / 255.0

# Reshape for model input
if c == 1:
    input_img = img_norm.reshape(1, h, w, 1)
else:
    input_img = np.repeat(img_norm[..., np.newaxis], c, axis=-1)
    input_img = input_img.reshape(1, h, w, c)

# Label map (add your map here)
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

# Predict
pred = model.predict(input_img)
pred_label = np.argmax(pred)
pred_character = label_map.get(pred_label, "Unknown")

print("Predicted Vatta Ezhuthu:", pred_character)

# Show final preprocessed image
plt.figure(figsize=(4, 4))
plt.title(f"Preprocessed Input - Predicted: {pred_character}")
plt.axis('off')
plt.imshow(img_norm, cmap='gray')
plt.show()
