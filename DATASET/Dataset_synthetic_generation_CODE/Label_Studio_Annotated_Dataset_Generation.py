import os
import cv2
import json
import pandas as pd

CSV_FILE = "project-1-at-2025-06-19-11-31-cdbbe6ab.csv"
OUTPUT_DIR = "dataset"
base_path = "C:/Users/HP/Downloads"

df = pd.read_csv(CSV_FILE, encoding='utf-8')
df = df[~df['ocr'].isna()]
df['ocr'] = df['ocr'].astype(str)

for index, row in df.iterrows():
    try:
        image_path = os.path.normpath(os.path.join(base_path, "." + row['ocr']))
        print(f"\n Processing row {index}")
        print(f"Full image path: {image_path}")

        if not os.path.exists(image_path):
            print(f" Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f" Failed to load image: {image_path}")
            continue

        h_img, w_img = image.shape[:2]
        print(f" Image loaded: {image_path} (W:{w_img}, H:{h_img})")

        bboxes = json.loads(row['bbox'])
        labels = json.loads(row['transcription'])

        if len(bboxes) != len(labels):
            print(f"Mismatch: {len(bboxes)} boxes vs {len(labels)} labels at row {index}")
            continue

        if len(bboxes) == 0:
            print(f" No bounding boxes at row {index}")
            continue

        for i, (box, label) in enumerate(zip(bboxes, labels)):
            try:
                x = int((box['x'] / 100) * w_img)
                y = int((box['y'] / 100) * h_img)
                w = int((box['width'] / 100) * w_img)
                h = int((box['height'] / 100) * h_img)

                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > w_img or y + h > h_img:
                    print(f"️ Invalid box {i}: ({x},{y},{w},{h}) skipped.")
                    continue

                cropped = image[y:y+h, x:x+w]
                if cropped.size == 0:
                    print(f" Empty crop at row {index}, box {i}")
                    continue

                cropped = cv2.resize(cropped, (32, 32))


                safe_label = f"label_{ord(label)}" if len(label) == 1 else f"label_{hash(label)}"
                label_folder = os.path.join(OUTPUT_DIR, safe_label)
                os.makedirs(label_folder, exist_ok=True)

                out_path = os.path.join(label_folder, f"{index}_{i}.png")
                cv2.imwrite(out_path, cropped)

                if os.path.exists(out_path):
                    print(f" Saved: {out_path}")
                else:
                    print(f" Failed to save: {out_path}")

            except Exception as crop_error:
                print(f" Error cropping row {index}, box {i}: {crop_error}")
                continue

    except Exception as e:
        print(f" General error at row {index}: {e}")
        continue

print("\nDataset generation complete.")
