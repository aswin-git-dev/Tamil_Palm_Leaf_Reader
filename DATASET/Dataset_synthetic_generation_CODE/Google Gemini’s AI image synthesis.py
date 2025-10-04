#!/usr/bin/env python3
"""
Gemini-Only Synthetic Dataset Generator for Tamil Vatteluttu OCR

This script uses Google's Gemini multimodal generative model to create
handwriting-style variations of class images (Tamil characters) to top up
each class folder to a desired target count.

No Albumentations or local geometric augmentations are applied. All stylistic
variation is requested from Gemini. Images are resized before sending and
after receiving to ensure uniform size.

"""

import os
import io
import sys
import cv2
import base64
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import google.generativeai as genai


# ================= USER SETTINGS =================
# Path to your input dataset containing numeric subfolders (0..N-1)
DATASET_ROOT = r"C:\D\Tamil NLP Project\datasets\datasets_new"   # <-- change

# Path to output (augmented) dataset; originals copied here + Gemini variants
OUTPUT_ROOT  = r"C:\D\Tamil NLP Project\datasets\datas"    # <-- change

# Number of class folders (e.g., 75 means folders 0..74)
NUM_CLASSES  = 75

# Target images per class after augmentation
TARGET_COUNT = 200

# Final normalized image size (square)
IMG_SIZE     = 448

# How many Gemini style variants to request *per API call*
GEM_BATCH    = 5   # Gemini asked to produce up to this many variants per call

# Gemini model name (try "gemini-1.5-pro" for higher quality if quota allows)
GEMINI_MODEL = "gemini-2.5-flash"

# Inline API key (REPLACE WITH YOUR *NEW* KEY; DO NOT SHARE PUBLICLY)
GOOGLE_API_KEY = "PUT_API_KEY_HERE" #FOR DOUBLE BLIND REVIEW THE API KEY IS REMOVED HERE, YOU CAN TEST USING ANY GEMINI API KEY- FOR TESTING THIS CODE 

GEMINI_PROMPT_STYLE = (
    "You are generating synthetic training data for a historical Tamil Vatteluttu OCR model. "
    "Given a single example glyph image, generate diverse handwritten-style variants that represent the *same* character. "
    "Apply variations in stroke thickness, stroke curvature, ink flow, brush/pen pressure, natural writing slant, and aging effects such as mild fading or bleed. "
    "Ensure each generated image contains only one glyph, centered, with no noise, borders, artifacts, or text outside the character. "
    "Output each image as a clean black glyph on a transparent or pale off-white background, in a centered square format (e.g., 224x224). "
    "Maintain legibility while simulating realistic handwriting diversity."
)

# ==================================================


# ------------------- GEMINI INIT -------------------
def init_gemini():
    if not GOOGLE_API_KEY or GOOGLE_API_KEY.startswith("PASTE_"):
        print("[ERR] Google API key not set. Please edit GOOGLE_API_KEY in script.")
        sys.exit(1)
    genai.configure(api_key=GOOGLE_API_KEY)
    try:
        return genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        print(f"[ERR] Gemini init failed: {e}")
        sys.exit(1)


# ---------------- IMAGE HELPERS --------------------
def load_image_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"failed to read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if IMG_SIZE is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img

def save_image_rgb(img_rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)

def encode_img_to_png_bytes(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def decode_response_image_bytes(img_bytes):
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)
        if IMG_SIZE is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None

def decode_base64_to_rgb(b64_str):
    try:
        raw = base64.b64decode(b64_str)
        return decode_response_image_bytes(raw)
    except Exception:
        return None


# --------- GEMINI VARIANT GENERATION ---------------
def gemini_generate_variants(model, seed_img_rgb, class_id, want_variants):
    """
    Ask Gemini for up to `want_variants` handwriting-style variants of seed_img_rgb.
    Returns a list of numpy RGB images.
    """
    png_bytes = encode_img_to_png_bytes(seed_img_rgb)

    # We request multiple variants in one call via instructions in text prompt.
    # Gemini may return fewer than requested; we'll loop at call site if needed.
    prompt = (
        GEMINI_PROMPT_STYLE
        + f" This image belongs to class id {class_id}. "
        f"Please return up to {want_variants} distinct variant images."
    )

    try:
        # multimodal input: text + image
        resp = model.generate_content(
            [
                prompt,
                {"mime_type": "image/png", "data": png_bytes},
            ],
            generation_config={"temperature": 0.95, "top_p": 0.95},
        )
    except Exception as e:
        print(f"[WARN] Gemini request failed for class {class_id}: {e}")
        return []

    imgs = []

    # Gemini responses are structured as parts; images may be inline
    for part in resp.parts:
        # Inline image binary
        if hasattr(part, "inline_data") and part.inline_data:
            if part.inline_data.mime_type.startswith("image/"):
                arr = decode_response_image_bytes(part.inline_data.data)
                if arr is not None:
                    imgs.append(arr)

        # Text that might contain base64
        elif hasattr(part, "text") and part.text:
            txt = part.text.strip()
            # naive base64 sniff
            if "base64" in txt.lower():
                candidate = txt.split()[-1].strip()
                arr = decode_base64_to_rgb(candidate)
                if arr is not None:
                    imgs.append(arr)

    return imgs


# ----------------- MAIN LOOP -----------------------
def main():
    model = init_gemini()
    report = {}

    # Ensure output root exists
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(DATASET_ROOT, str(class_id))
        out_dir   = os.path.join(OUTPUT_ROOT,  str(class_id))
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(class_dir):
            print(f"[WARN] missing input folder: {class_dir}")
            report[class_id] = 0
            continue

        # Collect source images
        src_imgs = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not src_imgs:
            print(f"[WARN] no images in {class_dir}")
            report[class_id] = 0
            continue

        # Copy originals (resized) into output
        copied = 0
        for name in src_imgs:
            src = os.path.join(class_dir, name)
            dst = os.path.join(out_dir, os.path.splitext(name)[0] + ".png")  # normalize ext
            if os.path.exists(dst):
                continue
            try:
                img = load_image_rgb(src)
                save_image_rgb(img, dst)
                copied += 1
            except Exception as e:
                print(f"[ERR] copy failed {src}: {e}")

        # Count current images in out_dir
        cur_imgs = [
            f for f in os.listdir(out_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        have = len(cur_imgs)
        need = max(0, TARGET_COUNT - have)

        print(f"[INFO] Class {class_id}: have={have}, need={need}")
        if need == 0:
            report[class_id] = 0
            continue

        generated = 0
        pbar = tqdm(total=need, desc=f"class {class_id}", leave=False)

        # We'll repeatedly call Gemini until we meet need
        while generated < need:
            remain = need - generated
            batch_n = min(GEM_BATCH, remain)

            # pick a random seed image from the *input* originals (so Gemini sees raw user data)
            seed_name = random.choice(src_imgs)
            seed_img = load_image_rgb(os.path.join(class_dir, seed_name))

            variants = gemini_generate_variants(model, seed_img, class_id, batch_n)

            if not variants:
                # If Gemini failed to produce output, to avoid stuck loop we copy the seed again w/ suffix
                fallback = f"gem_fallback_{generated}.png"
                save_image_rgb(seed_img, os.path.join(out_dir, fallback))
                generated += 1
                pbar.update(1)
                continue

            # Save returned variants
            for v_img in variants:
                if generated >= need:
                    break
                fname = f"gem_{class_id}_{generated}.png"
                save_image_rgb(v_img, os.path.join(out_dir, fname))
                generated += 1
                pbar.update(1)

        pbar.close()
        report[class_id] = generated

    print("\n[REPORT] Gemini augmentation complete.")
    for cid, gen in report.items():
        print(f"Class {cid:02d}: Generated {gen} synthetic images")


if __name__ == "__main__":
    main()
