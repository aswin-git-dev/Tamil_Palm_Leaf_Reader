import streamlit as st
st.set_page_config(page_title="Tamil OCR Dashboard", layout="wide", initial_sidebar_state="collapsed")
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2  
import torch
from datetime import datetime

# Import custom processing functions (assuming these files exist and are correct)
from main1 import process_letter as process_leaf_letter
from main2 import process_word as process_leaf_word
from main3 import process_sentence as process_leaf_sentence
from main4 import process_paragraph as process_leaf_paragraph

from PIL import ImageFont, ImageDraw, Image as PILImage

# Import for Gemini API
import google.generativeai as genai

# Load environment variables for local development (if .env file exists)
# For Streamlit Community Cloud, use st.secrets instead
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # dotenv not installed, assume environment variables are set directly or via st.secrets

# Configure the Gemini API client ONCE at the top
# This prioritizes Streamlit secrets (for cloud deployment) then environment variables (for local .env)
GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("Gemini API key not found. Please set the GOOGLE_API_KEY environment variable (for local development via .env file) or add it to Streamlit secrets (for cloud deployment). Gemini enhancement will not be available.")


def enhance_tamil_with_gemini(text_to_enhance):
    """
    Enhances Tamil text (grammar correction, modernization) using Gemini/PaLM API.
    Attempts different models if the primary one fails.
    """
    if not GEMINI_API_KEY:
        return "⚠️ Error: Gemini API key is not configured. Cannot enhance text."
        
    if not text_to_enhance.strip():
        return ""

    # Define a list of models to try, in order of preference
    # 'models/' prefix is generally handled by the library, but explicit naming can sometimes help
    models_to_try = [
        'gemini-2.5-flash',
              # Legacy PaLM 2 model (good fallback if Gemini isn't available)
    ]

    enhanced_content = "⚠️ Error: No suitable Gemini/PaLM model found or all enhancement attempts failed."
    model_used = "None"

    for model_name in models_to_try:
        try:
            st.info(f"Attempting to use model: **{model_name}** for enhancement...")
            model = genai.GenerativeModel(model_name)

            # A more specific prompt for enhancement/grammar correction
            prompt = f"""This is an ancient Tamil text extracted from a palm-leaf manuscript, possibly from Nāladiyār or a similar Sangam-era or medieval text. The OCR output may contain minor script errors, older word forms, or broken compounds. Please correct and enhance the Tamil while preserving classical meaning, poetic style, and metaphors. If it's from a known work, try to reconstruct it closer to its original verse.

Original Tamil Text:
{text_to_enhance}

Enhanced Tamil Text:"""

            response = model.generate_content(prompt)
            
            if response.parts:
                enhanced_content = "".join([part.text for part in response.parts])
            elif hasattr(response, 'text'): # Fallback for older library versions or simpler responses
                enhanced_content = response.text
            else:
                enhanced_content = f"⚠️ Error: Gemini response format unexpected for model {model_name}. Please check API response structure."
            
            # If we successfully get a response without an exception, break the loop
            model_used = model_name
            break # Exit the loop if a model worked
            
        except Exception as e:
            st.warning(f"Failed with model **{model_name}**: {str(e)}")
            # Continue to the next model in the list
    
    if model_used != "None":
        st.success(f"Successfully enhanced text using model: **{model_used}**")
    
    return enhanced_content.strip()


def draw_unicode_text(image_cv2, segments, font_path="fonts/NirmalaB.ttf"):
    """Draw Tamil Unicode labels using PIL instead of cv2.putText"""
    # Convert to PIL
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Load font (adjust size as needed)
    font_size = max(18, int(image_cv2.shape[0] * 0.02)) # Dynamic font size
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        st.warning(f"Font file not found at {font_path}. Using default font.")
        font = ImageFont.load_default()


    for seg in segments:
        x, y, w_box, h_box = seg['bbox']
        char_label = seg['char']
        confidence = seg['confidence']
        color = (0, 255, 0) if confidence >= 0.5 else (255, 165, 0) # green or orange

        # Draw box (on image_cv2 still for color control)
        # Convert PIL color to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0]) # PIL (R,G,B) to OpenCV (B,G,R)
        cv2.rectangle(image_cv2, (x, y), (x + w_box, y + h_box), bgr_color, 2)

        # Draw text with PIL
        # Ensure coordinates are within image bounds for text
        text_x = x
        text_y = y - font_size - 5 # Position text above the box
        if text_y < 0:
            text_y = y + h_box + 5 # If not enough space above, put below

        draw.text((text_x, text_y), char_label, font=font, fill=color)

    # Convert back to OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_bounding_boxes_only(image_cv2, segments, box_thickness=2):
    image_copy = image_cv2.copy()
    for seg in segments:
        x, y, w_box, h_box = seg['bbox']
        confidence = seg['confidence']
        color = (0, 255, 0) if confidence >= 0.5 else (255, 165, 0)
        cv2.rectangle(image_copy, (x, y), (x + w_box, y + h_box), color, box_thickness)
    return image_copy


# Label map (assuming this is used by your process functions)
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
    66: 'சீ', 67: 'ணீ', 68: 'ஆ', 69: 'ளு', 70: 'இ', 71: 'ை', 72: 'ர', 73: 'ந', 74: 'ஒ', 75: 'ஃ'
}

def save_segmented_characters(segments, label_map, base_folder="segmented_dataset", image_size=(448, 448)):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_base = os.path.join(base_folder, f"dataset_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    for seg in segments:
        label_id = seg.get("label")  # Numeric label
        image = seg.get("padded")
        if image is None or label_id is None:
            continue

        folder_name = str(label_id)  # Save into folder "0", "1", ..., "71"
        folder_path = os.path.join(output_base, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        image_resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        unique_time = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = os.path.join(folder_path, f"{unique_time}.png")
        cv2.imwrite(filename, image_resized)

    return output_base



def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add semi-transparent palm leaf image as background
st.markdown(
    '''<style>
    body, .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%),
            url("images/vatta_button.png") no-repeat center center fixed;
        background-size: cover, contain;
        background-blend-mode: overlay;
    }
    .main {
        opacity: 0.97;
    }
    </style>''', unsafe_allow_html=True
)

load_css('styles.css')

if "script_type" not in st.session_state:
    st.session_state.script_type = "Palm Leaf (Vatta Ezhuthu)"
if "pred_type" not in st.session_state:
    st.session_state.pred_type = None

st.markdown("<div class='main-title floating-element'>Ancient Tamil Palm Leaf character recognition using Transformers<br><span style='font-size: 22px; font-weight: 400; opacity: 0.9;'>Palm Leaf OCR powered by deep learning</span></div>", unsafe_allow_html=True)

st.markdown("<div class='content-container'>", unsafe_allow_html=True)


st.markdown("<h3 class='section-header'> Select Prediction Mode</h3>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4, gap="medium")
with col1:
    if st.button(" Letter Recognition", key="letter_btn"):
        st.session_state.pred_type = "Letter"
        st.rerun()
with col2:
    if st.button(" Word Detection", key="word_btn"):
        st.session_state.pred_type = "Word"
        st.rerun()
with col3:
    if st.button(" Sentence Analysis", key="sentence_btn"):
        st.session_state.pred_type = "Sentence"
        st.rerun()
with col4:
    if st.button(" Paragraph OCR", key="paragraph_btn"):
        st.session_state.pred_type = "Paragraph"
        st.rerun()

pred_type = st.session_state.pred_type
if pred_type:
    st.info(f" Prediction Mode: *{pred_type}*")


if pred_type:
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3 class='section-header'> Upload Your Images</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(" Choose image files for OCR processing", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.markdown(f"### ️ File: {uploaded_file.name}")
            st.image(Image.open(file_path), caption=" Uploaded Image", use_container_width=True)

            with st.spinner(" Processing the image with ML Model..."):
                st.markdown("""
                <div class='progress-indicator'>
                    <div class='progress-bar'></div>
                </div>
                """, unsafe_allow_html=True)

                result = {
                    "Letter": process_leaf_letter,
                    "Word": process_leaf_word,
                    "Sentence": process_leaf_sentence,
                    "Paragraph": process_leaf_paragraph,
                }[pred_type](file_path)


            if result:
                st.markdown("---")
                st.markdown("<h3 class='section-header'> Preprocessing Pipeline</h3>", unsafe_allow_html=True)
                try:
                    imgs = result.get("preprocessing", []) 
                    labels = [" Original", "Grayscale", "Enhanced", "Binarized"]
                    cols = st.columns(min(len(imgs), 4)) 
                    for i, img in enumerate(imgs[:4]):
                        if img is not None:
                            with cols[i]:
                                st.image(img, caption=labels[i], use_container_width=True)
                except Exception as e:
                    st.warning(f" Some preprocessing images couldn't be loaded: {e}")

                if pred_type != "Letter" and result.get("valid_segments"):
                    st.markdown("---")
                    st.markdown("<h3 class='section-header'> Character Detection Map</h3>", unsafe_allow_html=True)
                    original_img_for_drawing = result['preprocessing'][0].copy() if result['preprocessing'] and len(result['preprocessing']) > 0 else cv2.imread(file_path)
                    labeled_img = draw_unicode_text(original_img_for_drawing.copy(), result['valid_segments'], font_path="fonts/NirmalaB.ttf")
                    labeled_rgb = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)
                    st.image(labeled_rgb, caption="Character + Label Mapping", use_container_width=True)
                    boxed_img = draw_bounding_boxes_only(original_img_for_drawing.copy(), result['valid_segments'])
                    boxed_rgb = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
                    st.image(boxed_rgb, caption="Character Bounding Boxes ", use_container_width=True)

                st.markdown("---")
                st.markdown("<h3 class='section-header'> Recognition Results</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white; padding: 25px; border-radius: 16px; margin: 20px 0;
                                    box-shadow: 0 12px 24px rgba(0,0,0,0.15);'>
                    <h4 style='margin: 0 0 15px 0; font-size: 24px;'> Extracted Text (Raw OCR)</h4>
                    <p style='font-size: 20px; font-weight: 500; margin: 0;
                                    background: rgba(255,255,255,0.1); padding: 15px;
                                    border-radius: 8px; backdrop-filter: blur(10px);'>
                        {result['complete_line']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Detected Words:** {' • '.join(result['words'])}")
                st.markdown(f"**Character Count:** {len(result['complete_line'])}")
                predicted_text = result.get('complete_line', '').strip()
                if predicted_text and GEMINI_API_KEY:
                    st.markdown("---")
                    st.markdown("<h3 class='section-header'> Gemini AI: Enhanced Tamil Version</h3>", unsafe_allow_html=True)
                    if st.button("Enhance Tamil Text with Gemini", key=f"gemini_enhance_btn_{uploaded_file.name}"):
                        with st.spinner("Enhancing Tamil text with Gemini AI... This might take a moment."):
                            enhanced_text = enhance_tamil_with_gemini(predicted_text)
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #FF6B6B 0%, #FFB647 100%);
                                        color: white; padding: 25px; border-radius: 16px; margin: 20px 0;
                                        box-shadow: 0 12px 24px rgba(0,0,0,0.15);'>
                                <h4 style='margin: 0 0 15px 0; font-size: 24px;'> Enhanced Text</h4>
                                <p style='font-size: 20px; font-weight: 500; margin: 0;
                                            background: rgba(255,255,255,0.1); padding: 15px;
                                            border-radius: 8px; backdrop-filter: blur(10px);'>
                                    {enhanced_text}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                elif not GEMINI_API_KEY:
                    st.warning("Please configure your Google API Key to enable Gemini features (text enhancement). Refer to previous instructions for setting up .env or Streamlit secrets.")
                if pred_type != "Letter":
                    st.markdown("---")
                    st.markdown("<h3 class='section-header'> Character Segments</h3>", unsafe_allow_html=True)
                    if result.get("word_segments"):
                        for idx, word in enumerate(result["word_segments"]):
                            word_text, segments = word
                            st.markdown(f"**Word {idx+1}:** {word_text}")
                            for i in range(0, len(segments), 10):
                                row_segs = segments[i:i+10]
                                cols = st.columns(len(row_segs))
                                for j, seg in enumerate(row_segs):
                                    with cols[j]:
                                        if 'padded' in seg and seg['padded'] is not None:
                                            st.image(seg['padded'], caption=f"{seg['char']} ({seg['confidence']:.2f})", width=90)
                                        else:
                                            st.write(f"No image for {seg['char']}")
                    else:
                        st.info("No word segments found for display.")
                if result.get("valid_segments"):
                    output_dir = save_segmented_characters(result['valid_segments'], label_map)
                    st.success(f"📁 Character dataset saved to: {output_dir}")
                else:
                    st.info("No valid character segments to save.")
                st.markdown("---")
                st.download_button(
                    "📥 Download Extracted Text", 
                    data=predicted_text,
                    file_name=f"tamil_ocr_result_palm_leaf_{uploaded_file.name}.txt", 
                    mime="text/plain",
                    key=f"download_{uploaded_file.name}_raw"
                )
            else:
                st.error(" No valid text could be extracted from the image. Please try with a clearer image or different settings.")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
