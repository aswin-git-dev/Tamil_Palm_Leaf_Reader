# Ancient-Tamil-Palm-Leaf-character-recognition-using-Transformers

Here is the complete dataset => https://drive.google.com/file/d/1GIgTY7WTzwa2qkfgMlDHKwr56LLhx-T7/view?usp=sharing

Here is the saved fine tuned model files of Trocr and trained model of CNN file => https://drive.google.com/file/d/1ADxISjBYB9ywgwLJlqDaua1Nq6LZPZ6K/view?usp=sharing

Here’s a step-by-step guide to execute all programs in this project, starting from Python package installation to running the models and the Streamlit Web app. 

---

## How to Run This Project

### 1. Clone the Repository
```powershell
git clone https://github.com/anonymous-git-hub/Ancient-Tamil-Palm-Leaf-character-recognition-using-Transformers.git
cd "Ancient Tamil Palm Leaf character recognition using Transformers"
```

### 2. Install Python & Create Virtual Environment
Make sure you have Python 3.8+ installed.  
Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Required Packages
Install all necessary packages for model training, testing, and the Streamlit app:
```powershell
pip install torch torchvision transformers tensorflow scikit-learn opencv-python streamlit pillow matplotlib seaborn pandas google-generativeai python-dotenv
```
> If you use Jupyter notebooks, also install:
```powershell
pip install notebook ipykernel
```

### 4. Download Dataset & Models
- Download the dataset from the link in the README and extract it to the DATASET folder.
- Download the trained model files (CNN and TrOCR) and place them in the appropriate folders (e.g., `TR OCR` for TrOCR, as referenced in predict_leaf.py).

### 5. Train Models (Optional)
If you want to retrain models:
- Open and run the notebooks in `MODEL TRAINING/`:
  - CNN_MODEL_TRAINING_vattaeluthu_tamil.ipynb
  - TROCR_VATTAELUTHU_FINETUING_TESTING.ipynb
- Follow the notebook instructions for training and saving models.

### 6. Test Models
Run the test scripts to evaluate the models:
```powershell
python "CNN Trained Model Testing/test letter.py"
python "CNN Trained Model Testing/test word.py"
```
Or use the metrics notebook in `TESTING AND METRICS/` for detailed evaluation.

### 7. Run the Streamlit App
To launch the web interface:
```powershell
streamlit run "STREAMLIT APP/app.py"
```
- The app will open in your browser. You can upload palm leaf images and get predictions.

### 8. Additional Scripts
- Synthetic dataset generation: Run scripts in Dataset_synthetic_generation_CODE as needed.

---

**Note:**  
- Ensure all model files and datasets are in the correct folders as referenced by the scripts.
- For GPU acceleration, install CUDA and compatible PyTorch/TensorFlow versions.

---

