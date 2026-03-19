# 🫁 Pneumonia X-Ray Classifier

> An industry-oriented machine learning system that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using traditional ML algorithms.

---

## 🔗 Live Demo

👉 **[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/Naseeya/pneumonia-classifier)**

Upload any chest X-ray image and get an instant diagnosis prediction!

---

## 📌 Project Overview

This project was built as part of the **Tamizhian Skill RISE 4.0 Virtual Internship** (Machine Learning & AI track).

Manual inspection of medical images is slow and error-prone. This project demonstrates how machine learning can automate chest X-ray classification to assist medical professionals in detecting pneumonia quickly and accurately.

---

## 📁 Dataset

- **Source:** [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** NORMAL, PNEUMONIA
- **Training samples:** 5,216 images
- **Test samples:** 624 images
- **Class distribution:** 1,341 Normal | 3,875 Pneumonia (imbalanced — handled using `class_weight='balanced'`)

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3 | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Pillow (PIL) | Image loading and preprocessing |
| Scikit-learn | ML model training and evaluation |
| XGBoost | Gradient boosting model |
| Matplotlib & Seaborn | Visualization |
| Gradio | Web UI for live predictions |
| Jupyter Notebook | Development environment |

---

## ⚙️ How It Works

```
Chest X-Ray Image
       ↓
Grayscale Conversion
       ↓
Resize to 64×64 pixels
       ↓
Normalize pixel values (0–1)
       ↓
Flatten to 4096 features
       ↓
SVM Classification Model
       ↓
NORMAL or PNEUMONIA
```

---

## 📊 Model Results

Three models were trained and compared:

| Model | Accuracy |
|-------|----------|
| SVM (RBF kernel) | **80.45%** ✅ Best |
| Random Forest | 76.76% |
| XGBoost | see notebook |

**SVM with RBF kernel** was selected as the final model due to its highest accuracy and strong performance on imbalanced medical data.

---

## 📈 Visualizations

The project generates the following output charts:

- `sample_xrays.png` — Sample images from both classes
- `confusion_matrix.png` — Model prediction breakdown
- `model_comparison.png` — Accuracy comparison across all 3 models
- `predictions.png` — Predictions on unseen test images (green = correct, red = wrong)

---

## 🚀 How to Run Locally

**1. Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-classifier.git
cd pneumonia-classifier
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost gradio Pillow joblib jupyter
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it as:
```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**4. Run the notebook**
```bash
jupyter notebook notebooks/pneumonia_classifier.ipynb
```

**5. Launch the UI**
```bash
python app.py
```

---

## 📂 Project Structure

```
pneumonia_classifier/
├── data/                        # Dataset (not uploaded — download from Kaggle)
├── notebooks/
│   └── pneumonia_classifier.ipynb   # Main project notebook
├── outputs/
│   ├── sample_xrays.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── predictions.png
├── app.py                       # Gradio web app
├── svm_model.pkl                # Saved trained model
├── label_encoder.pkl            # Saved label encoder
└── README.md                    # This file
```

---

## 💡 Key Learnings

- Image preprocessing and normalization for ML pipelines
- Handling imbalanced datasets using `class_weight='balanced'`
- Comparing multiple ML algorithms on the same problem
- Building and deploying a real-time ML web application using Gradio
- Deploying ML apps on Hugging Face Spaces

---

## 👩‍💻 Author

**Naseeya Begum**
Tamizhian Skill RISE 4.0 Internship — Machine Learning & AI
