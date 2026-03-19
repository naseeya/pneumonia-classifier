import gradio as gr
import numpy as np
from PIL import Image
import joblib
import os

# Load saved model
svm_model = joblib.load('svm_model.pkl')
le = joblib.load('label_encoder.pkl')

def predict_xray(image):
    # Preprocess exactly like training
    img = Image.fromarray(image).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    features = img_array.flatten().reshape(1, -1)
    
    # Predict
    prediction_enc = svm_model.predict(features)[0]
    prediction = le.inverse_transform([prediction_enc])[0]
    
    # Confidence
    decision = svm_model.decision_function(features)[0]
    confidence = round(abs(decision) / (abs(decision) + 1) * 100, 2)
    
    if prediction == 'PNEUMONIA':
        result = f"🔴 PNEUMONIA DETECTED\nConfidence: {confidence}%\n\nPlease consult a doctor immediately."
    else:
        result = f"🟢 NORMAL\nConfidence: {confidence}%\n\nNo signs of pneumonia detected."
    
    return result

app = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(label="Upload Chest X-Ray"),
    outputs=gr.Textbox(label="Diagnosis Result", lines=4),
    title="🫁 Pneumonia X-Ray Classifier",
    description="Upload a chest X-ray image to detect whether it shows signs of Pneumonia or is Normal. Powered by SVM + scikit-learn.",
    theme="soft"
)

app.launch()