import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pyttsx3
import uuid
import requests

# currency conversion function using the ExchangeRate-API
def convert_currency(amount_mad):
    api_key = "3b4d6e2699c8fa31e68bfa8a"  
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/MAD"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        eur_rate = data['conversion_rates']['EUR']
        usd_rate = data['conversion_rates']['USD']
        eur_amount = amount_mad * eur_rate
        usd_amount = amount_mad * usd_rate
        return round(eur_amount, 2), round(usd_amount, 2)
    else:
        st.error("Failed to fetch live exchange rates. Using static rates.")
        eur_rate, usd_rate = 0.089, 0.095  # in case of failing use this rates
        return round(amount_mad * eur_rate, 2), round(amount_mad * usd_rate, 2)

# Load model
model = tf.keras.models.load_model('models/moroccan_currency_classifier.keras')

# detect class names from the dataset
train_dir = 'data/train_test_datasets/train'
class_names = sorted(os.listdir(train_dir))

# initialize pyttsx3 for speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# streamlit app 
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #00b300;
        font-size: 36px;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px #222;
    }
    .subheader {
        text-align: center;
        color: #ccc;
        margin-bottom: 30px;
    }
    .prediction-card {
        background-color: #333;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: none;
        color: white;
    }
    .prediction-header {
        color: #00b300;
        font-size: 20px;
        text-align: center;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .amount-display {
        display: flex;
        justify-content: space-around;
        margin: 10px 0;
    }
    .currency-amount {
        font-size: 18px;
        font-weight: bold;
        color: white;
    }
    .confidence-badge {
        background-color: #00b300;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 16px;
        margin: 10px auto;
        display: block;
        width: fit-content;
    }
    .top-prediction {
        background-color: #444;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 16px;
        color: white;
    }
    .top-prediction-1 {
        border-left: 5px solid #00b300;
    }
    .top-prediction-2, .top-prediction-3 {
        border-left: 5px solid #888;
    }
    .image-container img {
        border-radius: 8px;
        border: 2px solid #444;
    }
    .stAudio {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Moroccan Currency Detection App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload images of Moroccan banknotes or coins for instant recognition</p>', unsafe_allow_html=True)

# language selection for Speech
language = st.selectbox('Select Language for Speech', ['English', 'Arabic', 'French'])

# file uploader for multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    
    results_container = st.container()
    
    with results_container:
        # Process each image
        for i in range(0, len(uploaded_files), 2):
            cols = st.columns(2)  #
            
            for j in range(2):
                if i + j < len(uploaded_files):
                    idx = i + j
                    uploaded_file = uploaded_files[idx]
                    
                    with cols[j]:
                        with st.container():
                            image = Image.open(uploaded_file)
                            st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                            st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # preprocess image
                            image_resized = image.resize((224, 224))
                            image_array = np.array(image_resized) / 255.0
                            image_array = np.expand_dims(image_array, axis=0)
                            
                            # make predictions
                            predictions = model.predict(image_array)
                            predicted_class = class_names[np.argmax(predictions)]
                            confidence = np.max(predictions) * 100
                            
                            # Convert currency to EUR and USD
                            amount_mad = float(predicted_class.split()[0])
                            eur_amount, usd_amount = convert_currency(amount_mad)
                            
                            # Get top 3 predictions
                            top_3_indices = predictions[0].argsort()[-3:][::-1]
                            top_3_labels = [class_names[i] for i in top_3_indices]
                            top_3_conf = [predictions[0][i] * 100 for i in top_3_indices]
                            
                            # Speech output
                            if language == 'English':
                                speech_text = f"Predicted Denomination: {predicted_class}. Equivalent to {eur_amount} Euro and {usd_amount} USD. Confidence: {confidence:.2f}%."
                            elif language == 'Arabic':
                                speech_text = f"راك عطيت {predicted_class}، معادل {eur_amount} يورو و {usd_amount} دولار، وأنا متأكد ب {confidence:.1f}% فالمية"
                            elif language == 'French':
                                speech_text = f"Dénomination prédite: {predicted_class}. Équivalent à {eur_amount} Euro et {usd_amount} USD. Confiance: {confidence:.2f}%."
                            
                            # display prediction results with enhanced styling
                            st.markdown(f"""
                            <div class="prediction-card">
                                <div class="prediction-header">Predicted: {predicted_class}</div>
                                <div class="amount-display">
                                    <span class="currency-amount">{eur_amount} EUR</span>
                                    <span class="currency-amount">{usd_amount} USD</span>
                                </div>
                                <div class="confidence-badge">Confidence: {confidence:.2f}%</div>
                            """, unsafe_allow_html=True)
                            st.write("Top 3 predictions:")
                            for i in range(3):
                                st.write(f"- {top_3_labels[i]} MAD ({top_3_conf[i]:.2f}%)")
        
                            
                            # Create audio file
                            audio_filename = f"temp_audio_{uuid.uuid4()}.mp3"
                            engine.save_to_file(speech_text, audio_filename)
                            engine.runAndWait()
                            
                            # display audio player
                            st.audio(audio_filename, format="audio/mp3")
                            
                            # clean up temporary audio file
                            os.remove(audio_filename)
                            
                            
                            st.markdown("<hr style='margin: 30px 0; opacity: 0.3;'>", unsafe_allow_html=True)
else:
    st.info("Please upload images of Moroccan banknotes or coins to get started.")
