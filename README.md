# 🇲🇦 Moroccan Currency Detection App 

Une application intelligente basée sur l'intelligence artificielle (CNN) permettant de détecter les billets de banque marocains à partir d'une image. 

Développée avec **TensorFlow/Keras**, **OpenCV** et déployée via **Streamlit**, cette application est conçue pour assister les personnes malvoyantes ou intégrer des distributeurs automatiques intelligents.

---

## 🚀 Fonctionnalités

- 📸 Chargement d'une image contenant un billet marocain
- 🤖 Détection automatique à l'aide d'un modèle CNN entraîné
- 🎙️ Synthèse vocale du résultat en **Darija**, **Français**, **Anglais** ou **Arabe**
- 💱 Conversion automatique du dirham en **EUR** ou **USD**
- 📊 Affichage des 3 prédictions les plus probables avec scores de confiance

---

## 🛠️ Technologies utilisées

- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- gTTS (Google Text-to-Speech)
- NumPy, Pandas, etc.

---


---

## 🧪 Comment utiliser

### 1. Cloner le dépôt

git clone https://github.com/yahiabak/Moroccan-Currency-Detection-App-Powered-by-CNN-AI.git
cd Moroccan-Currency-Detection-App-Powered-by-CNN-AI

### 2.Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate   

pip install -r requirements.txt

streamlit run app.py






