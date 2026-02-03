import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détecteur de Fractures VGG16", layout="centered")

@st.cache_resource
def load_my_model():
    # Charge ton modèle sauvegardé
    return tf.keras.models.load_model('model_VGG16.keras')

model = load_my_model()

st.title("🏥 Diagnostic d'Imagerie Médicale VGG16")
st.subheader("Classification de fractures par Deep Learning")
st.info("Note : Cet outil est une démonstration technologique et ne remplace pas un avis médical.")

# Zone de téléchargement
uploaded_file = st.file_uploader("Choisir une image de radiographie...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_container_width=True)
    
    with st.spinner("🔄 Analyse en cours..."):
        # 1. Prétraitement
        # Conversion en RGB au cas où l'image est en niveaux de gris ou possède un canal Alpha
        img = image.convert('RGB')
        img = img.resize((150, 150)) 
        
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        # 2. Prédiction (Bien indenté à l'intérieur du bloc IF)
        prediction = model.predict(img_array)
        score = prediction[0][0] 

        # 3. Logique d'affichage
        st.divider()
        if score < 0.5:
            confiance = (1 - score)
            st.error(f"### ⚠️ Résultat : FRACTURE DÉTECTÉE")
            st.metric(label="Indice de confiance", value=f"{confiance:.2%}")
        else:
            confiance = score
            st.success(f"### ✅ Résultat : PAS DE FRACTURE")
            st.metric(label="Indice de confiance", value=f"{confiance:.2%}")
