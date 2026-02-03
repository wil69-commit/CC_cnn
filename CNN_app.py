
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Détecteur de Fractures", layout="centered")

@st.cache_resource
def load_my_model():
    # Charge ton modèle sauvegardé (H5 ou SavedModel)
    return tf.keras.models.load_model('model_fracture.keras')

model = load_my_model()

st.title("🏥 Diagnostic d'Imagerie Médicale")
st.subheader("Classification de fractures par Deep Learning")
st.write("Téléchargez une radiographie pour vérifier la présence d'une fracture.")

# Zone de téléchargement
uploaded_file = st.file_uploader("Choisir une image de radiographie...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_container_width=True)
    
    st.write("🔄 Analyse en cours...")
    
    # Prétraitement de l'image (doit être identique à ton entraînement)
    # Si tu as utilisé 150x150 dans ton notebook, garde 150 ici
    img = image.resize((150, 150)) 
    img_array = np.array(img)
    
    # Normalisation (si tu as utilisé ResNet/VGG avec preprocess_input, adapte ici)
    if img_array.shape[-1] == 4: # Supprimer le canal alpha si PNG
        img_array = img_array[:,:,:3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Si tu n'as pas utilisé preprocess_input

    # Prédiction
prediction = model.predict(img_array)
score = prediction[0][0] # Proba d'être dans la classe 1 ('not fractured')

# Logique corrigée basée sur {'fractured': 0, 'not fractured': 1}
if score < 0.5:
    # Si le score est proche de 0, c'est la classe 'fractured'
    st.error(f"⚠️ **Résultat : FRACTURE DÉTECTÉE** (Confiance : {1-score:.2%})")
else:
    # Si le score est proche de 1, c'est la classe 'not fractured'
    st.success(f"✅ **Résultat : PAS DE FRACTURE** (Confiance : {score:.2%})")

st.info("Note : Cette application est un outil d'aide à la décision et ne remplace pas l'avis d'un radiologue.")

