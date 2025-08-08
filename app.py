import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import json
import os

# Configuration de la page
st.set_page_config(
    page_title="Classificateur de Chiens",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .history-item {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
<div class="main-header">
    <h1>🐕 Classificateur de Chiens</h1>
    <p>Uploadez une photo de chien et découvrez sa race !</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'history' not in st.session_state:
    st.session_state.history = []

# Import des utilitaires du modèle
from utils.model_utils import (
    load_model, 
    classify_dog_breed_with_model, 
    simulate_classification,
    get_model_info,
    validate_image
)

# Chargement du modèle au démarrage
@st.cache_resource
def load_dog_model():
    """Charge le modèle de classification de chiens"""
    return load_model()

# Initialisation du modèle
model = load_dog_model()

# Fonction de classification avec gestion d'erreurs
def classify_dog_breed(image):
    """
    Classification de race de chien avec le vrai modèle ou simulation
    """
    # Validation de l'image
    is_valid, message = validate_image(image)
    if not is_valid:
        st.error(f"❌ {message}")
        return {}
    
    # Utiliser le vrai modèle si disponible, sinon simulation
    if model is not None:
        try:
            results = classify_dog_breed_with_model(image, model)
            if results:
                return results
            else:
                st.warning("⚠️ Erreur lors de la classification avec le modèle. Utilisation de la simulation.")
                return simulate_classification(image)
        except Exception as e:
            st.error(f"❌ Erreur du modèle: {str(e)}")
            st.info("🔄 Utilisation de la simulation...")
            return simulate_classification(image)
    else:
        st.info("🤖 Utilisation du modèle de simulation (vrai modèle non disponible)")
        return simulate_classification(image)

# Fonction pour sauvegarder l'historique
def save_to_history(image, results, timestamp):
    # Convertir l'image en base64 pour le stockage
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    history_item = {
        'timestamp': timestamp,
        'image': img_str,
        'results': results
    }
    
    st.session_state.history.append(history_item)
    
    # Limiter l'historique à 10 éléments
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)

# Layout principal
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload de l'image")
    
    # Zone d'upload
    uploaded_file = st.file_uploader(
        "Choisissez une image de chien...",
        type=['png', 'jpg', 'jpeg'],
        help="Formats acceptés : PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Afficher l'image uploadée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image uploadée", use_column_width=True)
        
        # Bouton de classification
        if st.button("🔍 Analyser la race", type="primary"):
            with st.spinner("Analyse en cours..."):
                # Classification
                results = classify_dog_breed(image)
                
                # Sauvegarder dans l'historique
                save_to_history(image, results, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
                # Afficher les résultats
                st.success("✅ Analyse terminée !")
                
                # Créer un DataFrame pour l'affichage
                df_results = pd.DataFrame([
                    {'Race': breed, 'Confiance (%)': confidence}
                    for breed, confidence in results.items()
                ])
                
                # Afficher les résultats avec des barres de progression
                st.markdown("### 📊 Résultats de classification")
                
                for breed, confidence in results.items():
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    
                    with col_a:
                        st.write(f"**{breed}**")
                    
                    with col_b:
                        st.write(f"{confidence}%")
                    
                    with col_c:
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()

with col2:
    st.markdown("### 📋 Historique des analyses")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Analyse du {item['timestamp']}"):
                col_img, col_results = st.columns([1, 1])
                
                with col_img:
                    # Afficher l'image
                    img_data = base64.b64decode(item['image'])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, caption="Image analysée", use_column_width=True)
                
                with col_results:
                    st.markdown("**Résultats :**")
                    for breed, confidence in list(item['results'].items())[:5]:  # Top 5
                        st.write(f"• {breed}: {confidence}%")
                
                # Bouton pour supprimer de l'historique
                if st.button(f"🗑️ Supprimer", key=f"delete_{i}"):
                    st.session_state.history.pop(len(st.session_state.history) - 1 - i)
                    st.rerun()
    else:
        st.info("Aucune analyse effectuée pour le moment. Uploadez une image pour commencer !")

# Sidebar avec des informations supplémentaires
with st.sidebar:
    st.markdown("### ℹ️ Informations")
    st.markdown("""
    **Comment utiliser l'application :**
    1. Uploadez une image de chien
    2. Cliquez sur "Analyser la race"
    3. Consultez les résultats
    4. Retrouvez l'historique à droite
    
    **Fonctionnalités :**
    - Classification automatique des races
    - Pourcentages de confiance
    - Historique des analyses
    - Interface moderne et intuitive
    """)
    
    # Informations sur le modèle
    st.markdown("### 🤖 État du modèle")
    model_info = get_model_info()
    
    if model_info["model_loaded"]:
        st.success("✅ Modèle chargé avec succès")
        st.metric("Taille du fichier", f"{model_info['file_size'] / 1024 / 1024:.1f} MB")
        if "model_type" in model_info:
            st.metric("Type de modèle", model_info["model_type"])
    else:
        st.warning("⚠️ Modèle non disponible")
        st.info("Utilisation du modèle de simulation")
    
    st.metric("Races supportées", model_info["breeds_count"])
    
    st.markdown("### 📈 Statistiques")
    if st.session_state.history:
        st.metric("Analyses effectuées", len(st.session_state.history))
        st.metric("Dernière analyse", st.session_state.history[-1]['timestamp'])
    else:
        st.metric("Analyses effectuées", 0)
    
    # Bouton pour vider l'historique
    if st.session_state.history and st.button("🗑️ Vider l'historique"):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🐕 Classificateur de Chiens - Développé avec Streamlit</p>
</div>
""", unsafe_allow_html=True)
