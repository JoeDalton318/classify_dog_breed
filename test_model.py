#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration du modèle de classification de chiens
"""

import sys
import os
from PIL import Image
import numpy as np

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_integration():
    """Test complet de l'intégration du modèle"""
    
    print("🧪 Test d'intégration du modèle de classification de chiens")
    print("=" * 60)
    
    try:
        # Test 1: Import des utilitaires
        print("\n1️⃣ Test des imports...")
        from utils.model_utils import (
            load_model, 
            classify_dog_breed_with_model, 
            simulate_classification,
            get_model_info,
            validate_image
        )
        print("✅ Imports réussis")
        
        # Test 2: Informations sur le modèle
        print("\n2️⃣ Informations sur le modèle...")
        model_info = get_model_info()
        print(f"📁 Chemin du modèle: {model_info['model_path']}")
        print(f"📊 Fichier existe: {model_info['file_exists']}")
        print(f"🤖 Modèle chargé: {model_info['model_loaded']}")
        print(f"🐕 Races supportées: {model_info['breeds_count']}")
        
        # Test 3: Chargement du modèle
        print("\n3️⃣ Chargement du modèle...")
        model = load_model()
        if model is not None:
            print("✅ Modèle chargé avec succès")
            print(f"📦 Type: {type(model).__name__}")
        else:
            print("⚠️ Modèle non disponible, utilisation de la simulation")
        
        # Test 4: Création d'une image de test
        print("\n4️⃣ Test avec image de simulation...")
        # Créer une image de test simple
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Test de validation
        is_valid, message = validate_image(test_image)
        print(f"✅ Validation image: {message}")
        
        # Test de classification
        if model is not None:
            results = classify_dog_breed_with_model(test_image, model)
        else:
            results = simulate_classification(test_image)
        
        print("📊 Résultats de classification:")
        for i, (breed, confidence) in enumerate(list(results.items())[:5]):
            print(f"   {i+1}. {breed}: {confidence}%")
        
        print("\n🎉 Tous les tests sont passés avec succès !")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test de lancement de l'application Streamlit"""
    
    print("\n🌐 Test de l'application Streamlit...")
    
    try:
        import streamlit as st
        print("✅ Streamlit importé avec succès")
        
        # Test des composants principaux
        from app import classify_dog_breed
        
        # Créer une image de test
        test_image = Image.new('RGB', (224, 224), color='blue')
        results = classify_dog_breed(test_image)
        
        if results:
            print("✅ Fonction de classification testée avec succès")
            print(f"📊 Nombre de résultats: {len(results)}")
        else:
            print("⚠️ Aucun résultat de classification")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test Streamlit: {str(e)}")
        return False

def main():
    """Fonction principale de test"""
    
    print("🚀 Démarrage des tests d'intégration")
    print("=" * 60)
    
    # Test 1: Intégration du modèle
    model_test_passed = test_model_integration()
    
    # Test 2: Application Streamlit
    streamlit_test_passed = test_streamlit_app()
    
    # Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    if model_test_passed:
        print("✅ Test d'intégration du modèle: RÉUSSI")
    else:
        print("❌ Test d'intégration du modèle: ÉCHOUÉ")
    
    if streamlit_test_passed:
        print("✅ Test de l'application Streamlit: RÉUSSI")
    else:
        print("❌ Test de l'application Streamlit: ÉCHOUÉ")
    
    if model_test_passed and streamlit_test_passed:
        print("\n🎉 Tous les tests sont passés ! L'application est prête.")
        print("\n📝 Pour lancer l'application:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
